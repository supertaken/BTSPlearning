from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


IMAGE_DIMENSION = 1024
ACTIVE_BITS = 100
CODING_RATE = ACTIVE_BITS / IMAGE_DIMENSION
HNN_THRESHOLD = 0.005
CONNECTION_PROBABILITY = 0.8
PEOPLE_SEED = 42
CONNECTION_SEED = 42
OCCLUSION_SEED = 20260613
CAPACITY_STEP = 5
MAX_CAPACITY = 2000
MASK_MIN_PERCENT = 0
MASK_MAX_PERCENT = 100
MASK_STEP_PERCENT = 1
LEARNING_PASSES = 1
RECALL_UPDATES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-capacity", type=int, default=MAX_CAPACITY)
    parser.add_argument("--capacity-step", type=int, default=CAPACITY_STEP)
    parser.add_argument("--mask-step", type=int, default=MASK_STEP_PERCENT)
    parser.add_argument(
        "--rate-batch",
        type=int,
        default=8,
        help="Number of masking rates evaluated together on the GPU.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Clean database not found: {args.input_dir}")
    if args.max_capacity < args.capacity_step:
        raise ValueError("max-capacity must be at least capacity-step")
    if args.max_capacity % args.capacity_step != 0:
        raise ValueError("max-capacity must be divisible by capacity-step")
    if not 1 <= args.mask_step <= 100:
        raise ValueError("mask-step must be in [1, 100]")
    if args.rate_batch <= 0:
        raise ValueError("rate-batch must be positive")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def load_binary_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if image.shape != (32, 32):
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_NEAREST)
    return (image < 128).astype(np.uint8).reshape(-1)


def load_clean_database(
    input_dir: Path,
    num_people: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    eligible: list[tuple[Path, list[Path]]] = []
    for identity_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        if images:
            eligible.append((identity_dir, images))

    if len(eligible) < num_people:
        raise RuntimeError(
            f"Database has only {len(eligible)} usable identities; "
            f"{num_people} are required"
        )

    rng = random.Random(seed)
    selected = rng.sample(eligible, num_people)
    patterns: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for order, (identity_dir, images) in enumerate(selected, start=1):
        image_path = rng.choice(images)
        patterns.append(load_binary_image(image_path))
        records.append(
            {
                "Database_Order": order,
                "Identity": identity_dir.name,
                "Image_Name": image_path.name,
            }
        )

    return np.asarray(patterns, dtype=np.uint8), records


def encode_top_k(
    patterns: np.ndarray,
    mean_face: np.ndarray,
    active_bits: int,
) -> np.ndarray:
    scores = np.asarray(patterns, dtype=np.float32) - mean_face
    top_indices = np.argpartition(scores, -active_bits, axis=1)[:, -active_bits:]
    codes = np.zeros(scores.shape, dtype=np.uint8)
    codes[np.arange(len(scores))[:, None], top_indices] = 1
    return codes


def build_occlusion_orders(num_people: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_keys = rng.random((num_people, IMAGE_DIMENSION), dtype=np.float32)
    return np.argsort(random_keys, axis=1).astype(np.int16)


def precompute_masked_codes(
    clean_pixels: np.ndarray,
    mean_face: np.ndarray,
    occlusion_orders: np.ndarray,
    mask_rates: np.ndarray,
) -> np.ndarray:
    num_people = len(clean_pixels)
    all_codes = np.empty(
        (len(mask_rates), num_people, IMAGE_DIMENSION),
        dtype=np.uint8,
    )
    rows = np.arange(num_people)[:, None]
    for rate_index, rate in enumerate(mask_rates):
        masked = clean_pixels.copy()
        num_masked = int(round(IMAGE_DIMENSION * float(rate) / 100.0))
        if num_masked:
            masked[rows, occlusion_orders[:, :num_masked]] = 0
        all_codes[rate_index] = encode_top_k(masked, mean_face, ACTIVE_BITS)
        if rate_index % 10 == 0 or rate_index == len(mask_rates) - 1:
            print(
                f"[encoding] rate {int(rate):3d}% "
                f"({rate_index + 1}/{len(mask_rates)})"
            )
    return all_codes


def build_connection_mask(device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(CONNECTION_SEED)
    upper = np.triu(
        rng.random((IMAGE_DIMENSION, IMAGE_DIMENSION))
        < CONNECTION_PROBABILITY,
        k=1,
    )
    symmetric = (upper | upper.T).astype(np.float32)
    return torch.from_numpy(symmetric).to(device=device)


@torch.inference_mode()
def evaluate_grid(
    clean_codes_np: np.ndarray,
    masked_codes_np: np.ndarray,
    capacities: np.ndarray,
    mask_rates: np.ndarray,
    device: torch.device,
    rate_batch: int,
) -> np.ndarray:
    accuracy = np.empty(
        (len(capacities), len(mask_rates)),
        dtype=np.float32,
    )

    clean_codes = torch.from_numpy(clean_codes_np.astype(np.float32)).to(device)
    centered = clean_codes - CODING_RATE
    connection_mask = build_connection_mask(device)
    scatter = torch.zeros(
        (IMAGE_DIMENSION, IMAGE_DIMENSION),
        dtype=torch.float32,
        device=device,
    )
    previous_capacity = 0
    start_time = time.perf_counter()

    for capacity_index, capacity_value in enumerate(capacities):
        capacity = int(capacity_value)
        new_patterns = centered[previous_capacity:capacity]
        scatter.addmm_(new_patterns.T, new_patterns)
        previous_capacity = capacity

        weights = (scatter / IMAGE_DIMENSION) * connection_mask
        references = clean_codes[:capacity]
        reference_sums = references.sum(dim=1).unsqueeze(0)
        truth = torch.arange(capacity, device=device).unsqueeze(0)

        for rate_start in range(0, len(mask_rates), rate_batch):
            rate_end = min(rate_start + rate_batch, len(mask_rates))
            query_np = masked_codes_np[rate_start:rate_end, :capacity]
            query = torch.from_numpy(
                np.ascontiguousarray(query_np.reshape(-1, IMAGE_DIMENSION)).astype(
                    np.float32
                )
            ).to(device)

            recalled = (query @ weights > HNN_THRESHOLD).to(torch.float32)
            distances = (
                recalled.sum(dim=1, keepdim=True)
                + reference_sums
                - 2.0 * (recalled @ references.T)
            )
            predictions = distances.argmin(dim=1).reshape(rate_end - rate_start, capacity)
            batch_accuracy = (predictions == truth).float().mean(dim=1) * 100.0
            accuracy[capacity_index, rate_start:rate_end] = (
                batch_accuracy.cpu().numpy()
            )

            del query, recalled, distances, predictions

        elapsed = time.perf_counter() - start_time
        completed = capacity_index + 1
        eta = elapsed / completed * (len(capacities) - completed)
        print(
            f"[evaluate] capacity {capacity:4d} "
            f"({completed}/{len(capacities)}), "
            f"0%={accuracy[capacity_index, 0]:6.2f}%, "
            f"100%={accuracy[capacity_index, -1]:6.2f}%, "
            f"elapsed={elapsed / 60:.1f} min, ETA={eta / 60:.1f} min"
        )

    return accuracy


def experiment_parameters(
    args: argparse.Namespace,
    device: torch.device,
    elapsed_seconds: float,
) -> pd.DataFrame:
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    parameters = {
        "Neuron_Count": IMAGE_DIMENSION,
        "Image_Size": "32 x 32",
        "Encoder": "Mean-face subtraction + fixed Top-K",
        "Mean_Face_Fit_Size": args.max_capacity,
        "Fixed_Active_Bits": ACTIVE_BITS,
        "Coding_Rate": CODING_RATE,
        "HNN_Learning_Rule": "Sparse binary covariance Hopfield",
        "HNN_Formula": "W = (X - p)^T (X - p) / 1024",
        "HNN_Threshold": HNN_THRESHOLD,
        "Connection_Probability": CONNECTION_PROBABILITY,
        "People_Seed": PEOPLE_SEED,
        "Connection_Seed": CONNECTION_SEED,
        "Occlusion_Seed": OCCLUSION_SEED,
        "Learning_Passes": LEARNING_PASSES,
        "Recall_Updates": RECALL_UPDATES,
        "Random_Occlusion": "Nested random pixels forced to zero per image",
        "Masking_Range_Percent": (
            f"{MASK_MIN_PERCENT}..{MASK_MAX_PERCENT}, step {args.mask_step}"
        ),
        "Capacity_Range": (
            f"{args.capacity_step}..{args.max_capacity}, "
            f"step {args.capacity_step}"
        ),
        "Top1_Distance": "Hamming distance to encoded clean memories",
        "Compute_Device": device_name,
        "Elapsed_Minutes": elapsed_seconds / 60.0,
    }
    return pd.DataFrame(
        [{"Parameter": key, "Value": value} for key, value in parameters.items()]
    )


def export_excel(
    path: Path,
    accuracy: np.ndarray,
    capacities: np.ndarray,
    mask_rates: np.ndarray,
    records: list[dict[str, object]],
    parameters: pd.DataFrame,
) -> None:
    x = np.tile(mask_rates, len(capacities))
    y = np.repeat(capacities, len(mask_rates))
    z = accuracy.reshape(-1)
    xyz = pd.DataFrame(
        {
            "X_Masking_Rate_Percent": x,
            "Y_Database_Capacity": y,
            "Z_Success_Rate_Percent": z,
        }
    )
    matrix = pd.DataFrame(
        accuracy,
        index=capacities,
        columns=[f"Mask_{int(rate)}pct" for rate in mask_rates],
    )
    matrix.index.name = "Database_Capacity"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        xyz.to_excel(writer, sheet_name="XYZ_Data", index=False)
        matrix.to_excel(writer, sheet_name="Accuracy_Matrix")
        parameters.to_excel(writer, sheet_name="Experiment_Parameters", index=False)
        pd.DataFrame(records).to_excel(writer, sheet_name="Selected_Database", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                letter = column_cells[0].column_letter
                max_length = min(
                    70,
                    max(len(str(cell.value or "")) for cell in column_cells) + 2,
                )
                worksheet.column_dimensions[letter].width = max_length


def export_surface_plot(
    path: Path,
    accuracy: np.ndarray,
    capacities: np.ndarray,
    mask_rates: np.ndarray,
) -> None:
    x_grid, y_grid = np.meshgrid(mask_rates, capacities)
    figure = plt.figure(figsize=(14, 10), dpi=160)
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x_grid,
        y_grid,
        accuracy,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=min(200, len(capacities)),
        ccount=len(mask_rates),
    )
    axis.set_title("Covariance HNN Recall: Random Occlusion vs Database Capacity")
    axis.set_xlabel("Random Masking Rate (%)", labelpad=10)
    axis.set_ylabel("Database Capacity", labelpad=10)
    axis.set_zlabel("Top-1 Success Rate (%)", labelpad=10)
    axis.set_xlim(float(mask_rates.min()), float(mask_rates.max()))
    axis.set_ylim(float(capacities.min()), float(capacities.max()))
    axis.set_zlim(0, 100)
    axis.view_init(elev=28, azim=-132)
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.62, pad=0.1)
    colorbar.set_label("Success Rate (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    capacities = np.arange(
        args.capacity_step,
        args.max_capacity + 1,
        args.capacity_step,
        dtype=np.int32,
    )
    mask_rates = np.arange(
        MASK_MIN_PERCENT,
        MASK_MAX_PERCENT + 1,
        args.mask_step,
        dtype=np.int16,
    )

    print(f"[setup] device: {device}")
    if device.type == "cuda":
        print(f"[setup] GPU: {torch.cuda.get_device_name(device)}")
    print(f"[setup] clean database: {args.input_dir}")
    print(
        f"[setup] grid: {len(capacities)} capacities x "
        f"{len(mask_rates)} masking rates = "
        f"{len(capacities) * len(mask_rates):,} conditions"
    )

    total_start = time.perf_counter()
    clean_pixels, database_records = load_clean_database(
        args.input_dir,
        args.max_capacity,
        PEOPLE_SEED,
    )
    mean_face = clean_pixels.astype(np.float32).mean(axis=0)
    clean_codes = encode_top_k(clean_pixels, mean_face, ACTIVE_BITS)
    if not np.all(clean_codes.sum(axis=1) == ACTIVE_BITS):
        raise RuntimeError("Fixed Top-K encoder did not produce exactly 100 active bits")

    occlusion_orders = build_occlusion_orders(args.max_capacity, OCCLUSION_SEED)
    masked_codes = precompute_masked_codes(
        clean_pixels,
        mean_face,
        occlusion_orders,
        mask_rates,
    )
    accuracy = evaluate_grid(
        clean_codes,
        masked_codes,
        capacities,
        mask_rates,
        device,
        args.rate_batch,
    )
    if np.isnan(accuracy).any():
        raise RuntimeError("Evaluation finished with missing accuracy values")

    elapsed = time.perf_counter() - total_start
    parameters = experiment_parameters(args, device, elapsed)
    excel_path = args.output_dir / "HNN_Random_Mask_Capacity_3D.xlsx"
    image_path = args.output_dir / "HNN_Random_Mask_Capacity_3D_Surface.png"
    export_excel(
        excel_path,
        accuracy,
        capacities,
        mask_rates,
        database_records,
        parameters,
    )
    export_surface_plot(image_path, accuracy, capacities, mask_rates)

    print(f"[done] Excel: {excel_path}")
    print(f"[done] plot:  {image_path}")
    print(f"[done] elapsed: {elapsed / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
