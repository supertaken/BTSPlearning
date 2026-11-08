from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_SHAPE = (32, 32)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_MASKED_DIR = Path("mask_all_data")
MASK_LEVELS = range(0, 101)
MASK_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument(
        "--masked-dir",
        type=Path,
        default=DEFAULT_MASKED_DIR,
        help="Random-occlusion dataset (noise_000% ... noise_100%%). "
        "It is generated from --clean-dir when it is missing.",
    )
    parser.add_argument(
        "--regenerate-masks",
        action="store_true",
        help="Delete --masked-dir and rebuild it from --clean-dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-capacity", type=int, default=1)
    parser.add_argument("--max-capacity", type=int, default=2000)
    parser.add_argument("--capacity-step", type=int, default=5)
    parser.add_argument("--query-ratio", type=float, default=0.35)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean database not found: {args.clean_dir}")
    if args.min_capacity <= 0:
        raise ValueError("min-capacity must be positive")
    if args.max_capacity < args.min_capacity:
        raise ValueError("max-capacity must be at least min-capacity")
    if args.capacity_step <= 0:
        raise ValueError("capacity-step must be positive")
    if not 0 < args.query_ratio <= 1:
        raise ValueError("query-ratio must be in (0, 1]")


def percentage_from_name(path: Path) -> int:
    digits = "".join(filter(str.isdigit, path.name))
    return int(digits) if digits else 0


def load_binary_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if image.shape != IMAGE_SHAPE:
        image = cv2.resize(image, IMAGE_SHAPE, interpolation=cv2.INTER_NEAREST)
    return (image < 128).astype(np.uint8)


def generate_mask_dataset(clean_dir: Path, out_dir: Path, seed: int = MASK_SEED) -> None:
    """Build the random-occlusion levels that this script queries.

    One representative image per identity is selected once; at every level
    ``round(percent / 100 * 1024)`` pixels are drawn uniformly at random (without
    replacement) and forced to white (255), i.e. "0" in the binarised input.  The
    result is ``<out_dir>/noise_000% ... noise_100%`` with files named
    ``<identity>_<image name>.png``, which is exactly what the simulation reads.
    """
    random.seed(seed)
    np.random.seed(seed)

    identities: list[tuple[str, np.ndarray]] = []
    for folder in sorted(path for path in clean_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue
        image_path = random.choice(images)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if image.shape != IMAGE_SHAPE:
            image = cv2.resize(image, IMAGE_SHAPE, interpolation=cv2.INTER_NEAREST)
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        identities.append((f"{folder.name}_{image_path.name}", binary))

    if not identities:
        raise RuntimeError(f"No readable face images found under {clean_dir}")

    num_pixels = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    total = len(identities) * len(MASK_LEVELS)
    print(
        f"[masks] building {total:,} images "
        f"({len(identities)} identities x {len(MASK_LEVELS)} levels) in {out_dir}"
    )
    for percent in MASK_LEVELS:
        level_dir = out_dir / f"noise_{percent:03d}%"
        level_dir.mkdir(parents=True, exist_ok=True)
        num_masked = int(round(percent / 100.0 * num_pixels))
        for name, binary in identities:
            image = binary.reshape(-1).copy()
            if num_masked:
                image[np.random.choice(num_pixels, size=num_masked, replace=False)] = 255
            cv2.imwrite(str(level_dir / name), image.reshape(IMAGE_SHAPE))
        if percent % 10 == 0:
            print(f"  [masks] noise_{percent:03d}% ready")
    print(f"[masks] done: {out_dir}")


def ensure_mask_dataset(clean_dir: Path, masked_dir: Path, regenerate: bool) -> None:
    """Create the occlusion levels on first use so that a single command is enough."""
    if regenerate and masked_dir.exists():
        print(f"[masks] --regenerate-masks: removing {masked_dir}")
        shutil.rmtree(masked_dir)
    if masked_dir.is_dir() and any(masked_dir.iterdir()):
        return
    print(f"[masks] '{masked_dir}' is missing or empty; generating it from {clean_dir}")
    masked_dir.mkdir(parents=True, exist_ok=True)
    generate_mask_dataset(clean_dir, masked_dir)


def run_integrated_robustness_simulation(
    input_clean_dir: Path,
    mask_all_data_dir: Path,
    output_excel: Path,
    output_image: Path,
    people_range: range,
    prob_0_to_1: float,
    prob_1_to_0: float,
    active_rows_a: int,
    n_threshold: int,
    seed_people: int,
    seed_mapping: int,
    seed_queries: int,
    query_scale_ratio: float,
) -> None:
    noise_folders = sorted(
        (path for path in mask_all_data_dir.iterdir() if path.is_dir()),
        key=percentage_from_name,
    )
    if not noise_folders:
        raise RuntimeError("No masking-rate folders were found")

    clean_folders = sorted(path for path in input_clean_dir.iterdir() if path.is_dir())
    if not clean_folders:
        raise RuntimeError("No identity folders were found")

    accuracy_results: list[dict[str, float | int]] = []
    representative: dict[str, Path | None] = {}
    missing_queries = 0
    total_queries = 0
    for num_people in people_range:
        query_count = max(1, int(round(num_people * query_scale_ratio)))
        people_rng = random.Random(seed_people)
        actual_num_people = min(num_people, len(clean_folders))
        selected_folders = sorted(people_rng.sample(clean_folders, actual_num_people))

        clean_images: list[np.ndarray] = []
        image_mapping: list[tuple[str, str]] = []
        for folder in selected_folders:
            images = sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                continue
            # The occlusion files are named "<identity>_<image name>.png" and were built from
            # one representative image per identity.  Use that same image here, otherwise the
            # query below has no file to read and is silently dropped.
            if folder.name not in representative:
                representative[folder.name] = next(
                    (
                        candidate
                        for candidate in images
                        if (noise_folders[0] / f"{folder.name}_{candidate.name}").is_file()
                    ),
                    None,
                )
            image_path = representative[folder.name] or people_rng.choice(images)
            image = load_binary_image(image_path)
            if image is None:
                continue
            clean_images.append(image)
            image_mapping.append((folder.name, image_path.name))

        if not clean_images:
            raise RuntimeError(f"No readable images were found at capacity {num_people}")

        mapping_rng = np.random.RandomState(seed_mapping)
        mapping_matrix = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        for image in clean_images:
            selected_rows = mapping_rng.choice(32, size=active_rows_a, replace=False)
            row_mask = np.zeros(IMAGE_SHAPE, dtype=bool)
            row_mask[selected_rows, :] = True
            active_mask = (image == 1) & row_mask
            random_values = mapping_rng.rand(*IMAGE_SHAPE)
            set_mask = (mapping_matrix == 0) & active_mask
            reset_mask = (mapping_matrix == 1) & active_mask
            mapping_matrix[set_mask & (random_values < prob_0_to_1)] = 1
            mapping_matrix[reset_mask & (random_values < prob_1_to_0)] = 0

        reference_vectors = np.asarray(
            [
                (np.bitwise_and(image, mapping_matrix).sum(axis=0) > n_threshold).astype(
                    np.uint8
                )
                for image in clean_images
            ]
        )

        actual_query_count = min(query_count, len(image_mapping))
        query_indices = random.Random(seed_queries).sample(
            range(len(image_mapping)), actual_query_count
        )
        print(
            f"[evaluate] capacity={actual_num_people}, "
            f"queries={actual_query_count}"
        )

        for noise_folder in noise_folders:
            correct_matches = 0
            for query_index in query_indices:
                person_name, filename = image_mapping[query_index]
                image_path = noise_folder / f"{person_name}_{filename}"
                image = load_binary_image(image_path) if image_path.exists() else None
                total_queries += 1
                if image is None:
                    missing_queries += 1
                    continue

                query_vector = (
                    np.bitwise_and(image, mapping_matrix).sum(axis=0) > n_threshold
                ).astype(np.uint8)
                scores = np.sum(reference_vectors == query_vector, axis=1)
                if int(np.argmax(scores)) == query_index:
                    correct_matches += 1

            accuracy = (
                correct_matches / actual_query_count * 100
                if actual_query_count
                else 0.0
            )
            accuracy_results.append(
                {
                    "People_Scale": actual_num_people,
                    "Noise_Percentage": percentage_from_name(noise_folder),
                    "Accuracy": accuracy,
                }
            )

    if missing_queries:
        print(
            f"[warn] {missing_queries:,} of {total_queries:,} query images were missing from "
            f"'{mask_all_data_dir}' and were counted as failures. Rebuild the occlusion dataset "
            f"with --regenerate-masks so that it matches this clean folder."
        )

    flat_results = pd.DataFrame(accuracy_results)
    accuracy_matrix = flat_results.pivot(
        index="Noise_Percentage",
        columns="People_Scale",
        values="Accuracy",
    )
    if 0 in accuracy_matrix.index:
        print(
            f"[check] mean accuracy at 0 % masking: "
            f"{accuracy_matrix.loc[0].mean():.1f} % (expected ~100 % without occlusion)"
        )
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    accuracy_matrix.to_excel(output_excel)

    capacities = accuracy_matrix.columns.values
    noise_rates = accuracy_matrix.index.values
    capacity_grid, noise_grid = np.meshgrid(capacities, noise_rates)

    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        capacity_grid,
        noise_grid,
        accuracy_matrix.values,
        cmap="viridis",
        linewidth=0.5,
        antialiased=True,
        alpha=0.9,
        edgecolor="gray",
    )
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.5, aspect=10, pad=0.08)
    colorbar.set_label("Top-1 Recognition Accuracy (%)", fontsize=11)
    axis.set_title(
        "System Robustness Profile (Dynamic Query-Size Scaling)\n"
        "(32D Column-Feature Hardware Simulation)",
        fontsize=12,
    )
    axis.set_xlabel("Database Identity Scale", fontsize=11, labelpad=10)
    axis.set_ylabel("Stuck-at-1 Masking Rate (%)", fontsize=11, labelpad=10)
    axis.set_zlabel("Top-1 Match Accuracy (%)", fontsize=11, labelpad=10)
    axis.set_zlim(0, 100)
    axis.view_init(elev=28, azim=-125)
    figure.tight_layout()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_image, dpi=250)
    plt.close(figure)

    print(f"[done] Excel: {output_excel}")
    print(f"[done] plot:  {output_image}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_mask_dataset(args.clean_dir, args.masked_dir, args.regenerate_masks)
    people_range = range(
        args.min_capacity,
        args.max_capacity + 1,
        args.capacity_step,
    )
    run_integrated_robustness_simulation(
        input_clean_dir=args.clean_dir,
        mask_all_data_dir=args.masked_dir,
        output_excel=args.output_dir / "Global_3D_Accuracy_Matrix.xlsx",
        output_image=args.output_dir / "System_Robustness_3D_Surface.png",
        people_range=people_range,
        prob_0_to_1=0.50,
        prob_1_to_0=1.0,
        active_rows_a=5,
        n_threshold=0,
        seed_people=42,
        seed_mapping=32,
        seed_queries=25,
        query_scale_ratio=args.query_ratio,
    )


if __name__ == "__main__":
    main()
