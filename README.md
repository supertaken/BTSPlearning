# Codes for BTSP learning

This repository contains the simulation code for the BTSP rule and for the Hopfield
baseline used in the manuscript *"Optoelectronic Single-Snapshot Learning with
Non-Hebbian Dynamics"*.

| File | What it computes | Panels of the paper |
|---|---|---|
| `BTSP.py` | BTSP rule: binary 1P1R array with stochastic SET/RESET, random row gating, 32-bit readout | Fig. 4F, Fig. 5D |
| `HNN.py` | Hopfield baseline: 1,024 neurons, sparse Top-K coding, covariance weights, random occlusion | Fig. 5B, Fig. 5D |
| `dataset.zip` | Binarised 32 × 32 LFW faces (5,421 identities, 12,153 images) and their structured mask / sunglasses variants | — |
| `requirements.txt` | Python dependencies | — |

---

## 1. Requirements

- **Python 3.10**
- Dependencies:

```bash
pip install -r requirements.txt
```

(or `pip install numpy pandas openpyxl opencv-python matplotlib torch`)

`HNN.py` runs on CPU; a CUDA GPU is optional and only shortens the runtime.

---

## 2. Unpack the dataset

```bash
unzip dataset.zip
```

On Windows (PowerShell):

```powershell
Expand-Archive dataset.zip -DestinationPath .
```

After unpacking, the repository looks like this:

```
BTSPlearning-main/
├── BTSP.py
├── HNN.py
├── README.md
├── dataset.zip
└── dataset/
    ├── white_bg_binary_32x32/            # clean faces, one folder per identity
    │   └── Aaron_Eckhart/
    │       └── Aaron_Eckhart_0001.png    # 32 × 32, binarised (0 / 255)
    ├── masked_white_bg_binary_32x32/     # faces with a surgical mask (Fig. 5B)
    └── sunglasses_white_bg_binary_32x32/ # faces with sunglasses (Fig. 5B)
```

> All three folders use the **`<identity>/<identity>_<index>.png`** layout.
> Keep `dataset/` next to the two scripts, or pass absolute paths (Sections 3–4).

---

## 3. Running `HNN.py` (Hopfield baseline)

```bash
python HNN.py \
    --input-dir  dataset/white_bg_binary_32x32 \
    --output-dir results_hnn \
    --max-capacity 2000 --capacity-step 5 --mask-step 1 \
    --rate-batch 8 --device auto
```

- `--input-dir` must point to the **clean** face folder (one sub-folder per identity).
  `HNN.py` generates its own random-occlusion masks internally, so no extra data is needed.
- `--max-capacity` must not exceed the number of identity folders (5,421 here).
- Outputs in `results_hnn/`:
  - `HNN_Random_Mask_Capacity_3D.xlsx` — sheets `XYZ_Data`, `Accuracy_Matrix`,
    `Experiment_Parameters` (all seeds and settings actually used) and `Selected_Database`
  - `HNN_Random_Mask_Capacity_3D_Surface.png` — the 3-D surface of Fig. 5D (Hopfield curve)

---

## 4. Running `BTSP.py` (BTSP rule)

A single command is enough. `BTSP.py` builds the random-occlusion levels it needs on the
first run and reuses them afterwards, so only the clean face folder has to be provided.

### 4.1 One-line run

```bash
python BTSP.py --clean-dir dataset/white_bg_binary_32x32 --output-dir results
```

On the first run you will see:

```
[masks] 'mask_all_data' is missing or empty; generating it from dataset/white_bg_binary_32x32
[masks] building 547,521 images (5,421 identities x 101 levels) in mask_all_data
  [masks] noise_000% ready
  ...
[masks] done: mask_all_data
```

This creates `mask_all_data/noise_000% ... noise_100%` — 101 occlusion levels, one
`<identity>_<image name>.png` per identity and level, about 548,000 small PNG files (~120 MB,
a few minutes on a laptop). Later runs reuse the folder and start the simulation directly.

### 4.2 Options

| Option | Default | Meaning |
|---|---|---|
| `--clean-dir` | required | Clean binary face database, one folder per identity. |
| `--output-dir` | required | Where the workbook and the surface plot are written. |
| `--masked-dir` | `mask_all_data` | Occlusion levels; generated from `--clean-dir` if missing or empty. |
| `--regenerate-masks` | off | Delete `--masked-dir` and rebuild it from `--clean-dir`. |
| `--min-capacity` / `--max-capacity` | `1` / `2000` | Capacity sweep (images learned). |
| `--capacity-step` | `5` | Step of the capacity sweep. |
| `--query-ratio` | `0.35` | Fraction of the database used as blind queries. |

To regenerate the occlusion levels (for example after changing the clean dataset):

```bash
python BTSP.py --clean-dir dataset/white_bg_binary_32x32 --output-dir results --regenerate-masks
```

If you already generated the levels once, you can point the script at them explicitly and it
will not rebuild anything:

```bash
python BTSP.py --clean-dir dataset/white_bg_binary_32x32 --masked-dir mask_all_data --output-dir results
```

The structured mask / sunglasses sets shipped in `dataset/` are **not** used by this script:
they are organised by identity, not by occlusion level, and belong to the structured-occlusion
analysis of Fig. 5B.

### 4.3 Outputs and runtime

Outputs in `results/`:

- `Global_3D_Accuracy_Matrix.xlsx` — 101 (masking, rows) × 400 (capacity, columns) accuracy matrix
- `System_Robustness_3D_Surface.png` — the 3-D surface of Fig. 5D (BTSP curve)

The full sweep (400 capacities × 101 masking levels, 35 % of each database queried) reads
several million images and takes hours on a laptop. For a quick check use a short sweep, e.g.
`--max-capacity 50`, and confirm that the script finishes and writes both files.

---

## 5. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `ValueError: Index contains duplicate entries, cannot reshape` | `--masked-dir` points at a folder that is not organised **per occlusion level** (for example the identity-organised `dataset/masked_white_bg_binary_32x32`): every identity folder is read as a level, the parsed percentage is 0 for all of them, and no query file matches, so the result table contains duplicate `(People_Scale, Noise_Percentage)` keys. | Drop `--masked-dir` (the script builds its own level folder) or rebuild it with `--regenerate-masks`. Check the layout before a long run: the error only appears at the very end. |
| `RuntimeError: Database has only N usable identities; M are required` (`HNN.py`) | `--max-capacity` is larger than the number of identity folders. | Use `--max-capacity` ≤ 5,421 for the shipped dataset. |
| `error: the following arguments are required: --clean-dir, --output-dir` | Both scripts take their paths from the command line. | Pass `--clean-dir` and `--output-dir` as in the examples above. |
| Few images read, or accuracy is 0 everywhere | Images are not 32 × 32 PNG/JPG, or the folder depth does not match the layout in Sections 3–4. | Check the tree with `tree dataset` / `ls -R dataset`. The code resizes anything that is not 32 × 32, but it cannot recover a missing folder level. |
| `ModuleNotFoundError: No module named 'cv2'` (or `torch`) | Dependencies not installed. | `pip install numpy pandas openpyxl opencv-python matplotlib torch` |

---

## 6. Dataset

- Source images: Labeled Faces in the Wild (LFW), deep-funnelled version.
- Pre-processing: colour → grayscale → Otsu threshold → white background → resize to
  32 × 32 → binary (0 / 255). Most identities contribute one to three images; a few have
  many more (up to 482 for the most photographed subject).
- `masked_white_bg_binary_32x32` and `sunglasses_white_bg_binary_32x32` are the structured
  occlusions used for the retrieval experiments; the random-pixel occlusions used by both
  scripts are generated from the clean set — by `BTSP.py` itself (Section 4.1) and internally in
  `HNN.py`.

## 7. Citation

If you use this code, please cite the manuscript:

> Yang, Y. et al. Optoelectronic Single-Snapshot Learning with Non-Hebbian Dynamics.

## 8. Contact

Correspondence: Yang Chai — ychai@polyu.edu.hk



