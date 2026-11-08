# Codes for BTSP learning

This repository contains the simulation code for the BTSP rule and for the Hopfield
baseline used in the manuscript *"Optoelectronic Single-Snapshot Learning with
Non-Hebbian Dynamics"*.


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


```bash
python BTSP.py --clean-dir dataset/white_bg_binary_32x32 --output-dir results
```

---


## 5. Citation

If you use this code, please cite the manuscript:

> Yang, Y. et al. Optoelectronic Single-Snapshot Learning with Non-Hebbian Dynamics.

## 6. Contact

Correspondence: Yang Chai — ychai@polyu.edu.hk



