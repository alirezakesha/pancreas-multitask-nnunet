# ML Quiz — 3D Medical Images

Small starter setup for exploring the quiz NIfTI volumes and running nnU-Net v2.

## Data layout

| Split | Path | Notes |
|-------|------|--------|
| Train | `train/subtype{0,1,2}/` | Image + label pairs |
| Validation | `validation/subtype{0,1,2}/` | Image + label pairs |
| Test | `test/` | Images only (no labels) |

- `quiz_*_0000.nii.gz` — image
- `quiz_*.nii.gz` (no `_0000`) — mask (`0` bg, `1` pancreas, `2` lesion)

See `ReadMe.pdf` for the official quiz description.

## Setup

Use **Python 3.12** (PyTorch does not install cleanly on 3.13 here):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/MIC-DKFZ/nnUNet.git
pip install -e ./nnUNet
```

nnU-Net is installed **editable** from `./nnUNet` so you can add the classification head later.

### nnU-Net paths

```bash
source .venv/bin/activate
source setup_nnunet_env.sh
```

This sets `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` under this repo.

### Convert quiz data → nnU-Net format

```bash
source .venv/bin/activate
source setup_nnunet_env.sh
python convert_to_nnunet.py
```

Creates `nnUNet_raw/Dataset501_PancreasQuiz/` with:

- `imagesTr` / `labelsTr` — train only (252 cases)
- `imagesVal` / `labelsVal` — held-out validation (not used for training)
- `imagesTs` — test images
- `subtype_labels.csv` — case → subtype map for the classification head later

### Plan + preprocess (ResEnc M)

```bash
source .venv/bin/activate
source setup_nnunet_env.sh
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncM --verify_dataset_integrity -c 3d_fullres -np 4
```

Plans name to use later: `-p nnUNetResEncUNetMPlans`

Example **segmentation-only** train command (CUDA GPU recommended):

```bash
nnUNetv2_train 501 3d_fullres 0 -p nnUNetResEncUNetMPlans
```

### Multi-task trainer (seg + subtype classification)

Custom code lives in `custom/` (nnU-Net folder is untouched).  
`setup_nnunet_env.sh` sets `nnUNet_extTrainer` to that folder.

Classification uses **cross-attention pooling** (xattn) over the encoder bottleneck, as suggested in `ReadMe.pdf` (instead of GAP).

```bash
source .venv/bin/activate
source setup_nnunet_env.sh
nnUNetv2_train 501 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr MultiTaskTrainer
```

Read `custom/MultiTaskTrainer.py` — especially `train_step` — for a commented walkthrough.

## Explore

```bash
source .venv/bin/activate
jupyter notebook explore.ipynb
```

## Note on training hardware

This Mac has **MPS** available, but 3D nnU-Net training is much more practical on a **CUDA GPU** (Colab T4 / Kaggle P100). Local install is mainly for development and small smoke tests.
