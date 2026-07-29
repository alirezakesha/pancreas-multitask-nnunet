# Multi-task Pancreas Segmentation and Lesion Subtype Classification

A single 3D network that segments the pancreas and its lesion **and** classifies
the lesion subtype, built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
(2.8.1) with a 3D Residual-Encoder U-Net (ResEnc-M).

One shared encoder feeds two heads: nnU-Net's segmentation decoder and a
classification head on the encoder bottleneck. The two tasks are trained jointly
with `loss = loss_seg + lambda_cls * loss_cls`.

- **wandb project:** https://wandb.ai/keshavarzian-alireza-university-health-network/pancreas-multitask-nnunet
- **Engineering log:** [`NOTES.md`](NOTES.md) — measured timings, verified API
  signatures, every deviation from the spec, and why.

<!-- RESULTS TABLE: filled in from results/metrics.json at Milestone 6 -->

## Method

```
                 CT patch  64 x 128 x 192
                         |
        +----------------+----------------+
        |   ResEnc-M encoder (6 stages)   |   shared trunk, 102 M params
        |   32 64 128 256 320 320         |
        +----------------+----------------+
                         |
          skips  --------+------- bottleneck  320 x 4 x 4 x 6
            |                          |
   +--------v---------+     +----------v-----------+
   | U-Net decoder    |     | classification head  |
   | deep supervision |     | masked GAP pooling   |
   | 3 seg classes    |     | -> 3 subtype classes |
   +--------+---------+     +----------+-----------+
            |                          |
      Dice + CE loss          weighted CE + label smoothing
            |                          |
            +----------> total <-------+
                  loss_seg + 0.5 * loss_cls
```

Design decisions and the reasoning behind them:

| Concern | Approach |
|---|---|
| Class imbalance (subtypes 62/106/84) | inverse-frequency weighted cross-entropy computed from the **training fold only**, plus `label_smoothing=0.1` |
| Class imbalance (lesion is 1.5% of voxels) | nnU-Net's Dice + CE compound loss, foreground oversampling, deep supervision |
| Overfitting on 252 cases | nnU-Net's full augmentation pipeline, dropout 0.5 in the classification head, weight decay 3e-5, a fixed held-out split that is never trained on |
| Empty-patch subtype labels | classification CE is **lesion-gated** (only patches with lesion voxels); GAP pools lesion tokens only |
| Early noisy features | first **10 epochs** are segmentation-only (`λ_cls=0`), then joint training |
| Patch-level head, case-level label | per-tile softmax aggregated over the sliding window; both plain mean and lesion-volume-weighted mean are computed and the better one is picked on validation |

Selected trainer: **`nnUNetTrainerMultiTask`** (masked GAP, `lambda_cls=0.5`). Cross-attention and `λ=1.0` ablations were dropped after early runs.

## Reproducing

Requires a CUDA GPU with >= 8 GB. Developed on NVIDIA TITAN Xp; intended to
finish training on **Google Colab T4**. See [`NOTES.md`](NOTES.md) section 1 for
the exact local environment. On Pascal (Titan Xp) use **cu126** wheels; on Colab
T4 the default Colab PyTorch/CUDA build is fine.

### 1. Install

**Local (Pascal / cu126):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
git clone https://github.com/MIC-DKFZ/nnUNet.git
pip install -e ./nnUNet
wandb login                        # or put WANDB_API_KEY in .env
```

**Google Colab (T4):** use Colab's GPU runtime, then in a cell:

```bash
# upload/clone this repo, quiz data, and nnUNet as needed
pip install -r requirements.txt
pip install -e ./nnUNet
# wandb: %env WANDB_API_KEY=...   or wandb login
```

T4 is sm_75, so tensor cores work — keep AMP on (nnU-Net default). You can leave
`nnUNet_compile=f` as in `scripts/env.sh`, or try compile later if you want.
### 2. Environment

```bash
source .venv/bin/activate
source scripts/env.sh
```

This sets the three nnU-Net data roots inside the repo, disables `torch.compile`
(Triton needs sm_70+), enables nnU-Net's built-in wandb logger, and points
`nnUNet_extTrainer` at `src/trainers/` so the custom trainers are discovered
**without editing anything under `nnUNet/`**.

### 3. Data

Place the quiz data at the repo root as `train/subtype{0,1,2}/`,
`validation/subtype{0,1,2}/` and `test/`, then:

```bash
python src/convert_dataset.py      # -> nnUNet_raw/Dataset501_PancreasQuiz, subtype_map.json
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncM -c 3d_fullres \
    --verify_dataset_integrity -np 8
python src/make_splits.py          # -> single-fold splits_final.json (252 train / 36 val)
```

All 288 labelled cases go into `imagesTr`/`labelsTr` so they share one
preprocessing pass; the 36 provided validation cases are then held out by
`splits_final.json`, which is the standard nnU-Net idiom. They never appear in a
gradient update.

### 4. Train

Single selected model only (`nnUNetTrainerMultiTask`):

```bash
./scripts/launch_train.sh          # GPU 0, tmux locally / nohup on Colab
./scripts/watch_progress.sh
./scripts/launch_train.sh --c      # resume after interruption
```

Or directly (recommended on Colab):

```bash
source scripts/env.sh
export nnUNet_n_proc_DA=4          # Colab CPUs are limited
export CUDA_VISIBLE_DEVICES=0
nnUNetv2_train 501 3d_fullres 0 \
    -tr nnUNetTrainerMultiTask -p nnUNetResEncUNetMPlans --npz
```

150 epochs × 250 iterations. Expect roughly **~20 h on Titan Xp**; a T4 is
usually faster per epoch — check the first epoch time and rescale.

For a fast pipeline check (~7 min), use the smoke trainer (same code, 20 iters/epoch):

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 0 \
    -tr nnUNetTrainerMultiTaskSmoke -p nnUNetResEncUNetMPlans --npz
```
### 5. Predict and evaluate

```bash
TRAINER=nnUNetTrainerMultiTask

python src/predict_multitask.py --split val --trainer $TRAINER --fast
python src/evaluate.py --predictions results/predictions/${TRAINER}_val_fast

CUDA_VISIBLE_DEVICES=0 python src/benchmark_inference.py --trainer $TRAINER
```

`src/evaluate.py` reports DSC and NSD at 2 mm for whole pancreas and lesion, plus
macro-F1, per-class F1, balanced accuracy, MCC and the confusion matrix, writes
`results/metrics.json`, and **exits non-zero if a target is missed**.

### 6. Submission

```bash
python src/predict_multitask.py --split test --trainer $TRAINER --fast
python src/make_submission.py --predictions results/predictions/${TRAINER}_test_fast
```

Writes `results/subtype_results.csv` (`Names,Subtype`) and
`<name>_results.zip`, validating the CSV schema, the label values and geometry of
all 72 masks, and the zip structure.

## Inference speedup

The measured cost of inference here is **not** the sliding window: the quiz
volumes are pre-cropped ROIs, so a case averages only 2.4 tiles and raising
`tile_step_size` from 0.5 to 0.7 removes 1 tile out of 185 across the whole
validation set. The time goes into nnU-Net's export path, which resamples all
three float logit channels back to the original geometry on the CPU.

`--fast` moves that resampling and the argmax onto the GPU
(`export_prediction_on_gpu`), so only a `uint8` label map returns to the host.
This is orthogonal to disabling TTA and to `step_size`, and does not touch
numeric precision. `src/benchmark_inference.py` measures both arms over the whole
validation set and asserts the whole-pancreas DSC drop stays within 0.005.

<!-- SPEEDUP NUMBERS: filled in from results/metrics.json at Milestone 5 -->

## Repository layout

```
src/
  paths.py                  repo paths, env defaults, seeding (seed 1234)
  convert_dataset.py        quiz folders -> nnU-Net raw + subtype_map.json
  make_splits.py            single-fold splits_final.json
  predict_multitask.py      dual-output inference + GPU export path
  evaluate.py               DSC, spacing-aware NSD, classification metrics
  benchmark_inference.py    baseline vs optimized timing
  benchmark_amp.py          AMP on/off diagnostic
  make_submission.py        CSV + zip with schema validation
  trainers/                 discovered via nnUNet_extTrainer
    multitask_modules.py            masked GAP head + MultiTaskWrapper
    nnUNetTrainerMultiTask.py       selected multi-task trainer
    nnUNetTrainerMultiTaskSmoke.py  fast pipeline check
scripts/
  env.sh                    exports, sourced by everything
  launch_train.sh           single-GPU train (tmux or nohup)
  watch_progress.sh         progress summary
results/                    metrics.json, subtype_results.csv
NOTES.md                    engineering log and deviations
```
`nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/` and the quiz NIfTI
folders are gitignored — no data is committed.

## Data layout (input)

| Split | Path | Notes |
|-------|------|-------|
| Train | `train/subtype{0,1,2}/` | 252 image + label pairs |
| Validation | `validation/subtype{0,1,2}/` | 36 image + label pairs, held out |
| Test | `test/` | 72 images only |

`quiz_*_0000.nii.gz` is an image, `quiz_*.nii.gz` is its mask (`0` background,
`1` pancreas, `2` lesion). The subtype is the source folder, cross-checked
against the middle field of the filename. See `ReadMe.pdf` for the official quiz
description.
