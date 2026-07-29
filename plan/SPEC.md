# SPEC — Multi-task Pancreas Segmentation + Subtype Classification (nnUNetv2)

Agent instructions for implementing this project. Read all of Section 0 before writing code.

---

## 0. Rules of engagement

**R1. Do not edit nnU-Net source files.** Every extension is a subclass or a standalone script under `src/`. The only exception is `splits_final.json`, which is a config artifact, not source.

**R2. Verify every nnU-Net API before you call it.** This codebase changes signatures between minor versions. Before implementing a method that overrides a base method, run the corresponding `grep` given in this spec and match the real signature in the installed version. If the signature differs from what this spec assumes, follow the installed version and note the deviation in `NOTES.md`.

**R3. No fabricated APIs.** If you cannot find an attribute or method by grepping the installed source, stop and report it rather than guessing a plausible name.

**R4. Milestones are sequential.** Do not start milestone N+1 until milestone N's acceptance check passes. Each milestone has a check that produces observable output.

**R5. Log deviations.** Maintain `NOTES.md` at repo root recording: version numbers found, signature mismatches, and any decision where you had to choose between options.

**R6. Determinism.** Seed everything (`numpy`, `random`, `torch`) in every script. Record the seed in `NOTES.md`.

---

## 1. Constraints

| Constraint | Value |
|---|---|
| Framework | nnUNetv2, installed from source (editable) |
| Encoder | 3D ResEnc-M (`nnUNetResEncUNetMPlans`), mandatory |
| Hardware | Lab server, 3× NVIDIA Titan Xp (Pascal `sm_61`, 12 GB each) |
| Session limit | None, but keep resumability — runs are 12–20 h |
| Repo root | `/home/akeshavarzian/pancreas-multitask-nnunet` |
| Pretrained weights | **Forbidden** |
| External datasets | **Forbidden** |
| Validation set | Never used for gradient updates |

**Targets (Master/PhD tier):** whole-pancreas DSC ≥ 0.91 · lesion DSC ≥ 0.31 · macro-F1 ≥ 0.70 · inference runtime ≥ 10% faster than baseline.

### 1.1 Pascal-specific constraints — read before installing anything

**P1. Verify PyTorch has `sm_61` kernels.** Pascal is old enough that recent PyTorch builds may have dropped it. Before anything else:

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_capability(0))          # expect (6, 1)
print(torch.cuda.get_arch_list())                   # must contain 'sm_61'
x = torch.randn(1,1,32,32,32, device='cuda')
print(torch.nn.Conv3d(1,8,3,padding=1).cuda()(x).shape)   # must not raise
```

If `sm_61` is absent or the conv raises `no kernel image is available for execution on the device`, install a CUDA 11.8 or 12.1 build instead. Pin the working version in `requirements.txt` and record it in `NOTES.md`. Do not proceed past this check.

**P2. Pascal has no tensor cores.** GP102 runs FP16 arithmetic at a fraction of FP32 throughput. Consequences:

- **Keep AMP enabled anyway.** Its value here is *memory*, not speed. `nnUNetPlannerResEncM` targets roughly 9–11 GB of VRAM; on a 12 GB card, disabling AMP roughly doubles activation memory and will likely OOM. Do not "optimize" by turning it off.
- Expect little or no speed benefit from AMP. Benchmark 20 iterations both ways once and record the numbers in `NOTES.md` rather than assuming.

**P3. AMP/FP16 is not a valid answer to the inference-speedup requirement on this hardware.** See Section 7.2.

**P4. Do not use `torch.compile`.** Keep `nnUNet_compile=f`. Inductor's gains depend on hardware Pascal does not have.

**P5. Do not use `-num_gpus 3`. One training run occupies one GPU.**

First establish the fact this rests on:

```bash
python -c "import json;print(json.load(open('$nnUNet_preprocessed/Dataset001_PancreasQuiz/nnUNetResEncUNetMPlans.json'))['configurations']['3d_fullres']['batch_size'])"
grep -rn "num_gpus" nnunetv2/run/run_training.py nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
```

`3d_fullres` typically plans `batch_size: 2`. nnU-Net's DDP distributes the *planned* batch across GPUs, so 3 GPUs would give each one a batch of 0.67 — it does not divide. Confirm the direction of the split by reading the source; if the installed version multiplies rather than divides, record that in `NOTES.md` and re-evaluate.

Raising `batch_size` in the plans to 6 would make DDP run, but it does not buy what you want:

- **DDP converts GPUs into a larger batch, not into shorter wall-clock.** Each GPU still processes 2 samples per step, so step time is unchanged plus gradient all-reduce overhead. A 250-epoch run still takes ~17 h.
- Titan Xp has no NVLink, so all-reduce goes over PCIe.
- nnU-Net's LR schedule (SGD, 1e-2, poly decay) is tuned for the auto-planned batch size. Tripling the batch without re-tuning the LR wastes most of the larger batch; re-tuning it is a deviation you would have to defend in the report.
- The assessment requires the auto-configured ResEnc-M plan. Hand-editing `batch_size` moves you off it for no measured gain.

**Use the three GPUs as three concurrent single-GPU experiments instead** (Section 6.1). Same 17 h, three results rather than one. If you need the single run to finish sooner, the lever is fewer epochs, not more GPUs.

---

## 2. Repository layout

Create exactly this structure.

```
.
├── README.md
├── NOTES.md
├── requirements.txt
├── src/
│   ├── convert_dataset.py        # M1
│   ├── make_splits.py            # M1
│   ├── nnUNetTrainerMultiTask.py # M3 (symlinked/copied into nnunetv2)
│   ├── predict_multitask.py      # M5
│   ├── evaluate.py               # M6
│   └── benchmark_inference.py    # M5
├── scripts/
│   ├── env.sh                    # exports, sourced by everything
│   └── launch_ablations.sh       # M4.1
├── nnunet_data/                  # gitignored: raw / preprocessed / results
└── results/
    ├── metrics.json
    └── subtype_results.csv
```

---

## 3. Milestone 1 — Data conversion

**Goal:** raw quiz data → nnUNetv2 raw format.

### 3.1 Environment variables

Set in the notebook and in every script's preamble:

```bash
export REPO=/home/akeshavarzian/pancreas-multitask-nnunet
export nnUNet_raw=$REPO/nnunet_data/raw
export nnUNet_preprocessed=$REPO/nnunet_data/preprocessed
export nnUNet_results=$REPO/nnunet_data/results
export nnUNet_compile=f        # no benefit on Pascal, see P4
```

Set `nnUNet_n_proc_DA` from the actual core count: run `nproc`, then use roughly `min(12, nproc // 2)` **per concurrent training job**. Data augmentation was the bottleneck on Colab's 2 vCPUs; on the lab server it should not be, but if you run several jobs at once you can recreate the same starvation by oversubscribing. Verify with `nvidia-smi -l 1`: GPU utilization should sit near 100%, not oscillate.

Put `nnunet_data/` in `.gitignore`. Preprocessed data is large and must not be committed.

### 3.2 `src/convert_dataset.py`

Build `$nnUNet_raw/Dataset001_PancreasQuiz/`:

```
Dataset001_PancreasQuiz/
├── imagesTr/   quiz_<subtype>_<case>_0000.nii.gz    (train AND validation cases)
├── labelsTr/   quiz_<subtype>_<case>.nii.gz         (train AND validation cases)
├── imagesTs/   quiz_<case>_0000.nii.gz              (test images)
└── dataset.json
```

**Critical:** both the train split *and* the provided validation split go into `imagesTr`/`labelsTr`. They are separated later via `splits_final.json` (Section 3.3). This is the standard nnU-Net idiom — it lets the validation cases be preprocessed by the same pipeline while remaining excluded from gradient updates.

Emit `dataset.json`:

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "pancreas": 1, "lesion": 2},
  "numTraining": <count of ALL cases in labelsTr>,
  "file_ending": ".nii.gz"
}
```

Also emit `$nnUNet_preprocessed/Dataset001_PancreasQuiz/subtype_map.json`:

```json
{"quiz_0_041": 0, "quiz_1_112": 1, ...}
```

Keys are case identifiers **without** the `_0000` suffix and without `.nii.gz`. Derive the subtype from the source folder (`subtype0/` → 0), not by parsing the filename — verify the two agree and raise on mismatch.

### 3.3 `src/make_splits.py`

Write `$nnUNet_preprocessed/Dataset001_PancreasQuiz/splits_final.json`:

```python
[{"train": [<all 252 training case ids>],
  "val":   [<all 36 provided validation case ids>]}]
```

A single-element list means fold 0 is the only fold. Training with `-f 0` then uses exactly the intended split. Assert the two lists are disjoint and that their union equals the set of files in `labelsTr`.

### 3.4 Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM \
    -c 3d_fullres --verify_dataset_integrity
```

Run **before** `make_splits.py` writes the file if planning overwrites it; if so, run `make_splits.py` afterwards and confirm the file persists.

> **ACCEPTANCE M1:** `nnUNet_preprocessed/Dataset001_PancreasQuiz/nnUNetResEncUNetMPlans.json` exists; `--verify_dataset_integrity` passes with no warnings; `subtype_map.json` has one entry per case in `labelsTr`; `splits_final.json` validates as above. Print label value counts across all masks and confirm the set is exactly `{0,1,2}`.

---

## 4. Milestone 2 — Segmentation-only baseline

Train the stock trainer for a short run to prove the pipeline end-to-end before adding a second head.

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 \
    -p nnUNetResEncUNetMPlans --npz
```

Kill it after ~5 epochs. Record seconds/epoch in `NOTES.md` and use it to set the real budget in 5.2.

**Diagnostics to run now — they determine later choices:**

1. `nvidia-smi -l 1` during training. Utilization near 100% = GPU-bound (expected on this server). Oscillating 0→100% = CPU-bound on augmentation; raise `nnUNet_n_proc_DA`.
2. Peak VRAM from `nvidia-smi`. If it exceeds ~11 GB you are close to the 12 GB ceiling and must not disable AMP under any circumstances (P2).
3. Time 20 iterations with AMP on and off. Expect little difference on Pascal; record both.

> **ACCEPTANCE M2:** loss decreases across the 5 epochs; checkpoint written; seconds/epoch, peak VRAM, GPU-vs-CPU-bound verdict, and the AMP timing comparison all in `NOTES.md`.

---

## 5. Milestone 3 — Multi-task trainer

Single file: `src/nnUNetTrainerMultiTask.py`, copied (or symlinked) to `nnunetv2/training/nnUNetTrainer/nnUNetTrainerMultiTask.py`. nnU-Net discovers trainers by scanning that package for a class whose name matches the `-tr` argument, so the file must live there and the class name must match the filename convention.

### 5.1 Signature verification (do first)

```bash
grep -n "def build_network_architecture" -A 12 \
    nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
grep -n "def train_step" -A 30 nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
grep -n "def validation_step" -A 45 nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
grep -n "def set_deep_supervision_enabled" -A 8 \
    nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
grep -rn "output_channels" nnunetv2/../dynamic_network_architectures/building_blocks/ | head
```

In recent versions `build_network_architecture` is a `@staticmethod` with no `self`. Match whatever is installed.

### 5.2 Hyperparameter overrides

In `__init__`, after `super().__init__(...)`:

```python
self.num_epochs = 250
self.num_iterations_per_epoch = 250     # base default — keep it
self.num_val_iterations_per_epoch = 50  # base default — keep it
self.save_every = 10                    # base default 50
```

**Do not** set `num_epochs = 1000` and stop early. The LR scheduler is a polynomial decay parameterised by `num_epochs`; stopping early leaves the LR un-annealed and measurably degrades the final model. Set the number you actually intend to train.

**Budget.** With no session limit, keep the default 250 iterations per epoch — the Colab plan cut this to 100 purely to survive disconnects, at a real cost in quality. Measure one epoch at M2 and extrapolate; expect somewhere in the 200–300 s range per epoch, so 250 epochs ≈ 15–20 h. That is 62.5k optimizer steps against nnU-Net's 250k default, which should be ample for 252 pre-cropped ROI cases.

If lesion DSC undershoots at M6, **raise `num_epochs` first** (400–500) — total optimizer steps is the dominant quality lever. Tune `lambda_cls` only after that.

### 5.3 Network wrapper

```python
class MultiTaskWrapper(nn.Module):
    def __init__(self, base, num_cls=3, p_drop=0.5):
        super().__init__()
        self.base = base
        self.encoder = base.encoder          # REQUIRED — see note
        self.decoder = base.decoder          # REQUIRED — see note
        feat = base.encoder.output_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Dropout(p_drop), nn.Linear(feat, num_cls))
        self.return_cls = True

    def forward(self, x):
        skips = self.base.encoder(x)
        seg = self.base.decoder(skips)
        if not self.return_cls:
            return seg                       # inference-compatible path
        return seg, self.cls_head(skips[-1])
```

Re-exposing `.encoder` and `.decoder` is load-bearing, not cosmetic: the base trainer's `set_deep_supervision_enabled` assigns `mod.decoder.deep_supervision = enabled` and will raise `AttributeError` on a wrapper that hides them.

`return_cls=False` is the escape hatch for Milestone 5 — `nnUNetPredictor` assumes the network returns a single tensor and will break on a tuple.

Verify `output_channels` is the correct attribute for the installed `dynamic_network_architectures` version; if absent, obtain the feature width with a dry forward pass on a dummy tensor of the configured patch size and record this in `NOTES.md`.

### 5.4 Classification loss and imbalance

In `initialize()` (after `self.loss` is built by the base class):

```python
counts = np.bincount(list(self.subtype_map.values()), minlength=3)
w = torch.tensor((counts.sum() / (3 * counts)), dtype=torch.float32)
self.cls_loss = nn.CrossEntropyLoss(weight=w.to(self.device),
                                    label_smoothing=0.1)
```

Load `subtype_map.json` in `__init__` from `$nnUNet_preprocessed/Dataset001_PancreasQuiz/`.

Imbalance is mild (62/106/84) but the report requires an explicit strategy; inverse-frequency weighting + label smoothing + dropout is the documented answer. Note in the report that patch sampling introduces a second, subtler imbalance: cases contribute unequal numbers of informative patches.

### 5.5 `train_step` / `validation_step`

Copy the base implementations verbatim, then modify. Do not write from scratch.

The batch dict already carries case identifiers — **no dataloader changes are needed**:

```python
def train_step(self, batch):
    data = batch['data'].to(self.device, non_blocking=True)
    target = [i.to(self.device, non_blocking=True) for i in batch['target']]
    cls_gt = torch.tensor([self.subtype_map[k] for k in batch['keys']],
                          dtype=torch.long, device=self.device)
    self.optimizer.zero_grad(set_to_none=True)
    with autocast(...):                       # copy base's autocast context exactly
        seg_out, cls_out = self.network(data)
        l = self.loss(seg_out, target) + self.lambda_cls * self.cls_loss(cls_out, cls_gt)
    # ...remainder identical to base (grad scaler, clip, step)
    return {'loss': l.detach().cpu().numpy(),
            'loss_cls': ...,  'loss_seg': ...}
```

`target` is a **list** (deep supervision), not a tensor. `self.lambda_cls = 0.5` initially.

**Before trusting this:** print `batch.keys()` once inside `train_step` and confirm `'keys'` survives the installed augmentation pipeline. If it does not, subclass the dataloader to propagate it and record the change in `NOTES.md`.

In `validation_step`, additionally accumulate `cls_out.argmax(1)` and `cls_gt` so per-epoch validation macro-F1 can be logged.

### 5.6 wandb (mandatory deliverable)

Init in `on_train_start`, finish in `on_train_end`. Log per epoch: `train/loss`, `train/loss_seg`, `train/loss_cls`, `val/loss`, `val/dice_per_class`, `val/macro_f1`, `lr`, `epoch_time_s`. Use `resume="allow"` with a fixed run id so a Colab reconnect continues the same run rather than fragmenting it.

### 5.7 Launch

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetTrainerMultiTask -p nnUNetResEncUNetMPlans --npz
# resume after a crash:
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetTrainerMultiTask -p nnUNetResEncUNetMPlans --npz --c
```

> **ACCEPTANCE M3:** 10 epochs complete without error; both loss components appear in wandb and both decrease; `--c` resumes at the correct epoch after a deliberate kill; validation macro-F1 is being logged.

---

## 6. Milestone 4 — Full training run

Train to 250 epochs on one GPU. Do not modify code mid-run — a changed loss weight invalidates the curve you will put in the report.

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetTrainerMultiTask -p nnUNetResEncUNetMPlans --npz
```

Run under `tmux` or `nohup` so an SSH drop does not kill it.

### 6.1 Parallel experiments (three GPUs, two rounds)

Three GPUs means three concurrent runs, not a faster run. Plan them as two sequential rounds of three, ~17 h each.

**Round 1 — reduce risk on the hardest target.** Macro-F1 ≥ 0.70 is the tightest requirement; the classification head is the likely failure point.

| GPU | Variant | Purpose |
|---|---|---|
| 0 | GAP head, `lambda_cls = 0.5` | baseline / fallback submission |
| 1 | **cross-attention pooling head**, `lambda_cls = 0.5` | the assessment ReadMe explicitly asks for this (RSNA 2nd place); usually beats GAP |
| 2 | GAP head, `lambda_cls = 1.0` | tests whether the cls head is undertrained |

**Round 2 — decided by Round 1 results.** Take the winning head and loss weight forward, then spend the three slots on whichever of these the Round 1 numbers make relevant: longer schedule (400–500 epochs) if lesion DSC is short, unweighted CE as a control for the report's class-imbalance section, a segmentation-only run to quantify the multi-task cost on DSC, or a second seed of the leader to check the margin is real.

Do not run Round 2 blind. If Round 1 already clears all three targets, one confirmation run plus the inference-speedup work is enough.

**Resource note:** three concurrent jobs each spawn augmentation workers. Set `nnUNet_n_proc_DA` to roughly `nproc // 4` per job and stagger launches by a few minutes. Verify with `nvidia-smi` that all three GPUs stay near 100% — if they oscillate, you have oversubscribed the CPU and all three runs are slower than one would have been alone.

Each variant is its own trainer subclass with a distinct class name, its own wandb run, and its own entry in the `NOTES.md` results table.

> **ACCEPTANCE M4:** loss plateaued on all Round 1 runs; wandb covers all three; `NOTES.md` contains a table of validation whole-pancreas DSC, lesion DSC, and macro-F1 per variant, with the selected configuration stated and justified.

---

## 7. Milestone 5 — Inference + speedup

### 7.1 `src/predict_multitask.py`

Sliding-window prediction that returns both heads. Set `network.return_cls` appropriately, or write a custom loop. Per case:

- **Segmentation:** standard nnU-Net Gaussian-weighted patch aggregation → argmax → save `quiz_<case>.nii.gz` with the source image's affine and header.
- **Classification:** collect the softmax from every patch of the case and average, then argmax. A single patch is not a case-level prediction — most patches contain no lesion at all. Weight the average by each patch's predicted lesion voxel count as a variant and keep whichever scores better on the validation set.

### 7.2 `src/benchmark_inference.py`

1. **Baseline:** TTA disabled (`--disable_tta`), `step_size=0.5`, FP32. Time the full validation set, 3 repeats after a warmup pass. Record mean ± std wall-clock.
2. **Optimized:** implement **one** strategy and re-time on identical hardware and cases. **FP16/AMP inference is not an option on Pascal** (see P2) — it will not produce a meaningful speedup and must not be your headline result. Viable choices here:
   - `step_size=0.7` with retuned Gaussian sigma (fewer sliding-window patches; simplest reliable win)
   - low-res coarse localisation pass → crop to ROI → full-res pass (FLARE-style; largest win, most work)
   - skip patches whose intensity statistics indicate pure background before the forward pass
   - batch multiple sliding-window patches into one forward call instead of one at a time

   Pin the GPU with `CUDA_VISIBLE_DEVICES` and confirm no other job is running on it during timing, otherwise the numbers are noise.
3. Report percentage improvement **and** confirm segmentation DSC did not drop by more than 0.005. A speedup that costs accuracy is not a speedup.

> **ACCEPTANCE M5:** ≥10% runtime reduction with DSC preserved; both timings in `results/metrics.json`.

---

## 8. Milestone 6 — Evaluation

`src/evaluate.py`, on the validation set only, per Metrics Reloaded:

**Segmentation** — DSC and NSD (normalised surface distance, tolerance 1–2 mm) for:
- whole pancreas: `np.uint8(label > 0)`
- lesion: `np.uint8(label == 2)`

Report mean ± std across cases *and* the per-case table. Cases with no lesion voxels in the ground truth need an explicit convention (exclude from lesion DSC, report the count) — state it in the report.

**Classification** — macro-F1 (primary), per-class F1, balanced accuracy, MCC, 3×3 confusion matrix.

Write `results/metrics.json`. Fail loudly if any target is missed, with the shortfall printed.

> **ACCEPTANCE M6:** whole-pancreas DSC ≥ 0.91, lesion DSC ≥ 0.31, macro-F1 ≥ 0.70. If short: increase `num_iterations_per_epoch` and retrain (first lever), then tune `lambda_cls` (second).

---

## 9. Milestone 7 — Deliverables

**`<name>_results.zip`**
```
quiz_037.nii.gz, quiz_045.nii.gz, ...   # test predictions, labels {0,1,2}
subtype_results.csv                      # columns: Names,Subtype
```
`Names` must include the `.nii.gz` extension and match the test filenames exactly. Assert row count equals test case count and every `Subtype` ∈ {0,1,2}.

**Public GitHub repo** — README with reproduction steps, `requirements.txt` with pinned versions, no data committed, wandb run link.

**`<name>_results.pdf`** using the provided template, containing: method description, architecture diagram, class-imbalance and overfitting strategy, wandb loss curves, validation metrics table, inference speedup with before/after timings, and an **AI workflow section** documenting which AI coding tools were used and the approximate fraction of AI-generated code (the assessment explicitly asks for >50%).

> **ACCEPTANCE M7:** zip structure validated by script; CSV schema asserted; repo clones and installs clean.

---

## 10. Summary of what gets written

| File | Purpose |
|---|---|
| `src/convert_dataset.py` | quiz layout → nnUNetv2 raw + `subtype_map.json` |
| `src/make_splits.py` | `splits_final.json` enforcing the provided val split |
| `src/nnUNetTrainerMultiTask.py` | speed knobs, cls head, dual loss, wandb |
| `src/predict_multitask.py` | dual-output sliding-window inference |
| `src/benchmark_inference.py` | baseline vs. optimized timing |
| `src/evaluate.py` | Metrics Reloaded metrics → `metrics.json` |
| `scripts/env.sh` | environment exports, sourced by all entry points |
| `scripts/launch_ablations.sh` | pins one variant per GPU |
| `NOTES.md` | version findings, signature deviations, ablation table, decisions |

Nothing inside `nnunetv2/` is modified.
