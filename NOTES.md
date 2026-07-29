# NOTES

Running log of verified versions, API signature checks, deviations from
`plan/SPEC.md`, and decisions. Required by SPEC R5.

**Seed: `1234`** everywhere (`src/paths.py:SEED`, applied by `seed_everything()`
in every script and in the multi-task trainer's `__init__`). nnU-Net's own
dataloader worker seeding is left untouched.

---

## 1. Environment

| Component | Version |
|---|---|
| Python | 3.12 (`.venv`) |
| torch | 2.13.0+cu126 (CUDA 12.6) |
| torchvision | 0.28.0+cu126 |
| nnunetv2 | 2.8.1, editable from `./nnUNet` |
| dynamic_network_architectures | 0.4.4 |
| batchgenerators / batchgeneratorsv2 | 0.25.3 / 0.3.5 |
| acvl_utils | 0.2.6 |
| numpy / scipy / scikit-learn | 2.4.4 / 1.18.0 / 1.9.0 |
| SimpleITK / nibabel | 2.5.5 / 5.4.2 |
| wandb | 0.28.1 |
| Host | `wirlab-debvox`, 20 CPU cores, 62 GB RAM |
| GPUs | 3x NVIDIA TITAN Xp, 12 GB, compute capability (6, 1) |
| NVIDIA driver | 580.173.02 |
| cuDNN | 91002 |

### 1.1 P1 — `sm_61` kernel check

SPEC P1 asks for `'sm_61'` in `torch.cuda.get_arch_list()`. It is **not there**:

```
torch.cuda.get_arch_list() == ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
torch.cuda.get_device_capability(0) == (6, 1)
```

**Deviation (accepted, no reinstall).** CUDA cubins are binary-compatible
upward within a major compute-capability version, so the `sm_60` cubins execute
on a 6.1 device. The functional half of the P1 check confirms it:

```
torch.nn.Conv3d(1, 8, 3, padding=1).cuda()(torch.randn(1, 1, 32, 32, 32, device='cuda'))
-> torch.Size([1, 8, 32, 32, 32])   # no "no kernel image is available" error
```

A full 8-epoch ResEnc-M training run also completed on this build before the
project was restructured, so the whole conv/norm/AMP path is exercised, not
just a single conv. `cu126` is required: cuDNN >= 9.11 (shipped with `cu130`)
drops compute capability < 7.5.

---

## 2. nnU-Net 2.8.1 API verification (SPEC R2)

Every signature the SPEC assumes was checked against the installed source
before use. All match, with two additions worth recording.

| Symbol | Installed signature / behaviour | Matches SPEC? |
|---|---|---|
| `nnUNetTrainer.__init__` | `(self, plans, configuration, fold, dataset_json, device=cuda)` | yes, but **no `unpack_dataset` argument** in 2.8.1 (older versions had one) |
| `build_network_architecture` | `@staticmethod (plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True) -> nn.Module` | yes, static, no `self` |
| `train_step` | `(self, batch: dict) -> dict` at `nnUNetTrainer.py:1019` | yes |
| `validation_step` | `(self, batch: dict) -> dict` at `nnUNetTrainer.py:1066` | yes |
| `set_deep_supervision_enabled` | `(self, enabled: bool)`, assigns `mod.decoder.deep_supervision` (`nnUNetTrainer.py:938`) | yes — so the wrapper **must** re-expose `.decoder` |
| batch dict keys | `{'data', 'target', 'keys'}` from `nnunetv2/training/dataloading/data_loader.py:207` | yes, `'keys'` is present |
| `encoder.output_channels` | set in `ResidualEncoder.__init__` as `self.output_channels = features_per_stage` | yes, `[-1] == 320` |
| autocast context | `autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context()` | copied verbatim |
| gradient clipping | `clip_grad_norm_(self.network.parameters(), 12)` | copied verbatim |

### 2.1 Extra finding not in the SPEC — `perform_actual_validation`

`nnUNetTrainer.perform_actual_validation` (`nnUNetTrainer.py:1258`) passes
`self.network` straight into `nnUNetPredictor.manual_initialization`. The
predictor assumes the network returns a single tensor, so the multi-task
wrapper's `(seg, cls)` tuple would crash at the end of **every** training run,
not only in Milestone 5. `nnUNetTrainerMultiTask` therefore overrides
`perform_actual_validation` to set `return_cls = False` around the `super()`
call and restore it afterwards.

---

## 3. Deviations from `plan/SPEC.md`

**D1. Dataset id and directory layout.** SPEC section 2 asks for
`Dataset001_PancreasQuiz` under `nnunet_data/{raw,preprocessed,results}`. This
repo keeps the pre-existing **`Dataset501_PancreasQuiz`** and the root-level
`nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/` directories, which are
already gitignored. Nothing about the method depends on either choice; the id
`501` is used consistently in all commands and scripts.

**D2. Trainer discovery — no copy into `nnunetv2/`.** SPEC section 5 says to
copy or symlink the trainer into `nnunetv2/training/nnUNetTrainer/`. nnU-Net
2.8.1 provides a supported extension point that makes this unnecessary:
`recursive_find_trainer_class_by_name` falls back to the directories listed in
the `nnUNet_extTrainer` environment variable
(`nnunetv2/utilities/find_objects.py:24-43`). `scripts/env.sh` sets
`nnUNet_extTrainer=$REPO/src/trainers`, so the trainers live only in this repo
and SPEC R1 ("do not edit nnU-Net source files") holds with zero copying.

**D3. Epoch budget.** SPEC section 5.2 fixes `num_epochs = 250` on the estimate
of 200-300 s/epoch. The measured time is **485 s/epoch** (section 4 below), so
250 epochs would be ~34 h per run. Budget set to **150 epochs (~20 h)** with
`num_iterations_per_epoch` left at the default 250, so the polynomial LR
schedule is still fully annealed at the end of training (the failure mode SPEC
5.2 warns about). 37.5k optimizer steps.

**D4. Inference speedup strategy.** SPEC 7.2's first suggested option
(`step_size=0.7`) is **excluded by the assessment ReadMe**, which requires a
strategy "beyond disabling TTA and increasing step size". FP16 is separately
excluded by SPEC P2 on Pascal. See section 6 for what was implemented instead.

**D5. NSD implementation.** No `surface_distance`, `monai` or `medpy` in the
environment, and external packages are avoided. Normalised surface distance is
implemented directly in `src/evaluate.py` with
`scipy.ndimage.distance_transform_edt`, passing the real anisotropic voxel
spacing via `sampling=`.

---

## 3.5 Milestone 1 results

`src/convert_dataset.py` then `nnUNetv2_plan_and_preprocess -d 501 -pl
nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity -np 8` then
`src/make_splits.py`. Preprocessing took **54 s** for all 288 cases — the quiz
volumes are pre-cropped ROIs (median shape 59 x 117 x 180 voxels).

| Acceptance item | Result |
|---|---|
| `--verify_dataset_integrity` | passed, no warnings |
| `nnUNetResEncUNetMPlans.json` | written |
| `imagesTr` / `labelsTr` / `imagesTs` | 288 / 288 / 72, image-label names paired |
| `dataset.json` | `numTraining: 288`, labels `{background:0, pancreas:1, lesion:2}` |
| `subtype_map.json` | 288 entries, exactly covers `labelsTr` |
| `splits_final.json` | 1 fold, 252 train / 36 val, disjoint, union == `labelsTr` |
| Label value set across all 288 masks | exactly `{0, 1, 2}` |

Voxel counts across all 288 masks: background 498,469,423 / pancreas
26,396,648 / lesion 7,917,460 — lesion is **1.5%** of all voxels and 23% of
foreground, which is the segmentation-side imbalance the report has to discuss.

Subtype distribution reproduces the provided splits exactly: train
62 / 106 / 84, validation 9 / 15 / 12.

**Every one of the 288 cases has at least one lesion voxel.** The SPEC section 8
convention for lesion-DSC on lesion-free ground truth is therefore never
triggered; `src/evaluate.py` still implements and reports it (count = 0).

Fingerprint changed slightly versus the earlier 252-case run, as expected from
adding 36 cases: target spacing `[2.0, 0.732421875, 0.732421875]` (was
`[2.0, 0.73046875, 0.73046875]`). `batch_size 2` and `patch_size [64, 128, 192]`
are unchanged.

**API finding:** 2.8.1 stores preprocessed cases as **blosc2 `.b2nd`**
(`quiz_x_y.b2nd`, `quiz_x_y_seg.b2nd`, `quiz_x_y.pkl`), not `.npz`/`.npy`. Any
script that inspects the preprocessed folder must account for this.

---

## 4. Milestone 2 — timing and hardware measurements

Stock `nnUNetTrainer`, `nnUNetResEncUNetMPlans`, `3d_fullres`, fold 0, one Titan
Xp, `nnUNet_n_proc_DA=12` (single job).

| Measurement | Value |
|---|---|
| Seconds/epoch, 288 cases (250 train + 50 val iterations) | **486 s** (epoch 1, clean) |
| Seconds/epoch, earlier 252-case run | 485 s — adding 36 cases changed nothing, as expected for a fixed iteration count |
| Peak VRAM (steady state, `nvidia-smi`) | **7203 MiB of 12288** |
| GPU utilisation | 100% in 203 of 220 samples (min 95%) -> **GPU-bound** |
| Loss trend | train 0.1604 -> -0.0412, val 0.0320 -> -0.1171 over 2 epochs; pancreas pseudo-Dice 0.0 -> 0.4629 |

Epoch 0 measured 571 s but is discarded: concurrent dry-forward tests of the
multi-task wrapper were sharing GPU 0 at the time.

**GPU-bound, not augmentation-bound.** Utilisation sits pinned at 100% rather
than oscillating, so the Colab failure mode (2 vCPUs starving the GPU) does not
apply on this 20-core host. `nnUNet_n_proc_DA` is nevertheless lowered to 6 per
job for Milestone 4, where three jobs share the same 20 cores.

**Budget.** 486 s x 150 epochs = **20.3 h** per run, which is deviation D3.

### 4.1 AMP on vs off — SPEC P2 confirmed

`src/benchmark_amp.py`, 20 timed iterations after 5 warmup, full
forward + backward + clip + optimizer step on the real architecture and patch
size, synthetic data so the dataloader is excluded.

| | s/iteration | Peak VRAM (allocated) |
|---|---|---|
| AMP on | **1.621 +/- 0.010** | **6.01 GiB** |
| AMP off | 1.781 +/- 0.006 | 10.79 GiB |
| Ratio | 1.10x faster | 1.80x less memory |

Exactly the behaviour SPEC P2 predicts: on Pascal AMP buys **memory, not speed**.
The 10% speed gain is incidental (fewer bytes moved, no tensor cores to exploit),
while 10.79 GiB of allocated tensors on a 12 GB card leaves no headroom once
caching-allocator fragmentation and the CUDA context are added. **AMP stays on.**

Sanity check on the numbers: 1.62 s/iteration x 250 iterations = 405 s, plus 50
validation iterations and epoch overhead, which matches the observed 486 s/epoch.

---

## 4.2 Milestone 3 — multi-task trainer verification

Validated with `nnUNetTrainerMultiTaskSmoke`, which is the real trainer with
`num_iterations_per_epoch=20` and `total_epochs=10`. Same code paths, ~40 s per
epoch instead of 486 s, so pipeline bugs cost minutes rather than the 1.5 h of
GPU time that Milestone 4 needs. Its numbers are meaningless (200 optimizer
steps total, final validation Dice 0.0) — it only proves the machinery works.

| Acceptance item | Result |
|---|---|
| Discovery via `nnUNet_extTrainer` | all four trainers found in `src/trainers`, nothing copied into `nnunetv2/` |
| 10 epochs without error | yes, plus the final `perform_actual_validation` |
| `'keys'` survives augmentation | `batch dict keys: ['data', 'keys', 'target']` |
| Class weights from the train fold only | counts `[62, 106, 84]` -> weights `[1.3548, 0.7925, 1.0000]` (= 252/(3*count)) |
| Both loss components logged | `train_losses_seg` 0.659 -> 0.164, `train_losses_cls` logged per epoch |
| macro-F1 logged | `val_macro_f1` present every epoch |
| `--c` resume | killed at epoch 4, resumed at exactly epoch 4 with all 15 logging lists at equal length |
| `perform_actual_validation` override | ran the predictor on all 36 validation cases (`return_cls=False` path) |
| Checkpoint contents | `cls_head.fc.{weight,bias}` present; `trainer_name` correct |

### Two bugs found and fixed by this smoke test

**1. `perform_actual_validation` would crash every run without the override.**
Confirmed as predicted: the base implementation hands `self.network` to
`nnUNetPredictor`, which cannot consume the `(seg, cls)` tuple.

**2. wandb must be finalised after the final validation, not in `on_train_end`.**
Not in the SPEC, and it would have killed all three runs at the very end.
`run_training.py:204` calls `perform_actual_validation()` **after**
`run_training()` returns (so after `on_train_end`), and that logs
`final_val/*` summaries at `nnUNetTrainer.py:1408`. SPEC 5.6's "finish in
`on_train_end`" therefore raises `wandb.errors.UsageError: Run is finished` and
exits non-zero after 20 h of training. `_finish_wandb()` is now called from the
`finally` block of `perform_actual_validation` instead. Verified with
`--val` (exit code 0).

### Deviation D6 — use nnU-Net's built-in wandb logger

SPEC 5.6 asks for a hand-rolled `wandb.init` in `on_train_start` with
`resume="allow"` and a deterministic id. nnU-Net 2.8.1 already ships
`WandbLogger` (`nnunetv2/training/logging/nnunet_logger.py`), enabled with
`nnUNet_wandb_enabled=1` and configured by `nnUNet_wandb_project` /
`nnUNet_wandb_mode`. It reads the previous run id back out of
`fold_0/wandb/latest-run` when the trainer is constructed with
`continue_training=True`, so `--c` resumes the *same* wandb run natively. A
second `wandb.init` would create a competing run in the same process, so the
built-in one is used and the multi-task metrics are registered as extra
`LocalLogger` keys, which forwards them to wandb automatically. Metric names are
consequently nnU-Net's (`train_losses`, `val_losses`, `mean_fg_dice`,
`dice_per_class_or_region/class_N`, `lrs`) plus the seven added in
`nnUNetTrainerMultiTask.extra_log_keys`, rather than SPEC's `train/loss` style.

Run names come from `WANDB_NAME` / `WANDB_RUN_GROUP`, set per variant in
`scripts/launch_ablations.sh`, because the built-in logger sets neither.

---

## 5. Ablation results (Milestone 4)

| Variant | Head | `lambda_cls` | Epochs | Whole-pancreas DSC | Lesion DSC | Macro-F1 | Notes |
|---|---|---|---|---|---|---|---|
| **nnUNetTrainerMultiTask (selected)** | masked GAP | 0.5 | 150 | _TBD_ | _TBD_ | _TBD_ | kept |
| nnUNetTrainerMultiTaskXAttn | xattn | 0.5 | stopped early | — | — | — | dropped: `loss_cls` exploded under AMP |
| nnUNetTrainerMultiTaskCls1 | GAP | 1.0 | stopped early | — | — | — | dropped: stick to GAP λ=0.5 |

**Decision:** ship / train only `nnUNetTrainerMultiTask`. XAttn and Cls1 trainer
files removed on branch `feat/colab-gap-only`. Training target: Google Colab T4.

Selected configuration: **nnUNetTrainerMultiTask** (masked GAP, λ=0.5, 10-epoch
cls warmup, lesion-gated cls loss).

### Round 1 v2 recipe (restarted 2026-07-29 ~14:24, then stopped for Colab move)

Old Round 1 runs were killed: classification loss was stuck near chance
(`loss_cls ≈ 1.12`) for GAP/Cls1 and exploding (`≈ 100`) for XAttn.

Recipe kept in the selected trainer:

1. **Lesion-gated cls loss** — CE only on patches that contain lesion voxels
2. **Masked GAP pooling** — average lesion tokens only
3. **Warmup 10 epochs** — `λ_cls = 0` for epochs 0–9, then 0.5

(XAttn fp32 attention fix is obsolete after removing that variant.)

---

## 6. Inference speedup (Milestone 5)

_Filled in at M5._

---

## 7. Decision log

- **Plans configuration.** `nnUNetResEncUNetMPlans` / `3d_fullres` as planned by
  `nnUNetPlannerResEncM`, unmodified: `batch_size 2`,
  `patch_size [64, 128, 192]`, target spacing `[2.0, 0.73046875, 0.73046875]`,
  6-stage `ResidualEncoderUNet` with `features_per_stage
  [32, 64, 128, 256, 320, 320]`, deep supervision on, `batch_dice False`.
- **One GPU per run (SPEC P5).** Confirmed `batch_size` is 2 in the plan, so DDP
  across 3 GPUs cannot divide it. The three GPUs are used for three concurrent
  single-GPU experiments instead.
- **AMP stays enabled (SPEC P2)** regardless of the timing result, for the
  activation-memory headroom on a 12 GB card.
- **`nnUNet_compile=f` (SPEC P4)**: Triton requires sm_70+.
- **`nnUNet_n_proc_DA=6`** per job (20 cores / 3 concurrent jobs). The stock
  default resolved to 12 train + 6 val workers for a single job.
