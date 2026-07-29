"""Milestone 3: multi-task trainer — shared encoder, segmentation decoder, subtype head.

Discovered by nnU-Net through the ``nnUNet_extTrainer`` environment variable
(see ``scripts/env.sh``), so nothing inside ``nnunetv2/`` is modified.

    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 0 \
        -tr nnUNetTrainerMultiTask -p nnUNetResEncUNetMPlans --npz

``train_step`` and ``validation_step`` are the nnUNetTrainer 2.8.1 bodies with
the classification branch added; the autocast context and the grad-scaler /
gradient-clipping block are copied verbatim so AMP behaviour is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import time
from typing import List

_TRAINERS = Path(__file__).resolve().parent
_SRC = _TRAINERS.parent
for _entry in (str(_TRAINERS), str(_SRC)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import autocast, nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

import paths
from multitask_modules import MultiTaskWrapper, build_classification_head


class PolyLRWithGroupMultipliers(PolyLRScheduler):
    """PolyLRScheduler that honours a per-param-group ``lr_mult``.

    The stock scheduler assigns the same value to every param group, so a group
    created with its own learning rate is silently overwritten on the first step.
    """

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr * param_group.get('lr_mult', 1.0)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class nnUNetTrainerMultiTask(nnUNetTrainer):
    """ResEnc-M segmentation plus a subtype classification head on the bottleneck."""

    # --- variant knobs. Subclasses in this folder override only these. ---
    cls_head_type = "gap"
    lambda_cls = 0.5
    cls_dropout = 0.5
    label_smoothing = 0.1
    weighted_cls_loss = True
    # First N epochs train segmentation only (λ_cls treated as 0). Gives the
    # encoder a chance to draw the lesion before the subtype head starts.
    cls_warmup_epochs = 10
    # Label id of the lesion channel in the nnU-Net target (dataset.json).
    lesion_label = 2

    # Learning-rate multiplier for the classification head only. 1.0 keeps a
    # single param group, i.e. byte-for-byte the stock optimizer. Anything else
    # is needed when the head cannot survive nnU-Net's SGD(lr=0.01,
    # momentum=0.99) — effective step ~lr/(1-momentum) = 1.0. See D7 in NOTES.md.
    cls_head_lr_mult = 1.0

    # 150 epochs x 250 iterations at the measured ~485 s/epoch is roughly 20 h.
    # The polynomial LR schedule is parameterised by num_epochs, so this is the
    # number we actually train to — never a larger number stopped early.
    total_epochs = 150
    checkpoint_every = 10

    # Logged once per epoch on top of nnU-Net's own keys. Registered with the
    # LocalLogger so they land in progress.png data, the checkpoint, and wandb.
    extra_log_keys = (
        "train_losses_seg",
        "train_losses_cls",
        "val_losses_seg",
        "val_losses_cls",
        "val_macro_f1",
        "val_cls_balanced_accuracy",
        "epoch_duration_s",
        "train_cls_fraction_with_lesion",
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        paths.seed_everything()

        self.num_epochs = self.total_epochs
        self.save_every = self.checkpoint_every

        self.subtype_map = self._load_subtype_map()
        self.cls_loss = None
        self._logged_batch_keys = False

        self._register_extra_log_keys()
        self.logger.update_config({
            "multitask": {
                "cls_head_type": self.cls_head_type,
                "lambda_cls": self.lambda_cls,
                "cls_warmup_epochs": self.cls_warmup_epochs,
                "cls_dropout": self.cls_dropout,
                "label_smoothing": self.label_smoothing,
                "weighted_cls_loss": self.weighted_cls_loss,
                "lesion_gated_cls": True,
                "masked_cls_pooling": True,
                "num_epochs": self.num_epochs,
                "seed": paths.SEED,
            }
        })

    # ------------------------------------------------------------------ setup

    def _load_subtype_map(self) -> dict[str, int]:
        map_file = Path(self.preprocessed_dataset_folder_base) / "subtype_map.json"
        if not map_file.is_file():
            raise FileNotFoundError(
                f"{map_file} not found. Run src/convert_dataset.py, which writes the "
                f"case-id -> subtype map next to the preprocessed data."
            )
        return json.loads(map_file.read_text())

    def _register_extra_log_keys(self) -> None:
        """Add the multi-task keys to nnU-Net's LocalLogger store.

        LocalLogger.log asserts the key already exists, and plot_progress_png
        infers the epoch from the shortest list, so a key added mid-run is padded
        with NaN up to the current length.
        """
        store = self.logger.local_logger.my_fantastic_logging
        length = len(store["train_losses"])
        for key in self.extra_log_keys:
            values = store.setdefault(key, [])
            values.extend([float("nan")] * (length - len(values)))

    def load_checkpoint(self, checkpoint) -> None:
        super().load_checkpoint(checkpoint)
        # The checkpoint replaces the whole logging store, so re-register.
        self._register_extra_log_keys()

    @classmethod
    def build_network_architecture(cls,
                                   plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """Wrap the planned ResEnc-M network with a classification head.

        A classmethod rather than the base class's staticmethod so subclasses can
        select their head through ``cls``. Both call sites work with either:
        nnUNetTrainer.initialize calls ``self.build_network_architecture(...)``
        and nnUNetPredictor.initialize_from_trained_model_folder calls
        ``trainer_class.build_network_architecture(...)``, and
        ``inspect.signature`` on the bound method still shows ``plans_manager``,
        which is what both use to pick the new-style signature.
        """
        base = nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision,
        )
        bottleneck_channels = base.encoder.output_channels[-1]
        head = build_classification_head(
            cls.cls_head_type,
            bottleneck_channels,
            num_classes=paths.NUM_CLASSES,
            p_drop=cls.cls_dropout,
        )
        return MultiTaskWrapper(base, head)

    def initialize(self):
        super().initialize()
        self.cls_loss = self._build_classification_loss()

    def configure_optimizers(self):
        """Stock optimizer, optionally with a lower learning rate for the head.

        When ``cls_head_lr_mult == 1.0`` this defers entirely to the base class so
        the optimizer still has exactly **one** param group. That matters for
        resuming: ``optimizer.load_state_dict`` rejects a checkpoint whose param
        group count differs, so runs started before this method existed stay
        resumable with ``--c``.
        """
        if self.cls_head_lr_mult == 1.0:
            return super().configure_optimizers()

        network = self.network.module if self.is_ddp else self.network
        head_parameters = list(network.cls_head.parameters())
        head_ids = {id(p) for p in head_parameters}
        backbone_parameters = [p for p in network.parameters() if id(p) not in head_ids]

        optimizer = torch.optim.SGD(
            [{'params': backbone_parameters, 'lr_mult': 1.0},
             {'params': head_parameters, 'lr_mult': self.cls_head_lr_mult}],
            self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True,
        )
        lr_scheduler = PolyLRWithGroupMultipliers(optimizer, self.initial_lr, self.num_epochs)
        self.print_to_log_file(
            f"two param groups: backbone lr_mult 1.0 ({len(backbone_parameters)} tensors), "
            f"cls_head lr_mult {self.cls_head_lr_mult} ({len(head_parameters)} tensors) "
            f"-> head starts at lr {self.initial_lr * self.cls_head_lr_mult:g}"
        )
        return optimizer, lr_scheduler

    def _build_classification_loss(self) -> nn.Module:
        """Inverse-frequency weighted CE with label smoothing, from the train fold only."""
        train_keys, _ = self.do_split()
        missing = [k for k in train_keys if k not in self.subtype_map]
        if missing:
            raise KeyError(f"{len(missing)} training cases missing from subtype_map.json, e.g. {missing[:5]}")

        counts = np.bincount([self.subtype_map[k] for k in train_keys], minlength=paths.NUM_CLASSES)
        if (counts == 0).any():
            raise RuntimeError(f"A subtype is absent from the training fold: counts={counts.tolist()}")

        if self.weighted_cls_loss:
            weight = torch.tensor(counts.sum() / (paths.NUM_CLASSES * counts),
                                  dtype=torch.float32, device=self.device)
        else:
            weight = None

        self.print_to_log_file(
            f"classification head: {self.cls_head_type}, lambda_cls={self.lambda_cls}, "
            f"warmup_epochs={self.cls_warmup_epochs}, dropout={self.cls_dropout}, "
            f"label_smoothing={self.label_smoothing}, lesion_gated=True, masked_pool=True"
        )
        self.print_to_log_file(
            f"train-fold subtype counts: {counts.tolist()}, "
            f"class weights: {'none' if weight is None else np.round(weight.cpu().numpy(), 4).tolist()}"
        )
        # reduction='none' so train_step can zero out patches with no lesion (#1).
        return nn.CrossEntropyLoss(
            weight=weight, label_smoothing=self.label_smoothing, reduction='none'
        )

    def _classification_targets(self, batch: dict) -> torch.Tensor:
        """Case-level subtype labels for the patches in this batch.

        The batch dict carries 'keys' (nnUNetDataLoader.generate_train_batch), so
        no dataloader subclassing is needed.
        """
        if "keys" not in batch:
            raise KeyError(
                f"batch has no 'keys' entry (got {sorted(batch.keys())}); the augmentation "
                f"pipeline dropped the case identifiers and the dataloader needs subclassing"
            )
        try:
            labels = [self.subtype_map[k] for k in batch["keys"]]
        except KeyError as e:
            raise KeyError(f"case {e} is not in subtype_map.json") from e
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def _lesion_mask_from_target(self, target) -> torch.Tensor:
        """Binary lesion mask at full patch resolution, shape (B, D, H, W)."""
        seg = target[0] if isinstance(target, list) else target
        # nnU-Net targets are (B, 1, *spatial) for regular (non-region) training.
        if seg.ndim == 5 and seg.shape[1] == 1:
            seg = seg[:, 0]
        elif seg.ndim == 5:
            seg = seg[:, 0]
        return (seg == self.lesion_label).to(dtype=torch.float32)

    def _effective_lambda_cls(self) -> float:
        if self.current_epoch < self.cls_warmup_epochs:
            return 0.0
        return float(self.lambda_cls)

    def _gated_cls_loss(
        self, cls_output: torch.Tensor, cls_target: torch.Tensor, lesion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean CE over samples that contain at least one lesion voxel (#1).

        Returns ``(loss, fraction_with_lesion)``. If no sample in the batch has
        a lesion, returns a zero connected to the graph so AMP/backward stay happy.
        """
        has_lesion = lesion_mask.reshape(lesion_mask.shape[0], -1).sum(dim=1) > 0
        per_sample = self.cls_loss(cls_output, cls_target)
        if has_lesion.any():
            loss = per_sample[has_lesion].mean()
        else:
            loss = per_sample.sum() * 0.0
        fraction = has_lesion.float().mean()
        return loss, fraction

    # ------------------------------------------------------------------- steps

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        if not self._logged_batch_keys:
            self.print_to_log_file(f"batch dict keys: {sorted(batch.keys())}")
            self._logged_batch_keys = True

        cls_target = self._classification_targets(batch)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        lesion_mask = self._lesion_mask_from_target(target)
        lambda_cls = self._effective_lambda_cls()

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data, cls_mask=lesion_mask)
            loss_seg = self.loss(seg_output, target)
            loss_cls, cls_fraction = self._gated_cls_loss(cls_output, cls_target, lesion_mask)
            l = loss_seg + lambda_cls * loss_cls

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': l.detach().cpu().numpy(),
            'loss_seg': loss_seg.detach().cpu().numpy(),
            'loss_cls': loss_cls.detach().cpu().numpy(),
            'cls_fraction': float(cls_fraction.detach().cpu()),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        cls_target = self._classification_targets(batch)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        lesion_mask = self._lesion_mask_from_target(target)
        lambda_cls = self._effective_lambda_cls()

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data, cls_mask=lesion_mask)
            del data
            loss_seg = self.loss(seg_output, target)
            loss_cls, _ = self._gated_cls_loss(cls_output, cls_target, lesion_mask)
            l = loss_seg + lambda_cls * loss_cls

        output = seg_output
        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        # Online F1 only on patches that actually contain a lesion (#1).
        has_lesion = (lesion_mask.reshape(lesion_mask.shape[0], -1).sum(dim=1) > 0).cpu().tolist()
        cls_pred = cls_output.detach().float().argmax(1).cpu().tolist()
        cls_gt = cls_target.detach().cpu().tolist()
        cls_pred = [p for p, keep in zip(cls_pred, has_lesion) if keep]
        cls_gt = [g for g, keep in zip(cls_gt, has_lesion) if keep]

        return {
            'loss': l.detach().cpu().numpy(),
            'loss_seg': loss_seg.detach().cpu().numpy(),
            'loss_cls': loss_cls.detach().cpu().numpy(),
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
            'cls_pred': cls_pred,
            'cls_gt': cls_gt,
        }

    # ------------------------------------------------------------- epoch hooks

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        self.logger.log('train_losses_seg', float(np.mean(outputs['loss_seg'])), self.current_epoch)
        self.logger.log('train_losses_cls', float(np.mean(outputs['loss_cls'])), self.current_epoch)
        self.logger.log(
            'train_cls_fraction_with_lesion',
            float(np.mean(outputs['cls_fraction'])),
            self.current_epoch,
        )

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)

        cls_gt = np.asarray(outputs['cls_gt']) if outputs['cls_gt'] else np.asarray([])
        cls_pred = np.asarray(outputs['cls_pred']) if outputs['cls_pred'] else np.asarray([])
        if cls_gt.size == 0:
            macro_f1 = 0.0
            balanced_accuracy = 0.0
        else:
            macro_f1 = f1_score(cls_gt, cls_pred, average='macro',
                                labels=list(range(paths.NUM_CLASSES)), zero_division=0)
            balanced_accuracy = balanced_accuracy_score(cls_gt, cls_pred)

        self.logger.log('val_losses_seg', float(np.mean(outputs['loss_seg'])), self.current_epoch)
        self.logger.log('val_losses_cls', float(np.mean(outputs['loss_cls'])), self.current_epoch)
        self.logger.log('val_macro_f1', float(macro_f1), self.current_epoch)
        self.logger.log('val_cls_balanced_accuracy', float(balanced_accuracy), self.current_epoch)

    def on_epoch_end(self):
        # logged before super() so every list has one entry per epoch when
        # plot_progress_png infers the epoch from the shortest one
        duration = time() - self.logger.get_value('epoch_start_timestamps', step=self.current_epoch)
        self.logger.log('epoch_duration_s', float(duration), self.current_epoch)

        super().on_epoch_end()

        # super() already incremented current_epoch; report the epoch we just finished.
        finished = self.current_epoch - 1
        lam = 0.0 if finished < self.cls_warmup_epochs else float(self.lambda_cls)
        warmup = "warmup" if finished < self.cls_warmup_epochs else "cls_on"
        self.print_to_log_file(
            f"[{warmup} λ={lam:g}] "
            f"train_loss_seg {np.round(self.logger.get_value('train_losses_seg', step=-1), decimals=4)}  "
            f"train_loss_cls {np.round(self.logger.get_value('train_losses_cls', step=-1), decimals=4)}  "
            f"cls_frac {np.round(self.logger.get_value('train_cls_fraction_with_lesion', step=-1), decimals=3)}  "
            f"val_macro_f1 {np.round(self.logger.get_value('val_macro_f1', step=-1), decimals=4)}"
        )

    def _finish_wandb(self) -> None:
        """Close the wandb run. nnU-Net's WandbLogger never finalises its own.

        Must NOT be called from on_train_end: run_training.py calls
        perform_actual_validation *after* run_training returns, and that logs
        final_val/* summaries (nnUNetTrainer.py:1408). Finishing earlier makes
        wandb raise "Run is finished" and kills the process at the very end of a
        20 h run.
        """
        for logger in getattr(self.logger, 'loggers', []):
            run = getattr(logger, 'run', None)
            if run is not None and not getattr(run, '_is_finished', False):
                run.finish()

    # --------------------------------------------------------------- inference

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Run the end-of-training validation with the classification head muted.

        The base implementation hands ``self.network`` to
        ``nnUNetPredictor.manual_initialization``, and the predictor assumes the
        network returns a single tensor. Without this it crashes on the (seg, cls)
        tuple at the end of every run.
        """
        network = self.network.module if self.is_ddp else self.network
        if isinstance(network, OptimizedModule):
            network = network._orig_mod

        previous = getattr(network, 'return_cls', None)
        if previous is not None:
            network.return_cls = False
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            if previous is not None:
                network.return_cls = previous
            self._finish_wandb()
