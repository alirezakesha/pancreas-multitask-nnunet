"""
custom/MultiTaskTrainer.py
==========================
External nnU-Net trainer (does NOT modify the nnUNet/ folder).

Goal of the quiz:
  - keep the ResEnc-M segmentation network
  - add a small classification head for subtype {0,1,2}
  - train BOTH heads together with one shared encoder

How nnU-Net finds this file
---------------------------
Set:
  export nnUNet_extTrainer=/path/to/this/custom

Then train with:
  nnUNetv2_train 501 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr MultiTaskTrainer

Tutorial mental model
---------------------
image patch
   │
   ▼
 shared ResEnc encoder  ──► bottleneck tokens ──► cross-attn pool ──► Linear(3) ──► subtype
   │
   ▼
 seg decoder ──► pancreas/lesion mask

Why cross-attention (xattn) instead of GAP?
  The quiz ReadMe points to RSNA multi-task solutions and asks you to try
  cross-attention pooling. GAP averages every spatial location equally;
  xattn learns *which* locations matter for subtype (e.g. lesion region).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torch.amp import autocast


# ---------------------------------------------------------------------------
# 0) Cross-attention pooling (replaces GAP)
# ---------------------------------------------------------------------------
class CrossAttentionPooling(nn.Module):
    """
    Learnable cross-attention pooler over 3D feature maps.

    Pipeline
    --------
    bottleneck (B, C, D, H, W)
        -> flatten spatial dims into N = D*H*W tokens   (B, N, C)
        -> project tokens to attn_dim                   (B, N, attn_dim)  = Keys/Values
        -> one learnable query token                    (B, 1, attn_dim)  = Query
        -> MultiheadAttention(Q, K, V)                  (B, 1, attn_dim)
        -> squeeze -> (B, attn_dim) pooled vector

    Intuition
    ---------
    The query is like asking: "Where in this volume should I look to decide subtype?"
    Softmax attention then builds a weighted summary (not a dumb average like GAP).
    """

    def __init__(self, in_channels: int, attn_dim: int | None = None, num_heads: int = 4):
        super().__init__()
        # Keep attn_dim divisible by num_heads (required by MultiheadAttention).
        self.attn_dim = attn_dim or in_channels
        if self.attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({self.attn_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Map encoder channels -> attention embedding size (can be identity width).
        self.kv_proj = nn.Linear(in_channels, self.attn_dim)

        # Single learnable query token (shared across the batch; expanded per sample).
        # Shape: (1, 1, attn_dim) -> broadcast to (B, 1, attn_dim).
        self.query = nn.Parameter(torch.randn(1, 1, self.attn_dim) * 0.02)

        # batch_first=True => tensors are (batch, seq, embed)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.attn_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: encoder bottleneck, shape (B, C, D, H, W)
        Returns:
            pooled: (B, attn_dim)
        """
        b, c, d, h, w = feat.shape

        # (B, C, D, H, W) -> (B, C, N) -> (B, N, C)   where N = D*H*W
        tokens = feat.flatten(2).transpose(1, 2)

        # Keys / values from spatial tokens
        kv = self.kv_proj(tokens)              # (B, N, attn_dim)

        # Query: same learned token for every sample in the batch
        q = self.query.expand(b, -1, -1)       # (B, 1, attn_dim)

        # Cross-attn: query attends over all spatial tokens
        # out: (B, 1, attn_dim)
        out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)

        # (B, attn_dim)
        return self.norm(out.squeeze(1))


# ---------------------------------------------------------------------------
# 1) Network wrapper: ResEnc U-Net + classification head
# ---------------------------------------------------------------------------
class MultiTaskResidualUNet(nn.Module):
    """
    Thin wrapper around the nnU-Net / ResEnc segmentation network.

    Why a wrapper?
      We reuse the exact architecture from the plans (ResidualEncoderUNet),
      and only ADD a classification head on the bottleneck features.
    """

    def __init__(self, seg_net: nn.Module, num_subtypes: int = 3):
        super().__init__()

        # Keep the original nnU-Net network as a submodule so its weights
        # are optimized together with the new head.
        self.seg_net = seg_net

        # nnUNetTrainer.set_deep_supervision_enabled() does:
        #     self.network.decoder.deep_supervision = ...
        # so our wrapper MUST expose `.decoder` (and `.encoder` is handy too).
        self.encoder = seg_net.encoder
        self.decoder = seg_net.decoder

        # Bottleneck channel count = last encoder stage width.
        # For ResEnc-M on this dataset that is typically 320.
        if hasattr(seg_net.encoder, "output_channels"):
            bottleneck_channels = int(seg_net.encoder.output_channels[-1])
        else:
            # Fallback if attribute naming changes in a future DNA version.
            bottleneck_channels = 320

        # Quiz suggestion: cross-attention pooling instead of GAP.
        # num_heads=4 works cleanly when channels=320 (320 % 4 == 0).
        self.pool = CrossAttentionPooling(
            in_channels=bottleneck_channels,
            attn_dim=bottleneck_channels,
            num_heads=4,
        )

        # Pooled vector -> 3 subtype logits (no softmax; CE wants logits).
        self.cls_head = nn.Linear(self.pool.attn_dim, num_subtypes)

    def forward(self, x: torch.Tensor):
        """
        Default forward used by nnU-Net inference code.

        Returns ONLY segmentation, so sliding-window predict still works.
        """
        skips = self.encoder(x)
        return self.decoder(skips)

    def forward_with_classification(self, x: torch.Tensor):
        """
        Training / online-validation forward.

        Returns:
          seg_out:  segmentation logits
                    - tensor, or
                    - list of tensors if deep supervision is on
          cls_logits: (batch, 3) subtype logits
        """
        # skips[-1] is the lowest-resolution / richest encoder feature map
        # (= the bottleneck). Classification reads from here.
        skips = self.encoder(x)
        seg_out = self.decoder(skips)

        bottleneck = skips[-1]                         # (B, C, D', H', W')
        pooled = self.pool(bottleneck)                 # (B, C) via xattn
        cls_logits = self.cls_head(pooled)             # (B, 3)
        return seg_out, cls_logits

    def compute_conv_feature_map_size(self, input_size):
        # Some nnU-Net utilities ask the network for memory estimates.
        return self.seg_net.compute_conv_feature_map_size(input_size)


# ---------------------------------------------------------------------------
# 2) Trainer: inherits almost everything, only changes the multi-task parts
# ---------------------------------------------------------------------------
class MultiTaskTrainer(nnUNetTrainer):
    """
    Multi-task trainer for pancreas CT:
      loss_total = loss_seg + cls_weight * loss_cls

    We intentionally keep cls_weight small-ish at first so segmentation
    (the harder / primary task in nnU-Net) is not dominated early on.
    """

    # How strongly classification pulls on the shared encoder.
    # Start modest; you can raise this later (e.g. 0.5) once seg looks OK.
    CLS_LOSS_WEIGHT = 0.2

    # Number of subtype classes in the quiz.
    NUM_SUBTYPES = 3

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        # Parent sets up folders, plans, logging, default hyperparameters, etc.
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Default nnU-Net is 1000 epochs. For Colab / early experiments,
        # 200 is usually enough to see whether the setup works.
        # Raise again for a final serious run if you have GPU time.
        self.num_epochs = 200

        # case_id -> subtype int, loaded from subtype_labels.csv
        self.subtype_map = self._load_subtype_map()

        # Optional class weights to fight imbalance (subtype1 has more cases).
        self.cls_class_weights = self._compute_class_weights(self.subtype_map)

        self.print_to_log_file(
            f"[MultiTaskTrainer] loaded {len(self.subtype_map)} subtype labels; "
            f"cls_weight={self.CLS_LOSS_WEIGHT}; class_weights={self.cls_class_weights.tolist()}"
        )

    # ------------------------------------------------------------------
    # Helpers for subtype labels
    # ------------------------------------------------------------------
    def _subtype_csv_path(self) -> Path:
        """
        CSV was written by convert_to_nnunet.py next to the raw dataset:
          nnUNet_raw/Dataset501_PancreasQuiz/subtype_labels.csv
        """
        return Path(nnUNet_raw) / self.plans_manager.dataset_name / "subtype_labels.csv"

    def _load_subtype_map(self) -> dict[str, int]:
        csv_path = self._subtype_csv_path()
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Could not find subtype CSV at {csv_path}. "
                f"Run convert_to_nnunet.py first."
            )

        mapping: dict[str, int] = {}
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Test cases have empty subtype; skip them for training labels.
                if row["subtype"] == "" or row["subtype"] is None:
                    continue
                mapping[row["case_id"]] = int(row["subtype"])
        return mapping

    def _compute_class_weights(self, mapping: dict[str, int]) -> torch.Tensor:
        """
        Inverse-frequency weights:
          rare class -> larger weight in CrossEntropyLoss
        """
        counts = Counter(mapping.values())
        weights = []
        total = sum(counts[i] for i in range(self.NUM_SUBTYPES)) or 1
        for i in range(self.NUM_SUBTYPES):
            # smooth so a missing class does not explode
            c = max(counts.get(i, 0), 1)
            weights.append(total / (self.NUM_SUBTYPES * c))
        return torch.tensor(weights, dtype=torch.float32)

    def _labels_from_keys(self, keys: List[str]) -> torch.Tensor:
        """
        nnU-Net batches include `batch['keys']` = case ids in this batch.
        We convert those ids into a LongTensor of subtype labels.
        """
        labels = []
        for k in keys:
            if k not in self.subtype_map:
                raise KeyError(
                    f"Case '{k}' has no subtype in subtype_labels.csv. "
                    f"Classification needs labels for every training/val case."
                )
            labels.append(self.subtype_map[k])
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Build network: plans ResEnc + our class head
    # ------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(plans_manager,
                                   configuration_manager,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision=True) -> nn.Module:
        """
        Called by nnUNetTrainer.initialize() (and also at inference time).

        Step A: build the official ResEnc network from plans
        Step B: wrap it with MultiTaskResidualUNet (adds cls head)
        """
        seg_net = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return MultiTaskResidualUNet(seg_net, num_subtypes=MultiTaskTrainer.NUM_SUBTYPES)

    # ------------------------------------------------------------------
    # TRAINING STEP (most important to understand)
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        """
        One optimizer update.

        batch contains (from nnU-Net dataloader):
          - data:   image patch tensor  (B, C, D, H, W)
          - target: seg ground truth    (tensor OR list of tensors if deep supervision)
          - keys:   case ids            used here to fetch subtype labels
        """
        # --- 1) Move batch to GPU/CPU device --------------------------------
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            # Deep supervision: one GT mask per decoder scale
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Subtype labels for this batch, aligned with batch dimension.
        cls_target = self._labels_from_keys(batch["keys"])

        # --- 2) Zero gradients from previous step ---------------------------
        self.optimizer.zero_grad(set_to_none=True)

        # --- 3) Forward pass ------------------------------------------------
        # autocast: mixed precision on CUDA only (faster, less VRAM)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # seg_out: mask logits (list if deep supervision)
            # cls_logits: (B, 3) subtype scores
            seg_out, cls_logits = self.network.forward_with_classification(data)

            # Segmentation loss: EXACT same loss nnU-Net already built
            # (Dice + CE, with deep-supervision weighting if enabled).
            loss_seg = self.loss(seg_out, target)

            # Classification loss: weighted CE over 3 subtypes.
            # weights live on same device as logits.
            loss_cls = nn.functional.cross_entropy(
                cls_logits,
                cls_target,
                weight=self.cls_class_weights.to(cls_logits.device),
            )

            # Joint objective. Small cls weight => seg still leads early training.
            loss_total = loss_seg + self.CLS_LOSS_WEIGHT * loss_cls

        # --- 4) Backward + optimizer step -----------------------------------
        # grad_scaler is used with CUDA AMP; otherwise plain backward.
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # nnU-Net default: clip exploding gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # Values returned here are averaged by on_train_epoch_end().
        # Keep 'loss' as TOTAL so nnU-Net's default logging still works.
        return {
            "loss": loss_total.detach().cpu().numpy(),
            "seg_loss": loss_seg.detach().cpu().numpy(),
            "cls_loss": loss_cls.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        # Parent logs mean total loss as 'train_losses'
        super().on_train_epoch_end(train_outputs)

        # Extra tutorial logging so you can see both heads moving.
        collated = collate_outputs(train_outputs)
        self.print_to_log_file(
            "train_seg_loss", float(np.mean(collated["seg_loss"])),
        )
        self.print_to_log_file(
            "train_cls_loss", float(np.mean(collated["cls_loss"])),
        )

    # ------------------------------------------------------------------
    # VALIDATION STEP
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        """
        Same forward as training, but:
          - no optimizer step
          - also compute online Dice stats (nnU-Net pseudo-dice)
          - also compute classification accuracy for this batch
        """
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        cls_target = self._labels_from_keys(batch["keys"])

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            seg_out, cls_logits = self.network.forward_with_classification(data)
            loss_seg = self.loss(seg_out, target)
            loss_cls = nn.functional.cross_entropy(
                cls_logits,
                cls_target,
                weight=self.cls_class_weights.to(cls_logits.device),
            )
            loss_total = loss_seg + self.CLS_LOSS_WEIGHT * loss_cls

        # For Dice / metrics, only the highest-resolution DS output is used.
        output = seg_out
        tgt = target
        if self.enable_deep_supervision:
            output = output[0]
            tgt = tgt[0]

        # ----- online segmentation pseudo-dice (copied from nnUNetTrainer) -----
        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float16
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        mask = None
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (tgt != self.label_manager.ignore_label).float()
                tgt = tgt.clone()
                tgt[tgt == self.label_manager.ignore_label] = 0
            else:
                mask = ~tgt[:, -1:] if tgt.dtype == torch.bool else 1 - tgt[:, -1:]
                tgt = tgt[:, :-1]

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, tgt, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # drop background channel
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        # ----- simple classification accuracy for this batch -----
        cls_pred = cls_logits.argmax(1)
        cls_correct = int((cls_pred == cls_target).sum().item())
        cls_count = int(cls_target.numel())

        return {
            "loss": loss_total.detach().cpu().numpy(),
            "seg_loss": loss_seg.detach().cpu().numpy(),
            "cls_loss": loss_cls.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
            # collate_outputs wants arrays for stacking
            "cls_correct": np.array([cls_correct]),
            "cls_count": np.array([cls_count]),
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # Parent logs dice + val loss from tp/fp/fn + 'loss'
        super().on_validation_epoch_end(val_outputs)

        collated = collate_outputs(val_outputs)
        cls_acc = float(collated["cls_correct"].sum() / max(collated["cls_count"].sum(), 1))
        self.print_to_log_file("val_cls_acc", cls_acc)
        self.print_to_log_file("val_seg_loss", float(np.mean(collated["seg_loss"])))
        self.print_to_log_file("val_cls_loss", float(np.mean(collated["cls_loss"])))

        # Store for progress plots / later wandb hook if you add one.
        self.logger.log("val_cls_acc", cls_acc, self.current_epoch)
