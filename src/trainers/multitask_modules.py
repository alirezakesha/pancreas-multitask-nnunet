"""Classification head and the multi-task network wrapper.

Imported by the trainers in this folder. nnU-Net's external trainer discovery
adds ``src/trainers`` to ``sys.path`` and imports every module in it as a
top-level module, so this file must not use relative imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn.functional as F
from torch import nn


def downsample_mask_to_bottleneck(
    mask: torch.Tensor, spatial_size: tuple[int, ...]
) -> torch.Tensor:
    """Max-pool a lesion mask down to the bottleneck grid.

    Max-pool (not average) so any lesion voxel in a window keeps the token
    "on". ``mask`` is ``(B, D, H, W)`` or ``(B, 1, D, H, W)`` at patch
    resolution; the return is ``(B, 1, d, h, w)`` matching the bottleneck.
    """
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    return F.adaptive_max_pool3d(mask.float(), spatial_size)


class GlobalAveragePoolingHead(nn.Module):
    """GAP over the bottleneck, optionally restricted to lesion tokens.

    When ``mask`` is given, features are averaged only where the downsampled
    lesion mask is on. Samples with an empty mask fall back to plain GAP so
    the forward still produces logits (the trainer zeros the cls loss for
    those samples anyway).
    """

    def __init__(self, in_channels: int, num_classes: int = 3, p_drop: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(
        self, bottleneck: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            pooled = F.adaptive_avg_pool3d(bottleneck, 1).flatten(1)
        else:
            m = downsample_mask_to_bottleneck(mask, bottleneck.shape[2:])
            weights = m.expand_as(bottleneck)
            token_mass = weights.sum(dim=(2, 3, 4), keepdim=True)
            has_lesion = token_mass > 0
            masked = (bottleneck * weights).sum(dim=(2, 3, 4), keepdim=True) / token_mass.clamp_min(1e-6)
            gap = F.adaptive_avg_pool3d(bottleneck, 1)
            pooled = torch.where(has_lesion, masked, gap).flatten(1)
        return self.fc(self.dropout(pooled))


CLASSIFICATION_HEADS = {
    "gap": GlobalAveragePoolingHead,
}


def build_classification_head(
    head_type: str, in_channels: int, num_classes: int = 3, p_drop: float = 0.5
) -> nn.Module:
    if head_type not in CLASSIFICATION_HEADS:
        raise ValueError(
            f"Unknown classification head {head_type!r}; expected one of {sorted(CLASSIFICATION_HEADS)}"
        )
    return CLASSIFICATION_HEADS[head_type](in_channels, num_classes=num_classes, p_drop=p_drop)


class MultiTaskWrapper(nn.Module):
    """Shared encoder, segmentation decoder, and a classification head.

    Two details are load-bearing:

    * ``.encoder`` and ``.decoder`` are exposed at the top level because
      ``nnUNetTrainer.set_deep_supervision_enabled`` assigns
      ``mod.decoder.deep_supervision`` and would raise ``AttributeError`` on a
      wrapper that hid them.
    * ``return_cls = False`` makes ``forward`` return the segmentation output
      alone. ``nnUNetPredictor`` assumes a single tensor and breaks on a tuple.

    ``cls_mask`` is an optional lesion mask at patch resolution used for
    masked GAP pooling.
    """

    def __init__(self, base: nn.Module, cls_head: nn.Module):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.cls_head = cls_head
        self.return_cls = True

    def forward(self, x: torch.Tensor, cls_mask: torch.Tensor | None = None):
        skips = self.encoder(x)
        seg = self.decoder(skips)
        if not self.return_cls:
            return seg
        return seg, self.cls_head(skips[-1], mask=cls_mask)
