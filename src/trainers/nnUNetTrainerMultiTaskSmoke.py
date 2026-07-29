"""Pipeline smoke test: the real multi-task trainer with tiny epochs.

Same code paths as nnUNetTrainerMultiTask (train_step, validation_step, the
extra logger keys, checkpointing, --c resume, wandb) but 20 train and 5
validation iterations per epoch, so a full 10-epoch Milestone 3 acceptance run
takes minutes instead of 1.5 h of GPU time that belongs to Milestone 4.

Not for producing results — the LR schedule is annealed over 10 epochs here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAINERS = Path(__file__).resolve().parent
if str(_TRAINERS) not in sys.path:
    sys.path.insert(0, str(_TRAINERS))

import torch

from nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class nnUNetTrainerMultiTaskSmoke(nnUNetTrainerMultiTask):
    total_epochs = 10
    checkpoint_every = 2

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
