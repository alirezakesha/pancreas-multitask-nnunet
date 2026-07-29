"""Repo paths, environment defaults and seeding.

Import this before anything from ``nnunetv2`` so the environment variables are
in place. nnU-Net 2.8.1 resolves its paths lazily (``nnunetv2/paths.py`` wraps
each one in ``_EnvPath``), so setting them here is equivalent to sourcing
``scripts/env.sh``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

SEED = 1234

REPO = Path(__file__).resolve().parent.parent

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_PancreasQuiz"

PLANS_NAME = "nnUNetResEncUNetMPlans"
CONFIGURATION = "3d_fullres"
FOLD = 0

NUM_CLASSES = 3

_DEFAULTS = {
    "nnUNet_raw": str(REPO / "nnUNet_raw"),
    "nnUNet_preprocessed": str(REPO / "nnUNet_preprocessed"),
    "nnUNet_results": str(REPO / "nnUNet_results"),
    "nnUNet_extTrainer": str(REPO / "src" / "trainers"),
    "nnUNet_compile": "f",
    "MPLCONFIGDIR": str(REPO / ".mplcache"),
}

for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

nnUNet_raw = Path(os.environ["nnUNet_raw"])
nnUNet_preprocessed = Path(os.environ["nnUNet_preprocessed"])
nnUNet_results = Path(os.environ["nnUNet_results"])

raw_dataset_dir = nnUNet_raw / DATASET_NAME
preprocessed_dataset_dir = nnUNet_preprocessed / DATASET_NAME

subtype_map_file = preprocessed_dataset_dir / "subtype_map.json"
splits_file = preprocessed_dataset_dir / "splits_final.json"

results_dir = REPO / "results"


def seed_everything(seed: int = SEED) -> int:
    """Seed random, numpy and torch. Torch is optional so data scripts stay light."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
