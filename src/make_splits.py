#!/usr/bin/env python3
"""Milestone 1: write the single-fold splits_final.json.

A one-element list makes fold 0 the only fold, so `nnUNetv2_train ... 0` trains
on exactly the 252 provided training cases and validates on exactly the 36
provided validation cases. Run this AFTER plan_and_preprocess; nnU-Net writes a
random 5-fold splits_final.json itself if the file is missing when training
starts, so this must be in place before the first training launch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402


def case_ids_in_labels_tr() -> set[str]:
    labels_dir = paths.raw_dataset_dir / "labelsTr"
    return {p.name[: -len(".nii.gz")] for p in labels_dir.glob("*.nii.gz")}


def main() -> None:
    source_splits_file = paths.preprocessed_dataset_dir / "source_splits.json"
    if not source_splits_file.is_file():
        raise FileNotFoundError(f"{source_splits_file} missing — run src/convert_dataset.py first")
    source_splits = json.loads(source_splits_file.read_text())

    train = sorted(source_splits["train"])
    val = sorted(source_splits["val"])

    if set(train) & set(val):
        raise RuntimeError(f"train and val overlap: {sorted(set(train) & set(val))}")

    on_disk = case_ids_in_labels_tr()
    union = set(train) | set(val)
    if union != on_disk:
        raise RuntimeError(
            f"splits do not cover labelsTr exactly. "
            f"Missing from splits: {sorted(on_disk - union)}. "
            f"Missing from labelsTr: {sorted(union - on_disk)}"
        )

    # 2.8.1 stores preprocessed cases as blosc2 (.b2nd); older versions used .npz/.npy.
    preprocessed = paths.preprocessed_dataset_dir / f"nnUNetPlans_{paths.CONFIGURATION}"
    if preprocessed.is_dir():
        extensions = (".b2nd", ".npz", ".npy")
        missing = {c for c in union if not any((preprocessed / f"{c}{e}").exists() for e in extensions)}
        if missing:
            raise RuntimeError(
                f"{len(missing)} cases are in the splits but not preprocessed, e.g. "
                f"{sorted(missing)[:5]} — re-run plan_and_preprocess"
            )
    else:
        print(f"WARNING: {preprocessed} not found; skipping the preprocessed-data check")

    paths.splits_file.write_text(json.dumps([{"train": train, "val": val}], indent=1))

    print(f"Wrote {paths.splits_file}")
    print(f"  folds: 1   train: {len(train)}   val: {len(val)}")
    subtype_map = json.loads(paths.subtype_map_file.read_text())
    for name, ids in (("train", train), ("val", val)):
        counts = {s: sum(1 for c in ids if subtype_map[c] == s) for s in range(paths.NUM_CLASSES)}
        print(f"  {name} subtype distribution: {counts}")


if __name__ == "__main__":
    main()
