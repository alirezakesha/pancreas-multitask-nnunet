#!/usr/bin/env python3
"""Milestone 1: quiz NIfTI folders -> nnUNetv2 raw dataset.

Both the training split and the provided validation split are written to
imagesTr/labelsTr. They are separated afterwards by splits_final.json
(src/make_splits.py), which is the standard nnU-Net idiom: it lets the
validation cases run through the same preprocessing pipeline while staying out
of every gradient update.

Writes:
  $nnUNet_raw/Dataset501_PancreasQuiz/{imagesTr,labelsTr,imagesTs}/
  $nnUNet_raw/Dataset501_PancreasQuiz/dataset.json
  $nnUNet_preprocessed/Dataset501_PancreasQuiz/subtype_map.json
  $nnUNet_preprocessed/Dataset501_PancreasQuiz/source_splits.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402  (must precede nnunetv2 imports)

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from nnunetv2.dataset_conversion.generate_dataset_json import (  # noqa: E402
    generate_dataset_json,
)

IMAGE_SUFFIX = "_0000.nii.gz"
EXPECTED_LABELS = {0, 1, 2}


def case_id_from_image(path: Path) -> str:
    """quiz_0_041_0000.nii.gz -> quiz_0_041;  quiz_037_0000.nii.gz -> quiz_037."""
    if not path.name.endswith(IMAGE_SUFFIX):
        raise ValueError(f"Not an nnU-Net channel-0 image name: {path}")
    return path.name[: -len(IMAGE_SUFFIX)]


def subtype_from_case_id(case_id: str) -> int:
    """quiz_<subtype>_<case> -> subtype. Raises if the id has no subtype field."""
    parts = case_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Cannot read a subtype from case id {case_id!r}")
    return int(parts[1])


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_integer_label(src: Path, dst: Path) -> set[int]:
    """Round float label artifacts to uint8 and assert the value set is {0,1,2}."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    img = nib.load(str(src))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
    values = set(np.unique(data).tolist())
    if not values.issubset(EXPECTED_LABELS):
        raise RuntimeError(f"Unexpected label values in {src}: {sorted(values)}")

    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(data, img.affine, header), str(dst))
    return values


def collect_labelled_split(split_dir: Path) -> list[tuple[Path, Path, int]]:
    """Return (image, label, subtype) for every case under split_dir/subtype*/."""
    subtype_dirs = sorted(split_dir.glob("subtype*"))
    if not subtype_dirs:
        raise FileNotFoundError(f"No subtype* folders under {split_dir}")

    rows: list[tuple[Path, Path, int]] = []
    for subtype_dir in subtype_dirs:
        folder_subtype = int(subtype_dir.name.removeprefix("subtype"))
        for image in sorted(subtype_dir.glob(f"*{IMAGE_SUFFIX}")):
            label = image.with_name(image.name.replace(IMAGE_SUFFIX, ".nii.gz"))
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {image}")

            # The subtype is taken from the folder; the filename must agree.
            name_subtype = subtype_from_case_id(case_id_from_image(image))
            if name_subtype != folder_subtype:
                raise RuntimeError(
                    f"Subtype mismatch for {image}: folder says {folder_subtype}, "
                    f"filename says {name_subtype}"
                )
            rows.append((image, label, folder_subtype))
    return rows


def collect_test_split(split_dir: Path) -> list[Path]:
    images = sorted(split_dir.glob(f"*{IMAGE_SUFFIX}"))
    if not images:
        raise FileNotFoundError(f"No test images under {split_dir}")
    return images


def main() -> None:
    paths.seed_everything()

    train = collect_labelled_split(paths.REPO / "train")
    val = collect_labelled_split(paths.REPO / "validation")
    test = collect_test_split(paths.REPO / "test")

    train_ids = [case_id_from_image(image) for image, _, _ in train]
    val_ids = [case_id_from_image(image) for image, _, _ in val]
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise RuntimeError(f"Case ids appear in both train and validation: {sorted(overlap)}")

    out = paths.raw_dataset_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "imagesTr").mkdir(parents=True)
    (out / "labelsTr").mkdir(parents=True)
    (out / "imagesTs").mkdir(parents=True)

    subtype_map: dict[str, int] = {}
    label_values: Counter[int] = Counter()

    for split_name, rows in (("train", train), ("val", val)):
        for image, label, subtype in rows:
            case_id = case_id_from_image(image)
            link_or_copy(image, out / "imagesTr" / f"{case_id}{IMAGE_SUFFIX}")
            label_values.update(write_integer_label(label, out / "labelsTr" / f"{case_id}.nii.gz"))
            subtype_map[case_id] = subtype
        print(f"  {split_name}: {len(rows)} cases -> imagesTr/labelsTr")

    for image in test:
        case_id = case_id_from_image(image)
        link_or_copy(image, out / "imagesTs" / f"{case_id}{IMAGE_SUFFIX}")
    print(f"  test: {len(test)} cases -> imagesTs")

    num_training = len(train) + len(val)
    generate_dataset_json(
        output_folder=str(out),
        channel_names={0: "CT"},
        labels={"background": 0, "pancreas": 1, "lesion": 2},
        num_training_cases=num_training,
        file_ending=".nii.gz",
        dataset_name=paths.DATASET_NAME,
        description=(
            "Pancreas CT multi-task quiz: whole-pancreas / lesion segmentation plus "
            "lesion subtype classification. imagesTr holds the provided train AND "
            "validation cases; they are separated by splits_final.json."
        ),
        converted_by="src/convert_dataset.py",
    )

    paths.preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.subtype_map_file.write_text(json.dumps(subtype_map, indent=1, sort_keys=True))
    (paths.preprocessed_dataset_dir / "source_splits.json").write_text(
        json.dumps({"train": sorted(train_ids), "val": sorted(val_ids)}, indent=1)
    )

    print(f"\nWrote {out}")
    print(f"  numTraining: {num_training} (train {len(train)} + provided validation {len(val)})")
    print(f"  label values across all masks: {sorted(label_values)}")
    print(f"  subtype distribution: {dict(sorted(Counter(subtype_map.values()).items()))}")
    print(f"  subtype_map: {paths.subtype_map_file}")

    if sorted(label_values) != sorted(EXPECTED_LABELS):
        raise RuntimeError(f"Label value set is {sorted(label_values)}, expected {sorted(EXPECTED_LABELS)}")


if __name__ == "__main__":
    main()
