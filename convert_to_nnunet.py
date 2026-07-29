#!/usr/bin/env python3
"""Convert quiz NIfTI folders into nnU-Net raw dataset format.

Train only goes to imagesTr/labelsTr (validation is held out).
Validation is stored as imagesVal/labelsVal for later eval (ignored by nnU-Net training).
Test goes to imagesTs.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

ROOT = Path(__file__).resolve().parent
DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_PancreasQuiz"
OUT = Path(os.environ.get("nnUNet_raw", ROOT / "nnUNet_raw")) / DATASET_NAME


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_integer_label(src: Path, dst: Path) -> None:
    """Round float label artifacts to uint8 {0,1,2} and write a real file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    img = nib.load(str(src))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
    uniq = set(np.unique(data).tolist())
    if not uniq.issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected labels in {src}: {sorted(uniq)}")
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(dst))


def case_id_from_image(path: Path) -> str:
    # quiz_0_041_0000.nii.gz -> quiz_0_041
    # quiz_037_0000.nii.gz   -> quiz_037
    name = path.name
    assert name.endswith("_0000.nii.gz"), path
    return name[: -len("_0000.nii.gz")]


def collect_split(split_dir: Path) -> list[tuple[Path, Path | None, int | None]]:
    """Return list of (image, label_or_None, subtype_or_None)."""
    rows = []
    if split_dir.name == "test":
        for img in sorted(split_dir.glob("*_0000.nii.gz")):
            rows.append((img, None, None))
        return rows

    for sub in sorted(split_dir.glob("subtype*")):
        subtype = int(sub.name.replace("subtype", ""))
        for img in sorted(sub.glob("*_0000.nii.gz")):
            lbl = img.with_name(img.name.replace("_0000.nii.gz", ".nii.gz"))
            if not lbl.exists():
                raise FileNotFoundError(f"Missing label for {img}")
            rows.append((img, lbl, subtype))
    return rows


def main() -> None:
    train = collect_split(ROOT / "train")
    val = collect_split(ROOT / "validation")
    test = collect_split(ROOT / "test")

    if OUT.exists():
        shutil.rmtree(OUT)

    images_tr = OUT / "imagesTr"
    labels_tr = OUT / "labelsTr"
    images_val = OUT / "imagesVal"
    labels_val = OUT / "labelsVal"
    images_ts = OUT / "imagesTs"

    subtype_rows = []

    for img, lbl, subtype in train:
        cid = case_id_from_image(img)
        link_or_copy(img, images_tr / f"{cid}_0000.nii.gz")
        write_integer_label(lbl, labels_tr / f"{cid}.nii.gz")
        subtype_rows.append({"case_id": cid, "split": "train", "subtype": subtype})

    for img, lbl, subtype in val:
        cid = case_id_from_image(img)
        link_or_copy(img, images_val / f"{cid}_0000.nii.gz")
        write_integer_label(lbl, labels_val / f"{cid}.nii.gz")
        subtype_rows.append({"case_id": cid, "split": "validation", "subtype": subtype})

    for img, _, _ in test:
        cid = case_id_from_image(img)
        link_or_copy(img, images_ts / f"{cid}_0000.nii.gz")
        subtype_rows.append({"case_id": cid, "split": "test", "subtype": ""})

    generate_dataset_json(
        output_folder=str(OUT),
        channel_names={0: "CT"},
        labels={"background": 0, "pancreas": 1, "lesion": 2},
        num_training_cases=len(train),
        file_ending=".nii.gz",
        dataset_name=DATASET_NAME,
        description="Pancreas CT multi-task quiz (segmentation + subtype classification).",
        converted_by="ML-Quiz-3DMedImg",
    )

    meta_csv = OUT / "subtype_labels.csv"
    with meta_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "split", "subtype"])
        writer.writeheader()
        writer.writerows(subtype_rows)

    print(f"Wrote {OUT}")
    print(f"  train: {len(train)}  validation: {len(val)}  test: {len(test)}")
    print(f"  subtype map: {meta_csv}")


if __name__ == "__main__":
    main()
