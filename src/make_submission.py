#!/usr/bin/env python3
"""Milestone 7: assemble and validate the submission zip.

Takes a finished test-set prediction folder (from src/predict_multitask.py
--split test) and produces:

    results/subtype_results.csv     columns Names,Subtype
    <name>_results.zip              the 72 predicted .nii.gz plus the CSV

Every structural requirement is asserted rather than assumed: one row per test
case, `Names` carrying the `.nii.gz` extension and matching the test filenames
exactly, `Subtype` in {0,1,2}, and the zip re-opened and checked after writing.

    python src/make_submission.py \\
        --predictions results/predictions/nnUNetTrainerMultiTask_test \\
        --name alireza_keshavarzian
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

VALID_SUBTYPES = {0, 1, 2}
VALID_LABELS = {0, 1, 2}


def test_case_ids() -> list[str]:
    image_folder = paths.raw_dataset_dir / "imagesTs"
    ids = sorted(p.name[: -len("_0000.nii.gz")] for p in image_folder.glob("*_0000.nii.gz"))
    if not ids:
        raise FileNotFoundError(f"no test images in {image_folder}")
    return ids


def write_csv(case_ids: list[str], subtypes: dict[str, int], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Names", "Subtype"])
        for case_id in case_ids:
            writer.writerow([f"{case_id}.nii.gz", subtypes[case_id]])


def validate_csv(csv_path: Path, expected_ids: list[str]) -> None:
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    header, data_rows = rows[0], rows[1:]
    if header != ["Names", "Subtype"]:
        raise AssertionError(f"CSV header is {header}, expected ['Names', 'Subtype']")
    if len(data_rows) != len(expected_ids):
        raise AssertionError(f"CSV has {len(data_rows)} rows, expected {len(expected_ids)}")

    names = [row[0] for row in data_rows]
    expected_names = [f"{case_id}.nii.gz" for case_id in expected_ids]
    if sorted(names) != sorted(expected_names):
        missing = sorted(set(expected_names) - set(names))
        extra = sorted(set(names) - set(expected_names))
        raise AssertionError(f"CSV names mismatch. missing={missing[:5]} unexpected={extra[:5]}")
    if len(set(names)) != len(names):
        raise AssertionError("CSV contains duplicate Names")

    for row in data_rows:
        if int(row[1]) not in VALID_SUBTYPES:
            raise AssertionError(f"Subtype {row[1]!r} for {row[0]} is not in {sorted(VALID_SUBTYPES)}")

    print(f"  CSV ok: {len(data_rows)} rows, header {header}, all Subtype in {sorted(VALID_SUBTYPES)}")


def validate_segmentations(case_ids: list[str], prediction_folder: Path) -> dict:
    """Check every prediction exists, has label values in {0,1,2}, and matches source geometry."""
    image_folder = paths.raw_dataset_dir / "imagesTs"
    label_counts = {label: 0 for label in VALID_LABELS}
    cases_without_pancreas, cases_without_lesion = [], []

    for case_id in case_ids:
        prediction_file = prediction_folder / f"{case_id}.nii.gz"
        if not prediction_file.is_file():
            raise AssertionError(f"missing prediction {prediction_file}")

        prediction = sitk.ReadImage(str(prediction_file))
        source = sitk.ReadImage(str(image_folder / f"{case_id}_0000.nii.gz"))
        if prediction.GetSize() != source.GetSize():
            raise AssertionError(
                f"{case_id}: prediction size {prediction.GetSize()} != image {source.GetSize()}"
            )
        if not np.allclose(prediction.GetSpacing(), source.GetSpacing(), atol=1e-4):
            raise AssertionError(f"{case_id}: spacing {prediction.GetSpacing()} != {source.GetSpacing()}")

        array = sitk.GetArrayFromImage(prediction)
        values = set(np.unique(array).tolist())
        if not values.issubset(VALID_LABELS):
            raise AssertionError(f"{case_id}: label values {sorted(values)} outside {sorted(VALID_LABELS)}")
        for label in values:
            label_counts[label] += int((array == label).sum())
        if 1 not in values and 2 not in values:
            cases_without_pancreas.append(case_id)
        if 2 not in values:
            cases_without_lesion.append(case_id)

    print(f"  segmentations ok: {len(case_ids)} files, geometry matches the source images, "
          f"labels within {sorted(VALID_LABELS)}")
    if cases_without_pancreas:
        print(f"  WARNING: {len(cases_without_pancreas)} cases predict no foreground at all: "
              f"{cases_without_pancreas[:5]}")
    print(f"  cases with no predicted lesion: {len(cases_without_lesion)}")
    return {"voxel_counts": label_counts,
            "cases_without_foreground": cases_without_pancreas,
            "cases_without_lesion": cases_without_lesion}


def build_zip(case_ids: list[str], prediction_folder: Path, csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for case_id in case_ids:
            archive.write(prediction_folder / f"{case_id}.nii.gz", arcname=f"{case_id}.nii.gz")
        archive.write(csv_path, arcname=csv_path.name)


def validate_zip(zip_path: Path, case_ids: list[str]) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise AssertionError(f"corrupt entry in zip: {bad}")
        names = sorted(archive.namelist())

    expected = sorted([f"{case_id}.nii.gz" for case_id in case_ids] + ["subtype_results.csv"])
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise AssertionError(f"zip contents mismatch. missing={missing[:5]} unexpected={extra[:5]}")
    if any("/" in name for name in names):
        raise AssertionError("zip entries must be flat, with no directory prefix")

    size_mb = zip_path.stat().st_size / 2**20
    print(f"  zip ok: {len(names)} flat entries ({len(case_ids)} masks + CSV), {size_mb:.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                       help="test prediction folder containing <case>.nii.gz and classification.json")
    parser.add_argument("--name", default="alireza_keshavarzian",
                       help="prefix for <name>_results.zip")
    parser.add_argument("--strategy", default=None, choices=(None, "mean", "lesion_weighted"),
                       help="aggregation to submit; defaults to the one selected in results/metrics.json")
    args = parser.parse_args()

    prediction_folder = Path(args.predictions)
    classification_file = prediction_folder / "classification.json"
    if not classification_file.is_file():
        raise FileNotFoundError(f"{classification_file} not found — run src/predict_multitask.py --split test")

    strategy = args.strategy
    if strategy is None:
        metrics_file = paths.results_dir / "metrics.json"
        if metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            for entry in metrics.values():
                if isinstance(entry, dict) and "classification" in entry:
                    strategy = entry["classification"].get("selected_strategy")
        strategy = strategy or "mean"
        print(f"aggregation strategy: {strategy} (from results/metrics.json)")
    else:
        print(f"aggregation strategy: {strategy} (from --strategy)")

    case_ids = test_case_ids()
    predictions = json.loads(classification_file.read_text())
    missing = [case_id for case_id in case_ids if case_id not in predictions]
    if missing:
        raise AssertionError(f"{len(missing)} test cases missing from classification.json: {missing[:5]}")
    subtypes = {case_id: int(predictions[case_id][f"pred_{strategy}"]) for case_id in case_ids}

    print(f"\nvalidating {len(case_ids)} test predictions in {prediction_folder}")
    segmentation_stats = validate_segmentations(case_ids, prediction_folder)

    csv_path = paths.results_dir / "subtype_results.csv"
    write_csv(case_ids, subtypes, csv_path)
    validate_csv(csv_path, case_ids)

    zip_path = paths.REPO / f"{args.name}_results.zip"
    build_zip(case_ids, prediction_folder, csv_path, zip_path)
    validate_zip(zip_path, case_ids)

    distribution = {subtype: sum(1 for v in subtypes.values() if v == subtype)
                    for subtype in sorted(VALID_SUBTYPES)}
    print(f"\npredicted test subtype distribution: {distribution}")
    print(f"  (training distribution was {{0: 71, 1: 121, 2: 96}} over 288 cases)")
    print(f"\nwrote {csv_path}")
    print(f"wrote {zip_path}")

    summary_file = paths.results_dir / "submission_summary.json"
    summary_file.write_text(json.dumps({
        "zip": str(zip_path),
        "csv": str(csv_path),
        "num_test_cases": len(case_ids),
        "aggregation_strategy": strategy,
        "predicted_subtype_distribution": distribution,
        "segmentation": {k: v for k, v in segmentation_stats.items() if k != "voxel_counts"},
        "voxel_counts": {str(k): v for k, v in segmentation_stats["voxel_counts"].items()},
    }, indent=1))
    print(f"wrote {summary_file}")


if __name__ == "__main__":
    main()
