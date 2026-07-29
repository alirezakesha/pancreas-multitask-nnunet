#!/usr/bin/env python3
"""Milestone 6: validation metrics following Metrics Reloaded.

Segmentation, for whole pancreas (``label > 0``) and lesion (``label == 2``):
  * DSC
  * NSD (normalised surface distance) at a 2 mm tolerance, spacing-aware

Classification:
  * macro-F1 (primary), per-class F1, balanced accuracy, MCC, confusion matrix

Writes ``results/metrics.json`` with mean +/- std and the full per-case table,
and exits non-zero if any target is missed.

    python src/evaluate.py --predictions results/predictions/nnUNetTrainerMultiTask_val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

TARGETS = {
    "whole_pancreas_dsc": 0.91,
    "lesion_dsc": 0.31,
    "macro_f1": 0.70,
}
NSD_TOLERANCE_MM = 2.0

REGIONS = {
    "whole_pancreas": lambda label: np.uint8(label > 0),
    "lesion": lambda label: np.uint8(label == 2),
}


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return the label array in (z, y, x) order plus matching voxel spacing."""
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.uint8)
    spacing = tuple(float(s) for s in reversed(image.GetSpacing()))
    return array, spacing


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    """DSC. Two empty masks count as a perfect match, following Metrics Reloaded."""
    pred_sum = int(prediction.sum())
    ref_sum = int(reference.sum())
    if pred_sum == 0 and ref_sum == 0:
        return 1.0
    intersection = int(np.logical_and(prediction, reference).sum())
    return 2.0 * intersection / (pred_sum + ref_sum)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels of a binary mask: the mask minus its 26-connected erosion."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask.astype(bool), structure=np.ones((3, 3, 3)), border_value=0)
    return np.logical_and(mask.astype(bool), np.logical_not(eroded))


def normalised_surface_distance(prediction: np.ndarray, reference: np.ndarray,
                                spacing: tuple[float, float, float],
                                tolerance_mm: float = NSD_TOLERANCE_MM) -> float:
    """NSD: fraction of both surfaces lying within `tolerance_mm` of the other.

    Implemented here because no surface-distance package is installed. The
    Euclidean distance transform is given the real anisotropic voxel spacing via
    `sampling=`, which matters at [2.0, 0.73, 0.73] mm.
    """
    pred_surface = surface_voxels(prediction)
    ref_surface = surface_voxels(reference)

    if not pred_surface.any() and not ref_surface.any():
        return 1.0
    if not pred_surface.any() or not ref_surface.any():
        return 0.0

    distance_to_reference = distance_transform_edt(~ref_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~pred_surface, sampling=spacing)

    pred_within = int((distance_to_reference[pred_surface] <= tolerance_mm).sum())
    ref_within = int((distance_to_prediction[ref_surface] <= tolerance_mm).sum())
    return (pred_within + ref_within) / (int(pred_surface.sum()) + int(ref_surface.sum()))


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate_segmentation(case_ids: list[str], prediction_folder: Path,
                          reference_folder: Path) -> tuple[dict, list[dict]]:
    per_case: list[dict] = []
    collected = {region: {"dsc": [], "nsd": []} for region in REGIONS}
    empty_reference_counts = {region: 0 for region in REGIONS}

    for case_id in case_ids:
        prediction_file = prediction_folder / f"{case_id}.nii.gz"
        reference_file = reference_folder / f"{case_id}.nii.gz"
        if not prediction_file.is_file():
            raise FileNotFoundError(f"missing prediction {prediction_file}")

        prediction, _ = read_mask(prediction_file)
        reference, spacing = read_mask(reference_file)
        if prediction.shape != reference.shape:
            raise RuntimeError(
                f"{case_id}: prediction shape {prediction.shape} != reference {reference.shape}"
            )

        row = {"case_id": case_id}
        for region, to_binary in REGIONS.items():
            predicted_region = to_binary(prediction)
            reference_region = to_binary(reference)

            if not reference_region.any():
                # convention: a case with no reference voxels for this region is
                # excluded from the region's aggregate, and counted here instead
                empty_reference_counts[region] += 1
                row[f"{region}_dsc"] = None
                row[f"{region}_nsd"] = None
                continue

            case_dsc = dice(predicted_region, reference_region)
            case_nsd = normalised_surface_distance(predicted_region, reference_region, spacing)
            row[f"{region}_dsc"] = case_dsc
            row[f"{region}_nsd"] = case_nsd
            collected[region]["dsc"].append(case_dsc)
            collected[region]["nsd"].append(case_nsd)
        per_case.append(row)

    summary = {
        region: {
            "dsc": summarise(values["dsc"]),
            "nsd": summarise(values["nsd"]),
            "nsd_tolerance_mm": NSD_TOLERANCE_MM,
            "cases_excluded_empty_reference": empty_reference_counts[region],
        }
        for region, values in collected.items()
    }
    return summary, per_case


def evaluate_classification(case_ids: list[str], classification_file: Path,
                            subtype_map: dict[str, int]) -> dict:
    predictions = json.loads(classification_file.read_text())
    reference = [subtype_map[case_id] for case_id in case_ids]
    labels = list(range(paths.NUM_CLASSES))

    results = {}
    for strategy in ("mean", "lesion_weighted"):
        key = f"pred_{strategy}"
        predicted = [predictions[case_id][key] for case_id in case_ids]
        results[strategy] = {
            "macro_f1": float(f1_score(reference, predicted, average="macro",
                                       labels=labels, zero_division=0)),
            "per_class_f1": [float(v) for v in f1_score(reference, predicted, average=None,
                                                        labels=labels, zero_division=0)],
            "balanced_accuracy": float(balanced_accuracy_score(reference, predicted)),
            "accuracy": float(np.mean(np.asarray(reference) == np.asarray(predicted))),
            "mcc": float(matthews_corrcoef(reference, predicted)),
            "confusion_matrix": confusion_matrix(reference, predicted, labels=labels).tolist(),
            "predictions": {case_id: int(p) for case_id, p in zip(case_ids, predicted)},
        }

    best = max(results, key=lambda s: results[s]["macro_f1"])
    return {"per_strategy": results, "selected_strategy": best, **results[best]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                       help="folder with predicted <case>.nii.gz and classification.json")
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--output", default=None, help="default results/metrics.json")
    parser.add_argument("--tag", default=None, help="label for this evaluation in metrics.json")
    parser.add_argument("--no-fail", action="store_true",
                       help="report shortfalls but exit 0 (for intermediate checkpoints)")
    args = parser.parse_args()

    paths.seed_everything()
    prediction_folder = Path(args.predictions)
    classification_file = prediction_folder / "classification.json"
    if not classification_file.is_file():
        raise FileNotFoundError(f"{classification_file} not found — run src/predict_multitask.py first")

    splits = json.loads(paths.splits_file.read_text())[0]
    case_ids = sorted(splits["val" if args.split == "val" else "train"])
    subtype_map = json.loads(paths.subtype_map_file.read_text())

    print(f"evaluating {len(case_ids)} {args.split} cases from {prediction_folder}")
    segmentation, per_case = evaluate_segmentation(
        case_ids, prediction_folder, paths.raw_dataset_dir / "labelsTr"
    )
    classification = evaluate_classification(case_ids, classification_file, subtype_map)

    achieved = {
        "whole_pancreas_dsc": segmentation["whole_pancreas"]["dsc"]["mean"],
        "lesion_dsc": segmentation["lesion"]["dsc"]["mean"],
        "macro_f1": classification["macro_f1"],
    }
    shortfalls = {name: {"target": target, "achieved": achieved[name],
                         "shortfall": target - achieved[name]}
                  for name, target in TARGETS.items() if achieved[name] < target}

    print("\nsegmentation")
    for region, values in segmentation.items():
        dsc, nsd = values["dsc"], values["nsd"]
        print(f"  {region:<15} DSC {dsc['mean']:.4f} +/- {dsc['std']:.4f}   "
              f"NSD@{NSD_TOLERANCE_MM:g}mm {nsd['mean']:.4f} +/- {nsd['std']:.4f}   "
              f"(n={dsc['n']}, excluded for empty reference: "
              f"{values['cases_excluded_empty_reference']})")

    print(f"\nclassification (aggregation: {classification['selected_strategy']})")
    print(f"  macro-F1 {classification['macro_f1']:.4f}   "
          f"balanced accuracy {classification['balanced_accuracy']:.4f}   "
          f"MCC {classification['mcc']:.4f}   accuracy {classification['accuracy']:.4f}")
    print(f"  per-class F1 {[round(v, 4) for v in classification['per_class_f1']]}")
    print("  confusion matrix (rows = reference, cols = predicted):")
    for row in classification["confusion_matrix"]:
        print(f"    {row}")
    for strategy, values in classification["per_strategy"].items():
        print(f"  [{strategy}] macro-F1 {values['macro_f1']:.4f}")

    print("\ntargets")
    for name, target in TARGETS.items():
        status = "MISS" if name in shortfalls else "ok"
        print(f"  {name:<20} {achieved[name]:.4f}  target {target:.2f}  [{status}]")

    metrics = {
        "tag": args.tag or prediction_folder.name,
        "split": args.split,
        "num_cases": len(case_ids),
        "predictions_folder": str(prediction_folder),
        "seed": paths.SEED,
        "targets": TARGETS,
        "achieved": achieved,
        "shortfalls": shortfalls,
        "segmentation": segmentation,
        "classification": classification,
        "per_case": per_case,
    }

    output_file = Path(args.output) if args.output else paths.results_dir / "metrics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(output_file.read_text()) if output_file.is_file() else {}
    if not isinstance(existing, dict):
        existing = {}
    existing[metrics["tag"]] = metrics
    output_file.write_text(json.dumps(existing, indent=1))
    print(f"\nwrote {output_file} (entry '{metrics['tag']}')")

    if shortfalls and not args.no_fail:
        print("\nFAILED: targets missed")
        for name, values in shortfalls.items():
            print(f"  {name}: {values['achieved']:.4f} < {values['target']:.2f} "
                  f"(short by {values['shortfall']:.4f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
