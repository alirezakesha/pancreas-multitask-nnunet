#!/usr/bin/env python3
"""Milestone 5: baseline vs optimized inference timing on the validation set.

Baseline    TTA disabled, ``tile_step_size=0.5``, nnU-Net's stock export path.
Optimized   identical, except the predicted logits are resampled and argmaxed on
            the GPU (``--fast`` in src/predict_multitask.py).

Both arms run on the same GPU over the same cases, warmup pass first, then
`--repeats` timed repeats. Reports mean +/- std wall clock, the percentage
improvement, and the change in whole-pancreas DSC so a speedup that costs
accuracy is visible.

Two strategies from SPEC 7.2 are deliberately not used:
  * raising ``step_size`` — the assessment ReadMe requires a strategy "beyond
    disabling TTA and increasing step size", and on this dataset it would gain
    nothing anyway (see NOTES.md: 185 tiles at 0.5 vs 184 at 0.7).
  * FP16 — no tensor cores on Pascal (SPEC P2/P3).

    CUDA_VISIBLE_DEVICES=0 python src/benchmark_inference.py --trainer nnUNetTrainerMultiTask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluate import evaluate_segmentation  # noqa: E402
from predict_multitask import (  # noqa: E402
    build_predictor,
    case_ids_for_split,
    predict_cases,
    resolve_model_folder,
)

MAX_ALLOWED_DSC_DROP = 0.005


def run_arm(name: str, fast: bool, model_folder: Path, checkpoint: str, fold: int,
            case_ids: list[str], image_folder: Path, output_root: Path,
            step_size: float, use_tta: bool, repeats: int) -> dict:
    output_folder = output_root / name
    predictor = build_predictor(model_folder, checkpoint, fold, step_size, use_tta,
                                torch.device('cuda'))

    print(f"\n[{name}] step_size={step_size} tta={use_tta} gpu_export={fast}")
    start = perf_counter()
    predict_cases(predictor, case_ids, image_folder, output_folder, fast=fast)
    print(f"  warmup {perf_counter() - start:.2f} s")

    totals, stage_totals = [], []
    for repeat in range(repeats):
        torch.cuda.synchronize()
        start = perf_counter()
        result = predict_cases(predictor, case_ids, image_folder, output_folder, fast=fast)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        totals.append(elapsed)
        stage_totals.append({k: float(np.sum(v)) for k, v in result['stage_times'].items()})
        print(f"  repeat {repeat + 1}/{repeats}: {elapsed:.2f} s  "
              + "  ".join(f"{k} {v:.2f}s" for k, v in stage_totals[-1].items()))

    del predictor
    torch.cuda.empty_cache()

    return {
        'name': name,
        'gpu_export': fast,
        'step_size': step_size,
        'tta': use_tta,
        'repeats': repeats,
        'num_cases': len(case_ids),
        'total_seconds_mean': float(np.mean(totals)),
        'total_seconds_std': float(np.std(totals)),
        'seconds_per_case_mean': float(np.mean(totals) / len(case_ids)),
        'all_totals': [float(t) for t in totals],
        'stage_seconds_mean': {k: float(np.mean([s[k] for s in stage_totals]))
                               for k in stage_totals[0]},
        'output_folder': str(output_folder),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', default='nnUNetTrainerMultiTask')
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--fold', type=int, default=paths.FOLD)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--step-size', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()

    paths.seed_everything()
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device visible')
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"(pin the GPU with CUDA_VISIBLE_DEVICES and keep it otherwise idle)")

    case_ids, image_folder = case_ids_for_split('val')
    if args.limit:
        case_ids = case_ids[: args.limit]
    model_folder = resolve_model_folder(args.trainer)
    output_root = Path(args.output_root) if args.output_root else \
        paths.results_dir / 'predictions' / f'benchmark_{args.trainer}'

    arms = {}
    for name, fast in (('baseline', False), ('optimized', True)):
        arms[name] = run_arm(name, fast, model_folder, args.checkpoint, args.fold,
                             case_ids, image_folder, output_root, args.step_size,
                             args.tta, args.repeats)

    baseline, optimized = arms['baseline'], arms['optimized']
    improvement = 100.0 * (1.0 - optimized['total_seconds_mean'] / baseline['total_seconds_mean'])

    reference_folder = paths.raw_dataset_dir / 'labelsTr'
    dsc = {}
    for name in arms:
        summary, _ = evaluate_segmentation(case_ids, Path(arms[name]['output_folder']),
                                           reference_folder)
        dsc[name] = {region: summary[region]['dsc']['mean'] for region in summary}

    dsc_drop = dsc['baseline']['whole_pancreas'] - dsc['optimized']['whole_pancreas']
    lesion_drop = dsc['baseline']['lesion'] - dsc['optimized']['lesion']

    print(f"\nbaseline   {baseline['total_seconds_mean']:.2f} +/- {baseline['total_seconds_std']:.2f} s "
          f"({baseline['seconds_per_case_mean']:.3f} s/case)")
    print(f"optimized  {optimized['total_seconds_mean']:.2f} +/- {optimized['total_seconds_std']:.2f} s "
          f"({optimized['seconds_per_case_mean']:.3f} s/case)")
    print(f"improvement: {improvement:.2f}%   (target >= 10%)")
    print(f"stage breakdown (mean seconds over the whole validation set):")
    for name in arms:
        print(f"  {name:<10} " + "  ".join(f"{k} {v:.2f}" for k, v in arms[name]['stage_seconds_mean'].items()))
    print(f"whole-pancreas DSC {dsc['baseline']['whole_pancreas']:.5f} -> "
          f"{dsc['optimized']['whole_pancreas']:.5f}  (drop {dsc_drop:+.5f}, allowed {MAX_ALLOWED_DSC_DROP})")
    print(f"lesion DSC         {dsc['baseline']['lesion']:.5f} -> {dsc['optimized']['lesion']:.5f} "
          f"(drop {lesion_drop:+.5f})")

    passed = improvement >= 10.0 and dsc_drop <= MAX_ALLOWED_DSC_DROP
    payload = {
        'trainer': args.trainer,
        'checkpoint': args.checkpoint,
        'num_cases': len(case_ids),
        'gpu': torch.cuda.get_device_name(0),
        'strategy': 'GPU trilinear resampling + GPU argmax of the predicted logits '
                    '(replaces nnU-Net CPU resampling of all logit channels)',
        'baseline': baseline,
        'optimized': optimized,
        'improvement_percent': improvement,
        'dsc': dsc,
        'whole_pancreas_dsc_drop': dsc_drop,
        'lesion_dsc_drop': lesion_drop,
        'max_allowed_dsc_drop': MAX_ALLOWED_DSC_DROP,
        'passed': bool(passed),
    }

    metrics_file = paths.results_dir / 'metrics.json'
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
    if not isinstance(existing, dict):
        existing = {}
    existing['inference_benchmark'] = payload
    metrics_file.write_text(json.dumps(existing, indent=1))
    print(f"\nwrote {metrics_file} (entry 'inference_benchmark')")

    if not passed:
        print('FAILED: needs >=10% improvement with whole-pancreas DSC drop <= 0.005')
        sys.exit(1)


if __name__ == '__main__':
    main()
