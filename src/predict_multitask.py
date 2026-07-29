#!/usr/bin/env python3
"""Milestone 5: dual-output inference — segmentation plus case-level subtype.

Both heads come out of a single pass over the sliding-window tiles. Only
``_internal_maybe_mirror_and_predict`` is overridden, so the Gaussian-weighted
patch aggregation, padding and geometry handling stay exactly nnU-Net's.

Case-level classification averages the per-tile softmax; a single tile is not a
case-level prediction because most tiles contain little or no lesion. Two
aggregations are computed and both are written out, so Milestone 6 can pick the
one that scores better on the validation set:

* ``mean``            — plain mean of the per-tile softmax
* ``lesion_weighted`` — weighted by each tile's predicted lesion voxel count

``--fast`` enables the Milestone 5 speedup (see ``export_prediction_on_gpu``).

    python src/predict_multitask.py --split val  --trainer nnUNetTrainerMultiTask
    python src/predict_multitask.py --split test --trainer nnUNetTrainerMultiTask --fast
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image  # noqa: E402

from nnunetv2.inference.export_prediction import export_prediction_from_logits  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

AGGREGATIONS = ("mean", "lesion_weighted")
LESION_LABEL = 2


class MultiTaskPredictor(nnUNetPredictor):
    """nnUNetPredictor that also collects per-tile classification probabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_cls_probs: list[np.ndarray] = []
        self.tile_lesion_voxels: list[int] = []

    def reset_tile_records(self) -> None:
        self.tile_cls_probs = []
        self.tile_lesion_voxels = []

    @property
    def collects_classification(self) -> bool:
        return hasattr(self.network, "return_cls")

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Base implementation, but splitting the (segmentation, classification) tuple.

        Returns the segmentation logits alone so the caller is unchanged, and
        records this tile's classification softmax and predicted lesion volume.
        """
        if not self.collects_classification or not self.network.return_cls:
            return super()._internal_maybe_mirror_and_predict(x)

        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction, cls_logits = self.network(x)
        cls_prob = torch.softmax(cls_logits.float(), dim=1)

        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                mirrored, mirrored_cls = self.network(torch.flip(x, axes))
                prediction += torch.flip(mirrored, axes)
                cls_prob += torch.softmax(mirrored_cls.float(), dim=1)
            prediction /= (len(axes_combinations) + 1)
            cls_prob /= (len(axes_combinations) + 1)

        self.tile_cls_probs.append(cls_prob[0].cpu().numpy())
        self.tile_lesion_voxels.append(int((prediction[0].argmax(0) == LESION_LABEL).sum().item()))
        return prediction

    def aggregate_classification(self) -> dict:
        """Combine the per-tile softmax into one case-level prediction per strategy."""
        if not self.tile_cls_probs:
            raise RuntimeError("no tiles recorded; was return_cls enabled?")

        probs = np.stack(self.tile_cls_probs)
        lesion_voxels = np.asarray(self.tile_lesion_voxels, dtype=np.float64)

        result = {"num_tiles": int(probs.shape[0]),
                  "tile_lesion_voxels": [int(v) for v in lesion_voxels]}

        mean_prob = probs.mean(axis=0)
        result["prob_mean"] = mean_prob.tolist()
        result["pred_mean"] = int(mean_prob.argmax())

        # fall back to the plain mean when no tile predicted any lesion
        if lesion_voxels.sum() > 0:
            weights = lesion_voxels / lesion_voxels.sum()
            weighted_prob = (probs * weights[:, None]).sum(axis=0)
        else:
            weighted_prob = mean_prob
        result["prob_lesion_weighted"] = weighted_prob.tolist()
        result["pred_lesion_weighted"] = int(weighted_prob.argmax())
        return result


def export_prediction_on_gpu(predicted_logits: torch.Tensor, properties: dict,
                             plans_manager: PlansManager, label_manager,
                             output_file_truncated: str, file_ending: str) -> None:
    """Milestone 5 speedup: resample and argmax the logits on the GPU.

    nnU-Net's export path (``convert_predicted_logits_to_segmentation_with_correct_shape``)
    resamples all three float logit channels back to the pre-resampling shape on
    the **CPU** with ``resampling_fn_probabilities``, then argmaxes. On this
    dataset the sliding window is only ~2.4 tiles per case, so that CPU resample
    — not the network — dominates inference time.

    Here the same trilinear interpolation runs on the GPU via
    ``F.interpolate`` and the argmax is taken on the GPU too, so only a uint8
    label map crosses back to the host. Reverting the crop and the transpose is
    unchanged.
    """
    target_shape = tuple(int(i) for i in properties['shape_after_cropping_and_before_resampling'])

    logits = predicted_logits if predicted_logits.is_cuda else predicted_logits.cuda()
    if tuple(logits.shape[1:]) != target_shape:
        logits = F.interpolate(logits[None].float(), size=target_shape,
                               mode='trilinear', align_corners=False)[0]
    segmentation = logits.argmax(0).to(torch.uint8).cpu().numpy()
    del logits

    reverted = np.zeros(properties['shape_before_cropping'], dtype=np.uint8)
    reverted = insert_crop_into_image(reverted, segmentation, properties['bbox_used_for_cropping'])
    reverted = reverted.transpose(plans_manager.transpose_backward)

    writer = plans_manager.image_reader_writer_class()
    writer.write_seg(reverted, output_file_truncated + file_ending, properties)


def build_predictor(model_folder: Path, checkpoint: str, fold: int, tile_step_size: float,
                    use_tta: bool, device: torch.device) -> MultiTaskPredictor:
    predictor = MultiTaskPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=(fold,),
                                                   checkpoint_name=checkpoint)
    if predictor.collects_classification:
        predictor.network.return_cls = True
    predictor.network = predictor.network.to(device).eval()
    return predictor


def predict_cases(predictor: MultiTaskPredictor, case_ids: list[str], image_folder: Path,
                  output_folder: Path, fast: bool = False, write_segmentations: bool = True) -> dict:
    """Predict every case, returning classification results and per-stage timings."""
    plans_manager = predictor.plans_manager
    configuration_manager = predictor.configuration_manager
    dataset_json = predictor.dataset_json
    label_manager = predictor.label_manager
    file_ending = dataset_json['file_ending']
    preprocessor = configuration_manager.preprocessor_class(verbose=False)

    output_folder.mkdir(parents=True, exist_ok=True)
    classification: dict[str, dict] = {}
    stage_times = {'preprocess': [], 'predict': [], 'export': []}

    for case_id in case_ids:
        image_file = image_folder / f"{case_id}_0000{file_ending}"

        start = perf_counter()
        data, _, properties = preprocessor.run_case([str(image_file)], None, plans_manager,
                                                    configuration_manager, dataset_json)
        data = torch.from_numpy(data)
        stage_times['preprocess'].append(perf_counter() - start)

        start = perf_counter()
        predictor.reset_tile_records()
        logits = predictor.predict_sliding_window_return_logits(data)
        torch.cuda.synchronize()
        stage_times['predict'].append(perf_counter() - start)

        start = perf_counter()
        if write_segmentations:
            output_truncated = str(output_folder / case_id)
            if fast:
                export_prediction_on_gpu(logits, properties, plans_manager, label_manager,
                                         output_truncated, file_ending)
            else:
                export_prediction_from_logits(logits.cpu(), properties, configuration_manager,
                                             plans_manager, dataset_json, output_truncated,
                                             save_probabilities=False)
        stage_times['export'].append(perf_counter() - start)

        classification[case_id] = predictor.aggregate_classification()
        del logits, data

    return {'classification': classification, 'stage_times': stage_times}


def resolve_model_folder(trainer: str) -> Path:
    folder = (paths.nnUNet_results / paths.DATASET_NAME /
              f"{trainer}__{paths.PLANS_NAME}__{paths.CONFIGURATION}")
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found — has {trainer} been trained?")
    return folder


def case_ids_for_split(split: str) -> tuple[list[str], Path]:
    if split == 'test':
        image_folder = paths.raw_dataset_dir / 'imagesTs'
        ids = sorted(p.name[: -len('_0000.nii.gz')] for p in image_folder.glob('*_0000.nii.gz'))
        return ids, image_folder
    splits = json.loads(paths.splits_file.read_text())[0]
    key = 'val' if split == 'val' else 'train'
    return sorted(splits[key]), paths.raw_dataset_dir / 'imagesTr'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'train', 'test'), default='val')
    parser.add_argument('--trainer', default='nnUNetTrainerMultiTask')
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--fold', type=int, default=paths.FOLD)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--step-size', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true', help='enable mirroring TTA (off by default)')
    parser.add_argument('--fast', action='store_true', help='GPU resampling/argmax export')
    parser.add_argument('--limit', type=int, default=None, help='only the first N cases (debugging)')
    args = parser.parse_args()

    paths.seed_everything()
    device = torch.device('cuda')

    case_ids, image_folder = case_ids_for_split(args.split)
    if args.limit:
        case_ids = case_ids[: args.limit]

    output_folder = Path(args.output_dir) if args.output_dir else \
        paths.results_dir / 'predictions' / f'{args.trainer}_{args.split}{"_fast" if args.fast else ""}'

    model_folder = resolve_model_folder(args.trainer)
    predictor = build_predictor(model_folder, args.checkpoint, args.fold,
                               args.step_size, args.tta, device)

    print(f"{args.trainer}: {len(case_ids)} {args.split} cases -> {output_folder}")
    print(f"  step_size={args.step_size} tta={args.tta} fast={args.fast} checkpoint={args.checkpoint}")

    start = perf_counter()
    result = predict_cases(predictor, case_ids, image_folder, output_folder, fast=args.fast)
    total = perf_counter() - start

    classification_file = output_folder / 'classification.json'
    classification_file.write_text(json.dumps(result['classification'], indent=1, sort_keys=True))

    stage_times = result['stage_times']
    print(f"  total {total:.1f} s for {len(case_ids)} cases ({total / len(case_ids):.2f} s/case)")
    for stage, values in stage_times.items():
        print(f"    {stage:<11} {np.sum(values):7.2f} s total  {np.mean(values):.3f} s/case")
    agreement = sum(1 for v in result['classification'].values()
                    if v['pred_mean'] == v['pred_lesion_weighted'])
    print(f"  aggregation strategies agree on {agreement}/{len(case_ids)} cases")
    print(f"  wrote {classification_file}")


if __name__ == '__main__':
    main()
