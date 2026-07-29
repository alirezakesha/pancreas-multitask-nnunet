#!/usr/bin/env python3
"""Milestone 2 diagnostic: AMP on vs off, and the memory cost of turning it off.

SPEC P2 claims AMP buys memory rather than speed on Pascal (no tensor cores) and
that disabling it risks OOM on a 12 GB card. This measures both instead of
assuming: forward + backward + optimizer step on the real planned architecture
and patch size, with synthetic data so the dataloader is not part of the timing.

    CUDA_VISIBLE_DEVICES=2 python src/benchmark_amp.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import autocast  # noqa: E402

from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402


def time_iterations(network, batch_size, patch_size, num_classes, amp: bool,
                    iterations: int, warmup: int) -> dict:
    device = torch.device("cuda")
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    scaler = torch.GradScaler("cuda") if amp else None
    loss_fn = torch.nn.CrossEntropyLoss()

    data = torch.randn(batch_size, 1, *patch_size, device=device)
    target = torch.randint(0, num_classes, (batch_size, *patch_size), device=device)

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(warmup + iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        optimizer.zero_grad(set_to_none=True)
        start.record()
        if amp:
            with autocast("cuda", enabled=True):
                output = network(data)
                loss = loss_fn(output[0] if isinstance(output, (list, tuple)) else output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = network(data)
            loss = loss_fn(output[0] if isinstance(output, (list, tuple)) else output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
            optimizer.step()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end) / 1000.0)

    return {
        "amp": amp,
        "iterations": iterations,
        "seconds_per_iteration_mean": float(np.mean(times)),
        "seconds_per_iteration_std": float(np.std(times)),
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", default="nnUNetTrainerMultiTask")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    paths.seed_everything()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible")

    plans = json.loads((paths.preprocessed_dataset_dir / f"{paths.PLANS_NAME}.json").read_text())
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(paths.CONFIGURATION)
    patch_size = tuple(configuration_manager.patch_size)
    batch_size = configuration_manager.batch_size

    print(f"device: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
    print(f"torch {torch.__version__}, arch list {torch.cuda.get_arch_list()}")
    print(f"trainer {args.trainer}, batch {batch_size}, patch {patch_size}")

    trainer_class = recursive_find_trainer_class_by_name(args.trainer)
    results = []
    for amp in (True, False):
        network = trainer_class.build_network_architecture(
            plans_manager, configuration_manager, 1, paths.NUM_CLASSES, True
        ).cuda()
        if hasattr(network, "return_cls"):
            network.return_cls = False
        network.train()
        try:
            result = time_iterations(network, batch_size, patch_size, paths.NUM_CLASSES,
                                     amp, args.iterations, args.warmup)
            print(f"  AMP={'on ' if amp else 'off'}: "
                  f"{result['seconds_per_iteration_mean']:.3f} +/- {result['seconds_per_iteration_std']:.3f} s/iter, "
                  f"peak {result['peak_vram_gib']:.2f} GiB")
        except torch.OutOfMemoryError:
            result = {"amp": amp, "error": "CUDA out of memory"}
            print(f"  AMP={'on ' if amp else 'off'}: OOM on a 12 GB card")
        results.append(result)
        del network
        torch.cuda.empty_cache()

    on, off = results
    if "error" not in on and "error" not in off:
        speedup = off["seconds_per_iteration_mean"] / on["seconds_per_iteration_mean"]
        memory = off["peak_vram_gib"] / on["peak_vram_gib"]
        print(f"\nAMP speedup: {speedup:.2f}x    AMP memory saving: {memory:.2f}x")

    out = paths.results_dir / "amp_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
