#!/usr/bin/env bash
# One-line-per-run summary of the Milestone 4 training runs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$REPO/nnUNet_results/Dataset501_PancreasQuiz"

printf '%-32s %8s %10s %10s %10s %10s\n' TRAINER EPOCH TRAIN_LOSS VAL_LOSS MACRO_F1 EPOCH_S
for dir in "$RESULTS"/*__nnUNetResEncUNetMPlans__3d_fullres/fold_0; do
    [[ -d "$dir" ]] || continue
    trainer=$(basename "$(dirname "$dir")")
    trainer=${trainer%%__*}

    log=$(ls -t "$dir"/training_log_*.txt 2> /dev/null | head -1) || continue
    [[ -n "$log" ]] || continue

    # nnU-Net's print_to_log_file leaves a trailing space, so no '$' anchor here
    epoch=$(grep -oE ': Epoch [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+' || echo '-')
    train_loss=$(grep -oE 'train_loss -?[0-9.]+' "$log" | tail -1 | awk '{print $2}' || echo '-')
    val_loss=$(grep -oE 'val_loss -?[0-9.]+' "$log" | tail -1 | awk '{print $2}' || echo '-')
    macro_f1=$(grep -oE 'val_macro_f1 [0-9.]+' "$log" | tail -1 | awk '{print $2}' || echo '-')
    epoch_s=$(grep -oE 'Epoch time: [0-9.]+' "$log" | tail -1 | awk '{print $3}' || echo '-')

    printf '%-32s %8s %10s %10s %10s %10s\n' \
        "$trainer" "${epoch:--}" "${train_loss:--}" "${val_loss:--}" "${macro_f1:--}" "${epoch_s:--}"
done

echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader 2> /dev/null || true
