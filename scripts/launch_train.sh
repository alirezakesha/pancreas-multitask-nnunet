#!/usr/bin/env bash
# Train nnUNetTrainerMultiTask (GAP head, lambda_cls=0.5) on one GPU.
#
#   ./scripts/launch_train.sh
#   ./scripts/launch_train.sh --c
#   ./scripts/launch_train.sh --gpu 0 --offline
#
# On Google Colab, prefer calling nnUNetv2_train directly (see README).
# This script uses tmux locally, or nohup when tmux/Colab is unavailable.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO/.venv/bin/activate"
fi
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh" > /dev/null

RESUME_FLAG=""
GPU=0
WANDB_MODE_CHOICE=online
while [[ $# -gt 0 ]]; do
    case "$1" in
        --c) RESUME_FLAG="--c" ;;
        --offline) WANDB_MODE_CHOICE=offline ;;
        --gpu)
            shift
            GPU="${1:?--gpu needs a device index}"
            ;;
        *)
            echo "unknown argument: $1 (expected --c, --offline, --gpu N)" >&2
            exit 1
            ;;
    esac
    shift
done

export nnUNet_wandb_mode="$WANDB_MODE_CHOICE"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_NAME="${WANDB_NAME:-nnUNetTrainerMultiTask}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gap-only}"

DATASET=501
CONFIGURATION=3d_fullres
FOLD=0
PLANS=nnUNetResEncUNetMPlans
TRAINER=nnUNetTrainerMultiTask

mkdir -p "$REPO/logs"
log="$REPO/logs/train_${TRAINER}.log"

train_cmd="nnUNetv2_train $DATASET $CONFIGURATION $FOLD -tr $TRAINER -p $PLANS --npz $RESUME_FLAG"

echo "trainer=$TRAINER  gpu=$GPU  wandb=$nnUNet_wandb_mode  n_proc_DA=$nnUNet_n_proc_DA"
echo "log=$log"
echo "cmd: $train_cmd"

use_tmux=0
if command -v tmux > /dev/null \
    && [[ -z "${COLAB_RELEASE_TAG:-}" ]] \
    && [[ -z "${COLAB_GPU:-}" ]]; then
    use_tmux=1
fi

if [[ "$use_tmux" -eq 1 ]]; then
    session="mt_train"
    if tmux has-session -t "$session" 2> /dev/null; then
        echo "tmux session '$session' already exists — attach with: tmux attach -t $session" >&2
        exit 1
    fi
    tmux new-session -d -s "$session" \
        "source '$REPO/.venv/bin/activate' 2>/dev/null; \
         source '$REPO/scripts/env.sh' > /dev/null; \
         export nnUNet_n_proc_DA=$nnUNet_n_proc_DA; \
         export PYTHONUNBUFFERED=1; \
         export CUDA_VISIBLE_DEVICES=$GPU; \
         export nnUNet_wandb_mode='$nnUNet_wandb_mode'; \
         export WANDB_NAME='$WANDB_NAME'; \
         export WANDB_RUN_GROUP='$WANDB_RUN_GROUP'; \
         $train_cmd 2>&1 | tee '$log'; \
         echo EXIT=\$?; sleep 86400"
    echo "launched in tmux session '$session'. monitor: tmux attach -t $session"
else
    # shellcheck disable=SC2086
    nohup $train_cmd > "$log" 2>&1 &
    echo "launched pid $! (nohup). monitor: tail -f $log"
fi
