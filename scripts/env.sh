#!/usr/bin/env bash
# Source this before any nnU-Net command:
#   source .venv/bin/activate && source scripts/env.sh
#
# Every entry point in src/ also calls src/paths.py, which reproduces these
# defaults in-process, so scripts still work if this file was not sourced.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO

# Local secrets (WANDB_API_KEY, etc.). Never commit .env.
if [[ -f "${REPO}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO}/.env"
    set +a
fi

export nnUNet_raw="${REPO}/nnUNet_raw"
export nnUNet_preprocessed="${REPO}/nnUNet_preprocessed"
export nnUNet_results="${REPO}/nnUNet_results"

# Trainer subclasses live in src/trainers/ and are discovered from there, so
# nothing inside nnunetv2/ is ever modified. See NOTES.md (deviation D2).
export nnUNet_extTrainer="${REPO}/src/trainers"

# Titan Xp is sm_61: torch.compile needs Triton, which requires sm_70+.
export nnUNet_compile=f

# nnU-Net 2.8.1 has a built-in wandb logger (nnunetv2/training/logging/nnunet_logger.py).
# It resumes the same run on --c, so no second wandb.init is needed.
# Auth: set WANDB_API_KEY in repo-root .env (gitignored), or run `wandb login`.
export nnUNet_wandb_enabled="${nnUNet_wandb_enabled:-1}"
export nnUNet_wandb_project="${nnUNet_wandb_project:-pancreas-multitask-nnunet}"
export nnUNet_wandb_mode="${nnUNet_wandb_mode:-online}"

# 20 cores / 3 concurrent training jobs. Override for a single run.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-6}"

# Keeps matplotlib from warning on every import ($HOME is not writable here).
export MPLCONFIGDIR="${REPO}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

export PYTHONHASHSEED=1234

echo "REPO=${REPO}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNet_extTrainer=${nnUNet_extTrainer}"
echo "nnUNet_compile=${nnUNet_compile}"
echo "nnUNet_n_proc_DA=${nnUNet_n_proc_DA}"
