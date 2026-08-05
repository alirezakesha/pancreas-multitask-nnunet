#!/usr/bin/env bash
# Source this after activating the venv:
#   source .venv/bin/activate && source setup_nnunet_env.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export nnUNet_raw="${ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${ROOT}/nnUNet_preprocessed"
export nnUNet_results="${ROOT}/nnUNet_results"

# External trainers live outside nnUNet/ (e.g. custom/MultiTaskTrainer.py)
export nnUNet_extTrainer="${ROOT}/custom"

# TITAN Xp / Pascal (SM 6.1): torch.compile uses Triton, which needs SM >= 7.0
export nnUNet_compile=false

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNet_extTrainer=${nnUNet_extTrainer}"
echo "nnUNet_compile=${nnUNet_compile}"
