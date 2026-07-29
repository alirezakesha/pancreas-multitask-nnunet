#!/usr/bin/env bash
# Source this after activating the venv:
#   source .venv/bin/activate && source setup_nnunet_env.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export nnUNet_raw="${ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${ROOT}/nnUNet_preprocessed"
export nnUNet_results="${ROOT}/nnUNet_results"

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
