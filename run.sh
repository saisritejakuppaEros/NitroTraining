#!/bin/bash

# Train Nitro-T model. Run from Nitro-T directory.
#
# Usage:
#   bash run.sh [num_gpus] [config]
#
# Examples:
#   bash run.sh
#   bash run.sh 4 configs/exp_laionasthetic.yaml
#   bash run.sh 4 configs/exp_bs2048_repad8_maskd2_ratio0.5_mmdit.yaml
#   CUDA_VISIBLE_DEVICES=2,3,4,6 bash run.sh 4
#
# Save path is work_root/exp_name from config (e.g. diffusion_training/checkpoints/<exp_name>/).
# num_gpus must match the number of GPUs in CUDA_VISIBLE_DEVICES.

# Which physical GPUs to use (e.g. 2,3,4,6 = use GPUs 2,3,4,6; these show in nvidia-smi).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,6}

# NCCL settings (set before PyTorch/accelerate)
export NCCL_NVLS_ENABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN

set -e

num_gpus=${1:-4}
config=${2:-configs/exp_laionasthetic.yaml}

# Clean stale streaming shared memory (helps avoid NCCL/barrier issues)
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()" 2>/dev/null || true

# Count GPUs from CUDA_VISIBLE_DEVICES
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "PyTorch detects: ${GPU_COUNT} GPUs"
echo "Using num_processes: ${num_gpus}"
if [ "${GPU_COUNT}" -ne "${num_gpus}" ]; then
    echo "WARNING: num_gpus (${num_gpus}) != PyTorch GPU count (${GPU_COUNT}). Use: bash run.sh ${GPU_COUNT}"
fi

# Print save paths for this experiment (from config)
PROJECT_DIR=$(python -c "
import os
from omegaconf import OmegaConf
default_cfg = OmegaConf.load('configs/default_config.yaml')
custom_cfg = OmegaConf.load('${config}')
cfg = OmegaConf.merge(default_cfg, custom_cfg)
if hasattr(cfg, 'output_dir') and cfg.output_dir:
    project_dir = os.path.abspath(cfg.output_dir)
else:
    project_dir = os.path.abspath(os.path.join(cfg.work_root, cfg.exp_name))
print(project_dir)
")
echo "Experiment: ${config}"
echo "Save path:  ${PROJECT_DIR}"
echo "  - config:       ${PROJECT_DIR}/config.yaml"
echo "  - checkpoints:  ${PROJECT_DIR}/checkpoints/"
echo "  - validation:   ${PROJECT_DIR}/validation_images/"
echo "  - tensorboard:  ${PROJECT_DIR}/"
echo ""

# Use accelerate config; ACCELERATE_USE_CUDA_VISIBLE_DEVICES ensures proper GPU mapping
ACCELERATE_USE_CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes ${num_gpus} \
    train.py --config ${config}
