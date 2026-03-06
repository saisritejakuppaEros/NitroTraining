# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

# CUDA_VISIBLE_DEVICES=2,3,4,6 bash run.sh 4

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
accelerate launch --config_file configs/accelerate_config.yaml train.py --config configs/exp_bs2048_repad8_maskd2_ratio0.5_mmdit.yaml
