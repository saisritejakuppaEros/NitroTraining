# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
accelerate launch --config_file configs/accelerate_config.yaml train.py --config configs/exp_bs2048_repad8_maskd2_ratio0.5_mmdit.yaml
