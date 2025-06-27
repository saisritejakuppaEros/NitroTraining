# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]


import torch


def build_optimizer(optimizer_type, optimizer_params={}):
    if optimizer_type.lower() in ["adamw"]:
        optimizer_class = torch.optim.AdamW
        optimizer_kwargs = {"lr": 1e-4, "weight_decay": 3e-2, "eps": 1e-10}
        optimizer_kwargs.update(optimizer_params)
    else:
        raise Exception()

    return optimizer_class, optimizer_kwargs
