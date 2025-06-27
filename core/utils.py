# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]


import einops
import torch
import torch.nn as nn


# get sigmas from noise_scheduler
def get_sigmas(timesteps, noise_scheduler, n_dim=4):
    sigmas = noise_scheduler.sigmas
    schedule_timesteps = noise_scheduler.timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices]
    sigma = sigma.view(-1, *([1] * (n_dim - 1)))
    return sigma.clone()


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def token_drop(
    y,
    y_mask,
    uncond_prompt_embeds,
    uncond_prompt_attention_mask,
    class_dropout_prob=0.1,
):
    """
    Drops labels to enable classifier-free guidance.
    """
    drop_ids = torch.rand(y.shape[0]).cuda() < class_dropout_prob
    y = torch.where(drop_ids[:, None, None], uncond_prompt_embeds, y)
    y_mask = torch.where(drop_ids[:, None], uncond_prompt_attention_mask, y_mask)
    return y, y_mask


def get_null_embed(pickle_file, device):
    data = torch.load(pickle_file)
    uncond_prompt_embeds = data["uncond_prompt_embeds"].to(device)
    uncond_prompt_attention_mask = data["uncond_prompt_attention_mask"].to(device)
    return uncond_prompt_embeds, uncond_prompt_attention_mask


def move_to_device(src_dict, device):
    """
    recursively move a dict of tensors to a device
    """
    assert isinstance(src_dict, dict)
    for k in src_dict:
        if isinstance(src_dict[k], torch.Tensor):
            src_dict[k] = src_dict[k].to(device)
        elif isinstance(src_dict[k], dict):
            move_to_device(src_dict[k], device)


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    ids_keep = torch.argsort(noise, dim=1)[:, :len_keep]
    ids_keep, _ = torch.sort(ids_keep, dim=1)
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked, ids_keep, len_keep


def patchify_and_apply_mask(im, patch_size, ids_keep=None):
    # Conceptually: N C H W -> N C H/P P W/P P -> N C H/P W/P P P -> N C T PP
    im = im.reshape(
        shape=(
            im.shape[0],
            im.shape[1],
            im.shape[2] // patch_size,
            patch_size,
            im.shape[3] // patch_size,
            patch_size,
        )
    )
    im = einops.rearrange(im, "n c h p w q -> n c (h w) (p q)")
    N, C, T, PP = im.shape
    if ids_keep is not None:
        im = torch.gather(im, dim=2, index=ids_keep.reshape(N, 1, -1, 1).repeat(1, C, 1, PP))
    return im
