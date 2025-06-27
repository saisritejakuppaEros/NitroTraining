# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]


from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderDC
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.nitro_mmdit_pipeline import NitroMMDiTPipeline
from core.models.nitro_mmdit_model import NitroMMDiTModel
from core.models.nitro_dit_pipeline import NitroDiTPipeline
from core.models.nitro_dit_model import NitroDiTModel


def build_model(cfg):
    if cfg.model.model_architecture == "DiT":
        model = NitroDiTModel(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            sample_size=cfg.model.latent_size,
            patch_size=cfg.model.patch_size,
            caption_channels=cfg.model.caption_channels,
            repa_depth=cfg.model.repa_depth,
            projector_dim=cfg.model.projector_dim,
            z_dims=cfg.model.z_dims,
            interpolation_scale=cfg.model.interpolation_scale,
        )

        if cfg.model.use_flash_attn:
            from core.models.flash_attn_processor import AttnProcessor2_0_FA

            model.set_attn_processor(AttnProcessor2_0_FA())

    elif cfg.model.model_architecture == "MMDiT":
        model = NitroMMDiTModel(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            sample_size=cfg.model.latent_size,
            patch_size=cfg.model.patch_size,
            caption_channels=cfg.model.caption_channels,
            dual_attention_layers=tuple(cfg.model.dual_attention_layers),
            qk_norm=cfg.model.qk_norm,
            repa_depth=cfg.model.repa_depth,
            projector_dim=cfg.model.projector_dim,
            z_dims=cfg.model.z_dims,
            interpolation_scale=cfg.model.interpolation_scale,
        )

        if cfg.model.use_flash_attn:
            from core.models.flash_attn_processor import JointAttnProcessor2_0_FA

            model.set_attn_processor(JointAttnProcessor2_0_FA())

    return model


def build_pipeline(transformer, cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = AutoModelForCausalLM.from_pretrained(cfg.model.text_encoder_name).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=cfg.training.num_train_timesteps,
        shift=cfg.training.fm_shift,
    )
    vae = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers").to(device)
    vae.scaling_factor = cfg.training.scaling_factor

    if cfg.model.model_architecture == "DiT":
        pipe = NitroDiTPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    if cfg.model.model_architecture == "MMDiT":
        pipe = NitroMMDiTPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    return pipe
