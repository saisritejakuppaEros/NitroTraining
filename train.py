# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

import argparse
import datetime
import logging
import os
from copy import deepcopy

import einops
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from diffusers import FlowMatchEulerDiscreteScheduler, get_constant_schedule_with_warmup
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.loss import calculate_diffusion_loss, calculate_repa_loss
from core.models.model_utils import build_model, build_pipeline
from core.optimizer import build_optimizer
from core.utils import (
    ema_update,
    get_null_embed,
    get_sigmas,
    move_to_device,
    patchify_and_apply_mask,
    token_drop,
)

logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)


def log_validation(transformer_state_dict, accelerator, cfg, step=0, save_dir=None):
    from data.validation_prompts import validation_prompts

    logger.info("Running validation... ")
    torch.cuda.empty_cache()

    transformer = build_model(cfg)
    transformer = transformer.to(accelerator.device)
    transformer.eval()  # very important
    transformer.load_state_dict(transformer_state_dict)

    pipe = build_pipeline(transformer, cfg, accelerator.device)

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.validation.seed)
    latents = torch.randn(
        size=(
            cfg.validation.num_images_per_prompt,
            cfg.model.in_channels,
            cfg.model.latent_size,
            cfg.model.latent_size,
        ),
        generator=generator,
        device=accelerator.device,
    )

    image_logs = []
    for prompt in validation_prompts:
        images = []
        for i in range(cfg.validation.num_images_per_prompt):
            image = pipe(
                prompt,
                latents=latents[i].unsqueeze(0),
                num_inference_steps=cfg.validation.num_inference_timesteps,
                height=cfg.model.latent_size * 32,  # hard-code the scale factor for now
                width=cfg.model.latent_size * 32,  # hard-code the scale factor for now
                generator=generator,
                guidance_scale=cfg.validation.cfg,
            ).images[0]
            images.append(image)
        image_logs.append({"validation_prompt": prompt, "images": images})

    if save_dir:
        step_dir = os.path.join(save_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        for i, log in enumerate(image_logs):
            prompt = log["validation_prompt"]
            for j, img in enumerate(log["images"]):
                safe_prompt = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt[:50])
                path = os.path.join(step_dir, f"{i:02d}_{safe_prompt}_{j}.png")
                img.save(path)
        logger.info(f"Saved validation images to {step_dir}")

    return image_logs


def main(args):

    default_cfg = OmegaConf.load("configs/default_config.yaml")
    custom_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, custom_cfg)

    project_dir = (
        cfg.output_dir
        if hasattr(cfg, "output_dir") and cfg.output_dir
        else os.path.join(cfg.work_root, cfg.exp_name)
    )

    # init accelerator
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=project_dir,
        kwargs_handlers=[init_handler],
        step_scheduler_with_optimizer=False,
    )
    logger.info(accelerator.state)

    logger.info(f"Config: {cfg}")

    if accelerator.is_main_process:
        os.makedirs(project_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=os.path.join(project_dir, "config.yaml"))
        
        # Flatten config to scalar values only for TensorBoard
        def flatten_config(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_config(v, new_key, sep=sep).items())
                elif isinstance(v, (int, float, str, bool)):
                    items.append((new_key, v))
                elif isinstance(v, list) and len(v) > 0 and all(isinstance(x, (int, float, str, bool)) for x in v):
                    items.append((new_key, str(v)))
            return dict(items)
        
        flat_config = flatten_config(OmegaConf.to_container(cfg, resolve=True))
        accelerator.init_trackers(
            project_name=cfg.project_name,
            config=flat_config,
        )

    total_batch_size = (
        cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.training.max_iters}")

    if cfg.training.use_precomputed_text_embeddings:
        uncond_prompt_embeds, uncond_prompt_attention_mask = get_null_embed(
            cfg.training.null_emb_path, accelerator.device
        )

    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=cfg.training.num_train_timesteps,
        shift=cfg.training.fm_shift,
    )

    # build model
    model = build_model(cfg)

    if cfg.training.transformer_ckpt != "":
        state_dict = load_file(cfg.training.transformer_ckpt)
        model.load_state_dict(state_dict, strict=False)

    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # enable deferred patch masking
    if cfg.training.patch_mixer_depth != -1:
        model.enable_masking(cfg.training.patch_mixer_depth, cfg.training.mask_ratio)

    if cfg.training.use_ema:
        model_ema = deepcopy(model).eval()

    # mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if cfg.training.grad_checkpointing:
        model.enable_gradient_checkpointing(cfg.training.nblocks_to_apply_grad_checkpointing)

    if not cfg.training.use_precomputed_text_embeddings:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_text_encoder = AutoModelForCausalLM.from_pretrained(cfg.model.text_encoder_name, torch_dtype=weight_dtype)
        model_text_encoder.requires_grad_(False)
        model_text_encoder = model_text_encoder.to(accelerator.device)
        model_text_encoder = torch.compile(model_text_encoder)

        # get null embedding
        inputs = tokenizer(
            "",
            return_tensors="pt",
            padding="max_length",
            max_length=cfg.model.caption_max_seq_length,
            truncation=True,
        )
        inputs.to(accelerator.device)
        uncond_prompt_embeds = model_text_encoder(**inputs, output_hidden_states=True)["hidden_states"][-1]
        uncond_prompt_attention_mask = inputs["attention_mask"]

    # build dataloader
    dataloader_kwargs = {
        "batch_size": cfg.training.train_batch_size,
        "shuffle": True,
        "num_workers": cfg.training.num_workers,
        "pin_memory": True,
    }
    if cfg.dataset.use_dummy_data:
        from core.dataset import DummyDataset

        dataset = DummyDataset(
            in_channels=cfg.model.in_channels,
            latent_size=cfg.model.latent_size,
            caption_max_seq_length=cfg.model.caption_max_seq_length,
            caption_channels=cfg.model.caption_channels,
        )
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)
    else:
        from core.dataset import build_streaming_latents_dataloader

        train_dataloader = build_streaming_latents_dataloader(
            cfg.dataset,
            latent_size=cfg.model.latent_size,
            caption_max_seq_length=cfg.model.caption_max_seq_length,
            caption_channels=cfg.model.caption_channels,
            load_precomputed_text_embeddings=cfg.training.use_precomputed_text_embeddings,
            **dataloader_kwargs,
        )

    # build optimizer and lr scheduler
    optimizer_class, optimizer_kwargs = build_optimizer(cfg.optimizer.type, cfg.optimizer.optimizer_kwargs)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    logger.info("optimizer info")
    logger.info(optimizer)

    # set lr scheduler
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.num_warmup_steps,
    )

    # Prepare everything
    if cfg.training.use_ema:
        model, model_ema, optimizer, lr_scheduler = accelerator.prepare(model, model_ema, optimizer, lr_scheduler)
    else:
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    global_step = 0

    if cfg.training.resume_from != "":
        accelerator.load_state(cfg.training.resume_from)
        logger.info(f"Resume training from {cfg.training.resume_from}")
        global_step = int(cfg.training.resume_from.split("-")[-1]) + 1
        logger.info(f"Resuming from global step {global_step}")

    progress_bar = tqdm(
        total=cfg.training.max_iters,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )

    # start training
    while True:
        for _, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            move_to_device(batch, accelerator.device)

            if cfg.training.use_precomputed_text_embeddings:
                y = batch["caption_latents"].to(weight_dtype)
                y_mask = batch["caption_attention_mask"].to(weight_dtype)
            else:
                with torch.no_grad():
                    tokenized_captions = tokenizer(
                        batch["captions"],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=cfg.model.caption_max_seq_length,
                        truncation=True,
                    )
                    tokenized_captions = tokenized_captions.to(device=accelerator.device)
                    y = model_text_encoder(**tokenized_captions, output_hidden_states=True)["hidden_states"][-1]
                    y_mask = tokenized_captions["attention_mask"]
                    y_mask = y_mask.to(device=accelerator.device, dtype=weight_dtype)

            zs = [batch["visual_encoder_features"]]  # make it list to support multiple features
            latents = (batch["image_latents"] * cfg.training.scaling_factor).to(weight_dtype)

            y, y_mask = token_drop(
                y,
                y_mask,
                uncond_prompt_embeds,
                uncond_prompt_attention_mask,
                cfg.training.class_dropout_prob,
            )

            bs = latents.shape[0]
            noise = torch.randn_like(latents)

            # timestep sampling and mix latent with noise
            u = compute_density_for_timestep_sampling(
                weighting_scheme=cfg.training.weighting_scheme,
                batch_size=bs,
                logit_mean=cfg.training.logit_mean,
                logit_std=cfg.training.logit_std,
                mode_scale=cfg.training.mode_scale,
            )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices]
            sigmas = get_sigmas(timesteps, noise_scheduler, n_dim=latents.ndim).to(device=accelerator.device)
            timesteps = timesteps.to(device=accelerator.device)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            grad_norm = None
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                model_pred, ids_keep, zs_tilde = model(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=y,
                    timestep=timesteps,
                    encoder_attention_mask=y_mask,
                    return_dict=False,
                )
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=cfg.training.weighting_scheme, sigmas=sigmas
                )
                target = noise - latents

                model_pred = einops.rearrange(
                    model_pred,
                    "n t (p1 p2 c) -> n c t (p1 p2)",
                    p1=cfg.model.patch_size,
                    p2=cfg.model.patch_size,
                )

                target = patchify_and_apply_mask(target, cfg.model.patch_size, ids_keep)

                diff_loss = calculate_diffusion_loss(model_pred, target, weighting)

                # REPA loss
                if zs_tilde is not None:
                    repa_loss = calculate_repa_loss(zs, zs_tilde, ids_keep)
                    loss = diff_loss + repa_loss * cfg.training.repa_coeff
                else:
                    loss = diff_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                    if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                        optimizer.zero_grad(set_to_none=True)
                        logger.warning("NaN or Inf detected in grad_norm, skipping iteration...")
                        continue

                optimizer.step()
                lr_scheduler.step()

                if cfg.training.use_ema:
                    if accelerator.sync_gradients:
                        ema_update(
                            accelerator.unwrap_model(model_ema),
                            accelerator.unwrap_model(model),
                            cfg.training.ema_rate,
                        )

            logs = {
                "loss": accelerator.gather(diff_loss).detach().mean().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if zs_tilde is not None:
                logs["repa_loss"] = accelerator.gather(repa_loss).detach().mean().item()
            if grad_norm is not None:
                if isinstance(grad_norm, torch.distributed.tensor.DTensor):
                    local_norm = grad_norm.to_local()
                else:
                    local_norm = grad_norm
                logs["grad_norm"] = accelerator.gather(local_norm).detach().mean().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                accelerator.wait_for_everyone()

                # Save checkpoint
                if global_step % cfg.training.save_freq == 0:
                    save_path = os.path.join(
                        os.path.join(project_dir, "checkpoints"),
                        f"checkpoint-{global_step}",
                    )
                    logger.info(f"Start to save state to {save_path}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                # Generate validation images
                if global_step and global_step % cfg.validation.validation_frequency == 0:
                    model_to_validate = model_ema if cfg.training.use_ema else model
                    model_state_dict = accelerator.get_state_dict(accelerator.unwrap_model(model_to_validate, keep_torch_compile=False))
                    if accelerator.is_main_process:
                        val_dir = os.path.join(project_dir, "validation_images")
                        log_validation(model_state_dict, accelerator, cfg, step=global_step, save_dir=val_dir)
                    torch.cuda.empty_cache()

                accelerator.wait_for_everyone()

            if global_step >= cfg.training.max_iters:
                break

        if global_step >= cfg.training.max_iters:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(os.path.join(project_dir, "checkpoints"), f"checkpoint-{global_step}")
        logger.info(f"Start to save state to {save_path}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
