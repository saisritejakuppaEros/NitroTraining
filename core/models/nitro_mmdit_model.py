# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.embeddings import PatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.utils import is_torch_version, logging

from core.models.rms_norm import RMSNorm
from core.utils import random_masking

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=dtype))  # (N, D)

        return timesteps_emb


class NitroMMDiTModel(SD3Transformer2DModel):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        caption_channels: int = 4096,
        caption_projection_dim: int = 1152,
        out_channels: int = 16,
        interpolation_scale: int = 1,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        repa_depth=-1,
        projector_dim=2048,
        z_dims=[768],
    ):
        super().__init__(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_projection_dim=caption_projection_dim,
            out_channels=out_channels,
            pos_embed_max_size=pos_embed_max_size,
            dual_attention_layers=dual_attention_layers,
            qk_norm=qk_norm,
        )

        self.patch_mixer_depth = None  # initially no masking applied
        self.mask_ratio = 0

        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        if repa_depth != -1:
            from core.models.projector import build_projector
            self.projectors = nn.ModuleList([build_projector(self.inner_dim, projector_dim, z_dim) for z_dim in z_dims])
            assert repa_depth >= 0 and repa_depth < num_layers
            self.repa_depth = repa_depth

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=self.config.interpolation_scale,
        )
        self.time_text_embed = TimestepEmbeddings(embedding_dim=self.inner_dim)
        self.context_embedder = nn.Linear(self.config.caption_channels, self.config.caption_projection_dim)
        self.text_embedding_norm = RMSNorm(self.inner_dim, scale_factor=0.01, eps=1e-5)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, dtype=encoder_hidden_states.dtype)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = self.text_embedding_norm(encoder_hidden_states)

        ids_keep = None
        len_keep = hidden_states.shape[1]
        zs = None
        for index_block, block in enumerate(self.transformer_blocks):

            if torch.is_grad_enabled() and self.gradient_checkpointing and block.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

            # patch masking
            if self.training and (self.patch_mixer_depth != -1) and (self.patch_mixer_depth == index_block):
                hidden_states, ids_keep, len_keep = random_masking(hidden_states, self.mask_ratio)

            # REPA
            if self.training and (self.repa_depth != -1) and (self.repa_depth == index_block):
                N, T, D = hidden_states.shape
                zs = [projector(hidden_states.reshape(-1, D)).reshape(N, len_keep, -1) for projector in self.projectors]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)


        # if inference, return the unpatchified output as usual
        # if training, return the patch sequence 
        if not self.training:
            patch_size = self.config.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    height,
                    width,
                    patch_size,
                    patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    self.out_channels,
                    height * patch_size,
                    width * patch_size,
                )
            )

            if not return_dict:
                return (output,)

            return Transformer2DModelOutput(sample=output)

        else:
            return hidden_states, ids_keep, zs

    def enable_masking(self, depth, mask_ratio):
        # depth: apply masking after block_[depth]. should be [0, nblks-1]
        assert depth >= 0 and depth < len(self.transformer_blocks)
        self.patch_mixer_depth = depth
        assert mask_ratio >= 0 and mask_ratio <= 1
        self.mask_ratio = mask_ratio

    def disable_masking(self):
        self.patch_mixer_depth = None

    def enable_gradient_checkpointing(self, nblocks_to_apply_grad_checkpointing):
        N = len(self.transformer_blocks)

        if nblocks_to_apply_grad_checkpointing == -1:
            nblocks_to_apply_grad_checkpointing = N
        nblocks_to_apply_grad_checkpointing = min(N, nblocks_to_apply_grad_checkpointing)

        # Apply to blocks evenly spaced out
        step = N / nblocks_to_apply_grad_checkpointing if nblocks_to_apply_grad_checkpointing > 0 else 0
        indices = [int((i + 0.5) * step) for i in range(nblocks_to_apply_grad_checkpointing)]

        self.gradient_checkpointing = True
        for blk_ind, block in enumerate(self.transformer_blocks):
            block.gradient_checkpointing = blk_ind in indices
            print(f"Block {blk_ind} grad checkpointing set to {block.gradient_checkpointing}")
