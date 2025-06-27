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


from typing import Any, Dict, Optional

import torch
from diffusers import PixArtTransformer2DModel as PixArtTransformer2DModelOriginal
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils import is_torch_version, logging
from torch import nn

from core.models.rms_norm import RMSNorm
from core.utils import random_masking

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Modified from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/pixart_transformer_2d.py
class NitroDiTModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "PatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = 8,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        caption_channels: Optional[int] = None,
        attention_type: Optional[str] = "default",
        repa_depth=-1,
        projector_dim=2048,
        z_dims=[768],
    ):
        super().__init__()
        # copy over some functions to avoid duplicating the code
        _tmp_original_model = PixArtTransformer2DModelOriginal()
        self.fuse_qkv_projections = _tmp_original_model.fuse_qkv_projections
        self.unfuse_qkv_projections = _tmp_original_model.unfuse_qkv_projections
        self.set_default_attn_processor = _tmp_original_model.set_default_attn_processor
        self.set_attn_processor = _tmp_original_model.set_attn_processor
        self.attn_processors = _tmp_original_model.attn_processors
        del _tmp_original_model

        self.patch_mixer_depth = None  # initially no masking applied
        self.mask_ratio = 0

        if repa_depth != -1:
            from core.models.projector import build_projector
            self.projectors = nn.ModuleList(
                [build_projector(cross_attention_dim, projector_dim, z_dim) for z_dim in z_dims]
            )
            assert repa_depth >= 0 and repa_depth < num_layers
            self.repa_depth = repa_depth

        # Validate inputs.
        if norm_type != "ada_norm_single":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_single" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels

        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        interpolation_scale = (
            self.config.interpolation_scale
            if self.config.interpolation_scale is not None
            else max(self.config.sample_size // 64, 1)
        )

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.out_channels,
        )

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, use_additional_conditions=False)
        self.caption_projection = None
        if self.config.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.config.caption_channels, hidden_size=self.inner_dim
            )

        self.text_embedding_norm = RMSNorm(
            self.inner_dim if self.caption_projection else self.config.caption_channels,
            scale_factor=0.01,
            eps=1e-5,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.text_embedding_norm(encoder_hidden_states)

        ids_keep = None
        len_keep = hidden_states.shape[1]
        zs = None
        # 2. Blocks
        for blk_ind, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing and block.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    None,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )

            # patch masking
            if self.training and (self.patch_mixer_depth != -1) and (self.patch_mixer_depth == blk_ind):
                hidden_states, ids_keep, len_keep = random_masking(hidden_states, self.mask_ratio)

            # REPA
            if self.training and (self.repa_depth != -1) and (self.repa_depth == blk_ind):
                N, T, D = hidden_states.shape
                zs = [projector(hidden_states.reshape(-1, D)).reshape(N, len_keep, -1) for projector in self.projectors]

        # 3. Output
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # if inference, return the unpatchified output as usual
        # if training, return the patch sequence 
        if not self.training:
            hidden_states = hidden_states.reshape(
                shape=(
                    -1,
                    height,
                    width,
                    self.config.patch_size,
                    self.config.patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    self.out_channels,
                    height * self.config.patch_size,
                    width * self.config.patch_size,
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
