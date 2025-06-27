# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]


from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate

try:
    from flash_attn_interface import flash_attn_func
except ModuleNotFoundError:
    from flash_attn import flash_attn_func

from einops import rearrange
from flash_attn.bert_padding import unpad_input
from flash_attn.modules.mha import FlashCrossAttention, FlashSelfAttention

unpad_input = torch._dynamo.disable(unpad_input)

def fa_sdpa(q, k, v, dropout_p=0.0, is_causal=False):
    """Flash-Attention drop-in replacement of torch.nn.functional.scaled_dot_product_attention function"""
    q, k, v = [x.permute(0, 2, 1, 3).contiguous() for x in [q, k, v]]
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
    return out.permute(0, 2, 1, 3)


# Modified from https://github.com/huggingface/diffusers/blob/393aefcdc7c7e786d7b2adf95750cf72fbfbed89/src/diffusers/models/attention_processor.py
# to use flash-attn instead of torch SDPA
class JointAttnProcessor2_0_FA:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        dtype_before_norm = query.dtype
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(dtype_before_norm)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype_before_norm)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            dtype_before_norm = encoder_hidden_states_query_proj.dtype
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj).to(
                    dtype_before_norm
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj).to(dtype_before_norm)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        orig_datatype = query.dtype
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        hidden_states = fa_sdpa(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(orig_datatype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


# Modified from https://github.com/huggingface/diffusers/blob/393aefcdc7c7e786d7b2adf95750cf72fbfbed89/src/diffusers/models/attention_processor.py
# to use flash-attn instead of torch SDPA
class AttnProcessor2_0_FA:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        # these are lightweight python objects so no cost in
        # initializing them always
        self.inner_cross_attn = FlashCrossAttention()
        self.inner_self_attn = FlashSelfAttention()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, "norm_q") and attn.norm_k is not None:
            key = attn.norm_k(key)

        orig_datatype = query.dtype
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        if attn.is_cross_attention:
            kv = torch.stack([key, value], dim=2)  # [b, s, 2, h, d]
            attention_mask = attention_mask[:, 0, 0, :]
            attention_mask = torch.where(attention_mask < 0.0, 0, 1).to(dtype=torch.int32)
            query, _, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
                query,
                torch.ones((batch_size, query.shape[1]), device=attention_mask.device),
            )
            kv, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(kv, attention_mask)
            hidden_states = self.inner_cross_attn(
                query,
                kv,
                cu_seqlens=cu_seqlens_q,
                max_seqlen=max_seqlen_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
            )
            hidden_states = rearrange(hidden_states, "(b s) ... -> b s ...", b=batch_size)

        else:
            qkv = torch.stack([query, key, value], dim=2)
            hidden_states = self.inner_self_attn(qkv)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(orig_datatype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
