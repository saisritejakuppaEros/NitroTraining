# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model_text_encoder = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16)
model_text_encoder.requires_grad_(False)
model_text_encoder = model_text_encoder.to("cuda:0")

inputs = tokenizer(
    "",
    return_tensors="pt",
    padding="max_length",
    max_length=256,
    truncation=True,
)
inputs.to("cuda:0")
uncond_prompt_embeds = model_text_encoder(**inputs, output_hidden_states=True)["hidden_states"][-1]
uncond_prompt_attention_mask = inputs["attention_mask"]

torch.save(
    {
        "uncond_prompt_embeds": uncond_prompt_embeds,
        "uncond_prompt_attention_mask": uncond_prompt_attention_mask,
    },
    "null_emb.pth",
)
