# AMD Nitro-T

<div>
  <!-- <a href=""><img src="https://img.shields.io/badge/AMD%20Developer%20Blog-red"></a> &ensp; -->
  <a href="https://huggingface.co/amd/Nitro-T-0.6B"><img src="https://img.shields.io/badge/Nitro%20T%200.6B-HuggingFace-yellow"></a> &ensp;
  <a href="https://huggingface.co/amd/Nitro-T-1.2B"><img src="https://img.shields.io/badge/Nitro%20T%201.2B-HuggingFace-yellow"></a> &ensp;
  <!-- <a href="https://github.com/AMD-AIG-AIMA/AMD-Diffusion-Distillation/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/AMD-AIG-AIMA/AMD-Diffusion-Distillation.svg?color=blue"></a> -->
</div>

---
![image_row_4x1](https://github.com/user-attachments/assets/5a4d580d-c8e8-4e1e-9ff1-f8ee880d7e87)


Nitro-T is a set of text-to-image diffusion models focused on highly efficient training. Our models achieve competitive scores on image generation benchmarks compared to previous models focused on efficient training while requiring less than 1 day of training from scratch on 32 AMD Instinct<sup>TM</sup> MI300X GPUs. 

This repository provides training and data preparation scripts to reproduce our results. We hope this codebase for efficient diffusion model training enables researchers to iterate faster on ideas and lowers the barrier for independent developers to build custom models.

The models can be found on HuggingFace:
* [Nitro-T-0.6B](https://huggingface.co/amd/Nitro-T-0.6B), a 512px DiT-based model
* [Nitro-T-1.2B](https://huggingface.co/amd/Nitro-T-1.2B), a 1024px MMDiT-based model


## Environment

The codebase in implemented using PyTorch. Follow the [official instructions](https://pytorch.org/get-started/locally/) to install it in your compute environment.

### Docker image
When running on AMD Instinct<sup>TM</sup> GPUs, it is recommended to use the [public PyTorch ROCm images](https://hub.docker.com/r/rocm/pytorch-training/) to get optimized performance out-of-the-box.

```bash
docker pull rocm/pytorch-training
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Preparing the training dataset

The Nitro-T models were trained on a dataset of ~35M images consisting of both real and synthetic data sources that are openly available on the internet. Use the scripts in `core/datasets/scripts` to download and pre-process the dataset. The scripts are based on the excellent [MicroDiT](https://github.com/SonyResearch/micro_diffusion/tree/main/micro_diffusion/datasets) repo and modified for our use case. 


## Training the models
Launch a training run using this script:
```
bash scripts/run_train.sh
```

Use the config files to control the training process
* `configs/accelerate.yaml`: Set the multi-GPU / multi-node distributed training setup, torch compile, etc.
* `configs/default_config.yaml`: Set the training hyperparameters, dataset paths.
* Experiment-specific configs override the values in `default_config.yaml` 


## Minimal inference example

You must use `diffusers>=0.34` in order to load the model from the Huggingface hub ([Issue](https://github.com/huggingface/diffusers/pull/11652))

```python
import torch
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

device = torch.device('cuda:0')
dtype = torch.bfloat16
resolution = 512
MODEL_NAME = "amd/Nitro-T-0.6B"

text_encoder = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=dtype)
pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    text_encoder=text_encoder,
    torch_dtype=dtype, 
    trust_remote_code=True,
)
pipe.to(device)

image = pipe(
    prompt="The image is a close-up portrait of a scientist in a modern laboratory. He has short, neatly styled black hair and wears thin, stylish eyeglasses. The lighting is soft and warm, highlighting his facial features against a backdrop of lab equipment and glowing screens.",
    height=resolution, width=resolution,
    num_inference_steps=20,
    guidance_scale=4.0,
).images[0]

image.save("output.png")
```


## Acknowledgements
We would like to thank [MicroDiT](https://github.com/SonyResearch/micro_diffusion) for sparking the idea for this project and providing easy dataset processing scripts, and Diffusers for providing modular building blocks for diffusion models.



## License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

This project is licensed under the [MIT License](https://mit-license.org/).