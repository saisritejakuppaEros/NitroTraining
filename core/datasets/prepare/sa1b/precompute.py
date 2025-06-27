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

import os
import sys
import time
from argparse import ArgumentParser
from datetime import timedelta

import numpy as np
import timm
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from diffusers import AutoencoderDC
from streaming import MDSWriter
from streaming.base.util import merge_index
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from prepare.sa1b.base import build_streaming_sa1b_precompute_dataloader
from utils import DATA_TYPES, UniversalTextEncoder

"""Example usage:
accelerate launch --multi_gpu --num_processes 8 precompute.py \
    --datadir ./sa1b/mds/ \
    --savedir ./sa1b/mds_latents_sdxl1_dfnclipH14/ \
    --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --batch_size 32
"""


def parse_args() -> ArgumentParser:
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="Local directory to store mds shards.",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="",
        help="Remote path to upload MDS-formatted shards to.",
    )
    parser.add_argument(
        "--image_resolutions",
        type=int,
        nargs="+",
        default=[512],
        help="List of image resolutions to use for processing.",
    )
    parser.add_argument(
        "--save_images",
        default=False,
        action="store_true",
        help="If True, also save images, else only latents",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Data type for the encoding models",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        choices=("float16", "float32"),
        default="float16",
        help="Data type to save the latents",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Name of VAE model to use for vision encoding.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
        help="Name of model to use for text encoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device to use for encoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Seed for random number generation.",
    )
    args = parser.parse_args()
    if isinstance(args.image_resolutions, int):
        args.image_resolutions = [args.image_resolutions]
    return args


def main(args: ArgumentParser) -> None:
    """Precompute image and text latents and store them in MDS format.

    By default, we only save the image latents for 512x512 image
    resolutions (using center crop).

    Note that the image latents will be scaled by the vae_scaling_factor.
    """
    cap_key = "caption"
    syn_cap_key = "caption_syn_pixart_llava15"

    init_kwargs_to_increase_timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=20000))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs_to_increase_timeout])
    device = accelerator.device
    device_idx = int(accelerator.process_index)

    # Set random seeds
    torch.manual_seed(device_idx + args.seed)
    torch.cuda.manual_seed(device_idx + args.seed)
    np.random.seed(device_idx + args.seed)

    dataloader = build_streaming_sa1b_precompute_dataloader(
        datadir=[args.datadir],
        batch_size=args.batch_size,
        resize_sizes=args.image_resolutions,
        drop_last=False,
        shuffle=False,
        caption_key=syn_cap_key,  # Using llava captions in MDS dataset
        tokenizer_name=args.text_encoder,
        prefetch_factor=2,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    print(f"Device: {device_idx}, Dataloader sample count: {len(dataloader.dataset)}")
    print(f"MP variable -> world size: {os.environ['WORLD_SIZE']}, " f"RANK: {os.environ['RANK']}, {device}")

    vae = AutoencoderDC.from_pretrained(
        args.vae,
        torch_dtype=torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32,
    ).eval()
    print("Created VAE: ", args.vae)

    text_encoder = UniversalTextEncoder(
        args.text_encoder,
        dtype=args.model_dtype,
        pretrained=True,
    )
    print("Created text encoder:", args.text_encoder)

    visual_encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_vitb14")
    del visual_encoder.head
    visual_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        visual_encoder.pos_embed.data,
        [16, 16],
    )
    visual_encoder.head = torch.nn.Identity()
    visual_encoder.eval()
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    inverse_mean, inverse_std = (-mean / std).tolist(), (1.0 / std).tolist()
    transform_for_visual_encoder = transforms.Compose(
        [
            transforms.Normalize(inverse_mean, inverse_std),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        ]
    )
    print("Created DINOv2 visual encoder")

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    visual_encoder = visual_encoder.to(device)

    columns = {
        cap_key: "str",
        f"{cap_key}_latents": "bytes",
        f"{cap_key}_attention_mask": "bytes",
        "latents": "bytes",
        "visual_encoder_features": "bytes",
    }
    if args.save_images:
        columns["jpg"] = "jpeg"

    remote_upload = os.path.join(args.savedir, str(accelerator.process_index))
    writer = MDSWriter(
        out=remote_upload,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for batch in tqdm(dataloader):
        image = torch.stack(batch["image_0"]).to(device)
        captions = torch.stack(batch[syn_cap_key]).to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=DATA_TYPES[args.model_dtype]):
                latents = vae.encode(image).latent.to(DATA_TYPES[args.save_dtype])

                attention_mask = None
                if f"{syn_cap_key}_attention_mask" in batch:
                    attention_mask = torch.cat(batch[f"{syn_cap_key}_attention_mask"]).to(device)

                conditioning = text_encoder.encode(
                    captions.view(-1, captions.shape[-1]),
                    attention_mask=attention_mask,
                )[0].to(DATA_TYPES[args.save_dtype])

                # undo the normalization and re-normalize for the visual encoder
                visual_encoder_features = visual_encoder.forward_features(transform_for_visual_encoder(image))
                visual_encoder_features = visual_encoder_features["x_norm_patchtokens"].to(DATA_TYPES[args.save_dtype])

        try:
            latents = latents.detach().cpu().numpy()
            conditioning = conditioning.detach().cpu().numpy()
            if attention_mask is not None:
                attention_mask = attention_mask.detach().cpu().numpy()
            visual_encoder_features = visual_encoder_features.detach().cpu().numpy()

            # Write the batch to the MDS file
            for i in range(latents.shape[0]):
                mds_sample = {
                    cap_key: batch["sample"][i][syn_cap_key],
                    f"{cap_key}_latents": np.reshape(conditioning[i], -1).tobytes(),
                    f"{cap_key}_attention_mask": np.reshape(attention_mask[i], -1).tobytes(),
                    "latents": latents[i].tobytes(),
                    "visual_encoder_features": visual_encoder_features[i].tobytes(),
                }
                if args.save_images:
                    mds_sample["jpg"] = batch["sample"][i]["jpg"]
                writer.write(mds_sample)
        except RuntimeError:
            print("Runtime error CUDA, skipping this batch")

    writer.finish()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    print(f"Process {accelerator.process_index} finished")
    time.sleep(10)

    # Merge the mds shards created by each device (only do on main process)
    if accelerator.is_main_process:
        shards_metadata = [os.path.join(args.savedir, str(i), "index.json") for i in range(accelerator.num_processes)]
        merge_index(shards_metadata, out=args.savedir, keep_local=True)


if __name__ == "__main__":
    main(parse_args())
