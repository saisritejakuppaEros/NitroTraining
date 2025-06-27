# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]


import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader, Dataset


# Modified from: https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/datasets/latents_loader.py
class StreamingLatentsDataset(StreamingDataset):
    """Dataset class for loading precomputed latents from mds format.

    Args:
        streams: List of individual streams (in our case streams of individual datasets)
        shuffle: Whether to shuffle the dataset
        latent_size: Size of latents
        caption_max_seq_length: Context length of text-encoder
        caption_channels: Dimension of caption embeddings
        cap_drop_prob: Probability of using all zeros caption embedding (classifier-free guidance)
        batch_size: Batch size for streaming
    """

    def __init__(
        self,
        streams: Optional[List[Stream]] = None,
        shuffle: bool = False,
        latent_size: Optional[int] = None,
        caption_max_seq_length: Optional[int] = None,
        caption_channels: Optional[int] = None,
        load_precomputed_text_embeddings: bool = True,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
            cache_limit="1tb",
            shuffle_algo="py1s",
        )

        self.latent_size = latent_size
        self.caption_max_seq_length = caption_max_seq_length
        self.caption_channels = caption_channels
        self.load_precomputed_text_embeddings = load_precomputed_text_embeddings

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        sample = super().__getitem__(index)

        processed_sample = {}

        processed_sample["captions"] = sample["caption"]

        if self.load_precomputed_text_embeddings:
            processed_sample["caption_latents"] = (
                torch.from_numpy(np.frombuffer(sample["caption_latents"], dtype=np.float16).copy())
                .reshape(self.caption_max_seq_length, self.caption_channels)
                .float()
            )

            processed_sample["caption_attention_mask"] = torch.from_numpy(
                np.frombuffer(sample["caption_attention_mask"], dtype=np.int64).copy()
            ).reshape(self.caption_max_seq_length)

        processed_sample["image_latents"] = (
            torch.from_numpy(np.frombuffer(sample["latents"], dtype=np.float16).copy())
            .reshape(-1, self.latent_size, self.latent_size)
            .float()
        )

        processed_sample["visual_encoder_features"] = (
            torch.from_numpy(np.frombuffer(sample["visual_encoder_features"], dtype=np.float16).copy())
            .reshape(256, 768)
            .float()
        )

        return processed_sample


# Modified from: https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/datasets/latents_loader.py
def build_streaming_latents_dataloader(
    dataset_config,
    batch_size: int,
    latent_size: int = 16,
    caption_max_seq_length: int = 120,
    caption_channels: int = 1024,
    load_precomputed_text_embeddings: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """Creates a DataLoader for streaming latents dataset."""

    streams = [
        Stream(
            remote=os.path.join(dataset_config.root, d),
            local=os.path.join(dataset_config.local, d),
        )
        for d in dataset_config.datasets
    ]

    dataset = StreamingLatentsDataset(
        streams=streams,
        shuffle=shuffle,
        latent_size=latent_size,
        caption_max_seq_length=caption_max_seq_length,
        caption_channels=caption_channels,
        load_precomputed_text_embeddings=load_precomputed_text_embeddings,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        persistent_workers=True,
        **dataloader_kwargs,
    )

    return dataloader


class DummyDataset(Dataset):
    def __init__(self, in_channels, latent_size, caption_max_seq_length, caption_channels):
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.caption_max_seq_length = caption_max_seq_length
        self.caption_channels = caption_channels

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        processed_sample = {}
        processed_sample["captions"] = "Dummy caption"
        processed_sample["image_latents"] = torch.randn((self.in_channels, self.latent_size, self.latent_size))
        processed_sample["caption_latents"] = torch.randn((self.caption_max_seq_length, self.caption_channels))
        processed_sample["caption_attention_mask"] = torch.ones((self.caption_max_seq_length)).long()
        processed_sample["visual_encoder_features"] = torch.randn((256, 768))
        return processed_sample
