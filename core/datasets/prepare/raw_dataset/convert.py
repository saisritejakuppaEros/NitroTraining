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

"""Convert raw_data (split_XX/00000.tar img2dataset format) to MDS for Nitro-T training.

Example usage:
    python convert.py --raw_dir /data/corerndimage/DiffusionModels/datapruning/raw_data \
        --local_mds_dir ./raw_dataset/mds/ --max_image_size 512 --num_proc 16
"""

import glob
import os
import shutil
import tarfile
from argparse import ArgumentParser
from multiprocessing import Pool, current_process
from typing import Generator, List, Tuple

import numpy as np
from PIL import Image
from streaming.base import MDSWriter
from streaming.base.util import merge_index
from torchvision import transforms
from tqdm import tqdm


def parse_arguments():
    parser = ArgumentParser(
        description="Convert raw_data (split_XX/00000.tar) to MDS format for Nitro-T."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="/data/corerndimage/DiffusionModels/datapruning/raw_data",
        help="Path to raw_data with split_XX/00000.tar structure (img2dataset format)",
    )
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        default="",
        help="Directory to store mds shards.",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=512,
        help="Resize all images so the shorter edge is this size (upscale small, downscale large). Only broken images are skipped.",
    )
    parser.add_argument("--num_proc", type=int, default=16)
    return parser.parse_args()


def current_process_index() -> int:
    # by default it starts from 1
    p = current_process()
    return p._identity[0] - 1


def read_tar(tar_path: str, path_out: str) -> Generator[Tuple[Image.Image, str, str, str], None, None]:
    os.makedirs(path_out, exist_ok=False)
    with tarfile.open(tar_path, "r") as tar:
        try:
            tar.extractall(path_out, filter="data")
        except TypeError:
            tar.extractall(path_out)

    txts = sorted(glob.glob(os.path.join(path_out, "*txt")))

    for t in txts:
        try:
            with open(t, "r") as ct:
                cap = ct.read()
            # assuming all files are in jpg (img2dataset format)
            img_path = t.replace(".txt", ".jpg")
            img = Image.open(img_path)
            image_name = os.path.basename(img_path)
            image_path = f"{tar_path}::{image_name}"
            yield img, cap, image_name, image_path
        except Exception:
            pass
    shutil.rmtree(path_out)
    if os.path.exists(os.path.dirname(path_out)):
        shutil.rmtree(os.path.dirname(path_out))


def write_tar(tars: List[str], args: ArgumentParser):
    columns = {
        "width": "int32",
        "height": "int32",
        "jpg": "jpeg",
        "caption": "str",
        "image_name": "str",
        "image_path": "str",
    }

    # make sure that write_tar is only called once per process
    save_dir = os.path.join(args.local_mds_dir, str(current_process_index()))
    os.makedirs(save_dir, exist_ok=True)

    # create a writer per process
    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
        exist_ok=True,  # Allow restarting if directory exists
    )

    resize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC,
    )

    temp_dir = os.path.join(save_dir, f"temp/wds_{current_process_index()}")

    proc_idx = current_process_index()
    for tar_idx, tar in enumerate(tars):
        skipped, total = 0, 0
        pbar = tqdm(
            read_tar(tar, temp_dir),
            desc=f"W{proc_idx}",
            leave=False,
            mininterval=2,
            dynamic_ncols=True,
        )
        for img, cap, image_name, image_path in pbar:
            try:
                w, h = img.size
                img = resize(img)
                w, h = img.size
                mds_sample = {
                    "jpg": img,
                    "caption": cap,
                    "width": w,
                    "height": h,
                    "image_name": image_name,
                    "image_path": image_path,
                }
                writer.write(mds_sample)
                total += 1
            except Exception:
                skipped += 1

        pbar.write(f"Tar {tar_idx + 1}/{len(tars)}: skipped={skipped} (errors), written={total}")
    writer.finish()


def main():
    args = parse_arguments()
    tars = glob.glob(os.path.join(args.raw_dir, "**/*.tar"), recursive=True)
    print(f"Total {len(tars)} tar files found in raw_data path: {args.raw_dir}")
    if args.num_proc > 1:
        print(f"Using {args.num_proc} workers")

    if not tars:
        raise ValueError(f"No .tar files found in {args.raw_dir}")

    tars_split = np.array_split(tars, args.num_proc)

    with Pool(processes=args.num_proc) as pool:
        pool.starmap(write_tar, [(ts, args) for ts in tars_split])

    shards_metadata = [
        os.path.join(args.local_mds_dir, str(i), "index.json")
        for i in range(args.num_proc)
    ]
    merge_index(shards_metadata, out=args.local_mds_dir, keep_local=True)


if __name__ == "__main__":
    main()
