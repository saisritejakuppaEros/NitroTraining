#!/usr/bin/env python3
"""Verify that image_paths stored in MDS files point to existing files.

For img2dataset format: image_path is "tar_path::image_name" (image inside tar).
Checks: 1) tar file exists, 2) tar contains the image member.
"""

import argparse
import os
import tarfile
from collections import defaultdict

from streaming import StreamingDataset


def parse_image_path(image_path: str):
    """Parse image_path. Format: 'tar_path::image_name' or plain path."""
    if "::" in image_path:
        tar_path, image_name = image_path.rsplit("::", 1)
        return tar_path, image_name
    return image_path, None


def check_tar_member_exists(tar_path: str, member_name: str) -> bool:
    """Check if tar file exists and contains the member."""
    if not os.path.isfile(tar_path):
        return False
    try:
        with tarfile.open(tar_path, "r") as tar:
            try:
                tar.getmember(member_name)
                return True
            except KeyError:
                return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify image paths in MDS exist")
    parser.add_argument(
        "mds_dir",
        type=str,
        help="Path to MDS directory (e.g. amd_latents/split_001)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max samples to check (0 = all)",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Fraction of samples to check (0-1)",
    )
    args = parser.parse_args()

    dataset = StreamingDataset(
        remote=None,
        local=args.mds_dir,
        shuffle=False,
    )

    total = len(dataset)
    max_check = args.max_samples if args.max_samples > 0 else total
    step = max(1, int(1.0 / args.sample_rate)) if args.sample_rate < 1.0 else 1

    print(f"MDS dir: {args.mds_dir}")
    print(f"Total samples: {total}")
    print(f"Checking up to {max_check} samples (step={step})", flush=True)

    missing_tar = []
    missing_member = []
    ok_count = 0
    checked = 0
    tar_cache = {}  # tar_path -> set of members (lazy load)

    for i in range(0, min(total, max_check), step):
        sample = dataset[i]
        image_path = sample.get("image_path", "")
        if not image_path:
            continue

        tar_path, member_name = parse_image_path(image_path)
        checked += 1

        if member_name is None:
            # Plain file path
            if os.path.isfile(tar_path):
                ok_count += 1
            else:
                missing_tar.append(image_path)
            continue

        # Tar format
        if not os.path.isfile(tar_path):
            missing_tar.append(image_path)
            continue

        if check_tar_member_exists(tar_path, member_name):
            ok_count += 1
        else:
            missing_member.append((tar_path, member_name))

    print(f"\nChecked {checked} samples")
    print(f"OK (path exists): {ok_count}")
    print(f"Missing tar file: {len(missing_tar)}")
    print(f"Tar exists but member missing: {len(missing_member)}")

    if missing_tar:
        print("\n--- Sample missing tar paths (first 10) ---")
        for p in missing_tar[:10]:
            print(f"  {p}")

    if missing_member:
        print("\n--- Sample missing members (first 10) ---")
        for tar, mem in missing_member[:10]:
            print(f"  {tar} :: {mem}")

    if not missing_tar and not missing_member:
        print("\nAll checked image paths exist.")
    else:
        print(f"\nSummary: {ok_count}/{checked} paths exist, {len(missing_tar) + len(missing_member)} missing")


if __name__ == "__main__":
    main()
