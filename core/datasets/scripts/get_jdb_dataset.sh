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

#!/bin/bash

# Get user input for data directory and dataset size
datadir=$1 # local dir to download raw dataset
dataset_size=$2 # small or all
num_gpus=$3
savedir=$4 # remote dir to write the processed files

num_proc=16
batch_size=32 # use batch size of 8 for <16GB GPU memory

# A. Download a small subset (~1%) of the dataset, if specified
if [ "$dataset_size" == "small" ]; then
    echo "Downloading ~1% of the dataset..."
    python prepare/jdb/download.py --datadir $datadir --max_image_size 512 --min_image_size 256 --valid_ids 0 1 --num_proc 2
# Or download the entire dataset, if specified
elif [ "$dataset_size" == "all" ]; then
    echo "Downloading the full dataset..."
    python prepare/jdb/download.py --datadir $datadir --max_image_size 512 --min_image_size 256 --num_proc $num_proc
else
    echo "Invalid dataset size option. Please use 'small' or 'all'."
    exit 1
fi

# B. Convert dataset to MDS format.
python prepare/jdb/convert.py --images_dir "${datadir}/raw/train/imgs/" --captions_jsonl "${datadir}/raw/train/train_anno_realease_repath.jsonl" --local_mds_dir "${datadir}/mds/train/"
python prepare/jdb/convert.py --images_dir "${datadir}/raw/valid/imgs/" --captions_jsonl "${datadir}/raw/valid/valid_anno_repath.jsonl" --local_mds_dir "${datadir}/mds/valid/"

# C. Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
for split in train valid; do
    accelerate launch --multi_gpu --num_processes $num_gpus prepare/jdb/precompute.py --datadir "${datadir}/mds/$split/" \
        --savedir "${savedir}/mds_latents_dcae_llama3.2-1b/$split/" --vae mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
        --text_encoder meta-llama/Llama-3.2-1B --batch_size $batch_size
done
