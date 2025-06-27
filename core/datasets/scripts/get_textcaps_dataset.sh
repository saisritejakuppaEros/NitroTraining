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
num_gpus=$2
savedir=$3 # remote dir to write the processed files

num_proc=16
batch_size=32 # use batch size of 8 for <16GB GPU memory

# Textcaps is fairly small so we download all of it. We also download and process the data is a single script.
python prepare/textcaps/convert.py --local_mds_dir "${datadir}/mds/"

# Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
accelerate launch --multi_gpu --num_processes $num_gpus --main_process_port 12346 prepare/textcaps/precompute.py --datadir "${datadir}/mds/" \
    --savedir "${savedir}/mds_latents_dcae_llama3.2-1b/" --vae mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
    --text_encoder meta-llama/Llama-3.2-1B --batch_size $batch_size