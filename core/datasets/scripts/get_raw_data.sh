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

# Prepare raw_data (split_XX/00000.tar img2dataset format) for Nitro-T training.
# No download step - data is already at raw_data.
#
# Usage: run from core/datasets/
#   ./scripts/get_raw_dataset.sh <datadir> <num_gpus> <savedir> [raw_dir]
#
# Example:
#   ./scripts/get_raw_dataset.sh /path/to/output 8 /path/to/savedir
#
# Arguments:
#   datadir  - local dir for MDS output (convert writes to datadir/mds/)
#   num_gpus - number of GPUs for precompute
#   savedir  - dir to write precomputed latents (savedir/mds_latents_...)
#   raw_dir  - (optional) path to raw_data; default: /data/corerndimage/DiffusionModels/datapruning/raw_data

# ./scripts/get_raw_dataset.sh ./raw_dataset_output 3 ./raw_dataset_output /data/corerndimage/DiffusionModels/datapruning/raw_data

# cd /data/corerndimage/DiffusionModels/datapruning/raw_data &&  ./get_data.sh

# cd /data/corerndimage/DiffusionModels/datapruning/model_training/Nitro-T/core/datasets




# # Which physical GPUs to use (e.g. 3,4 = use GPUs 3 and 4; these show in nvidia-smi).
# # To use GPUs 2,3,4,6: CUDA_VISIBLE_DEVICES=2,3,4,6 bash scripts/get_raw_data.sh . 4
# # Default: use GPUs 2,3,4,6 (4 GPUs, avoiding GPU 5 which has vLLM)
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,6}
# # Note: If you get "Duplicate GPU detected" errors, num_processes in accelerate config must match GPU count.

# # NCCL settings (set before PyTorch/accelerate)
export NCCL_NVLS_ENABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN

set -e  # Exit on any error

# datadir=${1:-./raw_dataset_output}
# num_gpus=${2:-4}
# savedir=${3:-./raw_dataset_output}
# raw_dir=${4:-/datasets/ai-core-object/d-gpu-06097851-2053-4b67-8400-b5d404c04261/teja/internet_dataset/laionasthetic_v2/split_000}

# num_proc=16
# batch_size=128

# echo "Raw data path: ${raw_dir}"
# echo "MDS output: ${datadir}/mds/"
# echo "Latents output: ${savedir}/mds_latents_dcae_llama3.2-1b/"
# echo "Num GPUs: ${num_gpus}"
# echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES} (these physical GPUs will be used; check nvidia-smi)"

# # Verify GPU count matches
# GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
# echo "PyTorch detects: ${GPU_COUNT} GPUs"
# if [ "${GPU_COUNT}" -ne "${num_gpus}" ]; then
#     echo "WARNING: CUDA_VISIBLE_DEVICES has ${num_gpus} indices but PyTorch only sees ${GPU_COUNT} GPUs!"
#     echo "This will cause 'Duplicate GPU detected' errors. Adjust num_gpus to ${GPU_COUNT} or fix CUDA_VISIBLE_DEVICES."
# fi

# # A. Convert tar -> MDS (skip download - data already at raw_data)
# echo "Converting raw_data to MDS format..."
# python prepare/raw_dataset/convert.py \
#     --raw_dir "${raw_dir}" \
#     --local_mds_dir "${datadir}/mds/" \
#     --max_image_size 512 \
#     --num_proc ${num_proc}

# # B. Precompute latents across multiple GPUs
# echo "Precomputing latents..."
# python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"

# # Force accelerate to respect CUDA_VISIBLE_DEVICES by using explicit config override
# # This prevents accelerate from setting its own gpu_ids
# ACCELERATE_USE_CUDA_VISIBLE_DEVICES=1 accelerate launch \
#    --config_file /root/.cache/huggingface/accelerate/default_config.yaml \
#    prepare/raw_dataset/precompute.py \
#    --datadir "${datadir}/mds/" \
#    --savedir "${savedir}/mds_latents_dcae_llama3.2-1b/" \
#    --vae mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
#    --text_encoder meta-llama/Llama-3.2-1B \
#    --batch_size ${batch_size}

# echo "Done. Precomputed latents at ${savedir}/mds_latents_dcae_llama3.2-1b/"










#!/bin/bash
# set -euo pipefail

# ============================================================
# CONFIG
# ============================================================
# Fix "Too many open files" (Errno 24) - multi-GPU + DataLoader workers + shared memory need higher limit
ulimit -n 65536 2>/dev/null || true

num_proc=48
batch_size=64
num_gpus=4   # set to your actual GPU count
export CUDA_VISIBLE_DEVICES=4,5,6,7

S3_BASE="ai-core-object/d-gpu-2b453449-a6ca-4718-b9bd-26b778c9ad7f/d-gpu-06097851-2053-4b67-8400-b5d404c04261/teja/internet_dataset"
RAW_BASE="/data/corerndimage/DiffusionModels/datapruning/raw_data"
S3_LATENTS_BASE="${S3_BASE}/amd_latents"

LOCAL_BASE="./tmp/pipeline"
LOCAL_MDS="${LOCAL_BASE}/mds"
LOCAL_LATENTS="${LOCAL_BASE}/latents"

TOTAL_SPLITS=38

# ============================================================
# HELPERS
# ============================================================
pad_split() {
    printf "split_%03d" "$1"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ============================================================
# MAIN LOOP
# ============================================================
mkdir -p "${LOCAL_MDS}" "${LOCAL_LATENTS}"

for i in $(seq 35 $((TOTAL_SPLITS - 1))); do
    SPLIT=$(pad_split "$i")
    log "========================================================"
    log "Processing ${SPLIT} (${i}/${TOTAL_SPLITS})"
    log "========================================================"

    RAW_DIR="${RAW_BASE}/${SPLIT}"
    MDS_DIR="${LOCAL_MDS}/${SPLIT}"
    LATENTS_DIR="${LOCAL_LATENTS}/${SPLIT}"

    # Clean MDS/latents dirs before starting (avoids FileExistsError from leftover data)
    # rm -rf "${MDS_DIR}" "${LATENTS_DIR}"
    mkdir -p "${MDS_DIR}" "${LATENTS_DIR}"

    # ----------------------------------------------------------
    # STEP 1: Convert raw (read directly from S3) -> MDS
    # ----------------------------------------------------------
    log "[${SPLIT}] Converting raw data from S3 to MDS format..."
    python prepare/raw_dataset/convert.py \
        --raw_dir "${RAW_DIR}" \
        --local_mds_dir "${MDS_DIR}" \
        --max_image_size 512 \
        --num_proc ${num_proc}
    
    # Verify conversion succeeded by checking for index.json
    if [ ! -f "${MDS_DIR}/index.json" ]; then
        log "ERROR: MDS conversion failed - index.json not found at ${MDS_DIR}/index.json"
        exit 1
    fi
    log "[${SPLIT}] MDS conversion done."

    # ----------------------------------------------------------
    # STEP 2: Clean shared memory before accelerate launch
    # ----------------------------------------------------------
    python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"

    # ----------------------------------------------------------
    # STEP 3: Precompute latents
    # ----------------------------------------------------------
    log "[${SPLIT}] Precomputing latents..."
    # num_processes MUST match GPU count (CUDA_VISIBLE_DEVICES) to avoid "Duplicate GPU detected"
    # (config_file alternative: --config_file /path/to/accelerate_config.yaml)
    ACCELERATE_USE_CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --num_processes ${num_gpus} \
        prepare/raw_dataset/precompute.py \
        --datadir "${MDS_DIR}" \
        --savedir "${LATENTS_DIR}" \
        --vae mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
        --text_encoder meta-llama/Llama-3.2-1B \
        --batch_size ${batch_size}
    log "[${SPLIT}] Latents precomputed."

    # ----------------------------------------------------------
    # STEP 4: Push latents to S3, delete local immediately after
    # ----------------------------------------------------------
    log "[${SPLIT}] Pushing latents to S3..."
    mc-minio mirror \
        --overwrite \
        --retry \
        --max-workers 4 \
        "${LATENTS_DIR}" \
        "${S3_LATENTS_BASE}/${SPLIT}"

    log "[${SPLIT}] Push complete. Removing local MDS + latents..."
    rm -rf "${MDS_DIR}" "${LATENTS_DIR}"
    log "[${SPLIT}] Local data removed."
    log "[${SPLIT}] Done."

    # break the loop
    # break
done

log "All ${TOTAL_SPLITS} splits processed successfully."
log "Latents available at: ${S3_LATENTS_BASE}/"