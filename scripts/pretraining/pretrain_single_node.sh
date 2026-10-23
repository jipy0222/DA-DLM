#!/usr/bin/env bash
set -eu

# Hydra overrides, including startup.path, follow the script name.
EXPERIMENT="${EXPERIMENT:-ar_owt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXPERIMENT}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" main.py \
  mode=pretrain \
  experiment="${EXPERIMENT}" \
  output_dir="${OUTPUT_DIR}" \
  runtime.devices="${GPUS_PER_NODE}" \
  runtime.num_nodes=1 \
  "$@"
