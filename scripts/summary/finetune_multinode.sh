#!/usr/bin/env bash
set -eu

NNODES="${NNODES:-${SLURM_NNODES:-}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-}}"
: "${NNODES:?set NNODES or SLURM_NNODES}"
: "${NODE_RANK:?set NODE_RANK or SLURM_NODEID}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank zero hostname}"
: "${MASTER_PORT:?set MASTER_PORT to a free rendezvous port}"

EXPERIMENT="${EXPERIMENT:-bd3lm_cnn}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXPERIMENT}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  mode=finetune \
  experiment="${EXPERIMENT}" \
  output_dir="${OUTPUT_DIR}" \
  runtime.devices="${GPUS_PER_NODE}" \
  runtime.num_nodes="${NNODES}" \
  "$@"
