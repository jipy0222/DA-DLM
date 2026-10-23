#!/usr/bin/env bash
set -eu

# Run once per node; Hydra overrides, including startup.path, go last.
NNODES="${NNODES:-${SLURM_NNODES:-}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-}}"
: "${NNODES:?set NNODES or SLURM_NNODES}"
: "${NODE_RANK:?set NODE_RANK or SLURM_NODEID}"
: "${MASTER_ADDR:?set MASTER_ADDR to rank zero hostname}"
: "${MASTER_PORT:?set MASTER_PORT to a free rendezvous port}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/dadlm_owt_b16}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  mode=pretrain \
  experiment=dadlm_owt \
  output_dir="${OUTPUT_DIR}" \
  runtime.devices="${GPUS_PER_NODE}" \
  runtime.num_nodes="${NNODES}" \
  "$@"
