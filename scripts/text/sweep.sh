#!/usr/bin/env bash
set -eu

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to a native diffusion text checkpoint}"
: "${CORPUS_PATH:?set CORPUS_PATH to a local datasets.save_to_disk snapshot}"

EXPERIMENT="${EXPERIMENT:-evaluate_dadlm_owt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/evaluation/${EXPERIMENT}_sweep}"

# Whitespace-separated value lists define a Cartesian product of CLI overrides.
NUM_STEPS_VALUES="${NUM_STEPS_VALUES:-64 128}"
STRATEGY_VALUES="${STRATEGY_VALUES:-default}"
TOP_P_VALUES="${TOP_P_VALUES:-0.9}"

read -r -a num_steps_values <<< "${NUM_STEPS_VALUES}"
read -r -a strategy_values <<< "${STRATEGY_VALUES}"
read -r -a top_p_values <<< "${TOP_P_VALUES}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for num_steps in "${num_steps_values[@]}"; do
  for strategy in "${strategy_values[@]}"; do
    for top_p in "${top_p_values[@]}"; do
      leaf="steps-${num_steps}_strategy-${strategy}_top-p-${top_p}"
      OUTPUT_DIR="${OUTPUT_ROOT}/${leaf}" \
      EXPERIMENT="${EXPERIMENT}" \
      MODE=evaluate_text \
        bash "${script_dir}/evaluate.sh" \
          "evaluation.store_dir=${OUTPUT_ROOT}/text_data" \
          "generation.diffusion.num_steps=${num_steps}" \
          "generation.diffusion.strategy=${strategy}" \
          "generation.sampling.top_p=${top_p}" \
          "$@"
    done
  done
done
