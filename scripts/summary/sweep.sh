#!/usr/bin/env bash
set -eu

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to a native diffusion summary checkpoint}"
: "${CORPUS_PATH:?set CORPUS_PATH to a local datasets.save_to_disk snapshot}"

EXPERIMENT="${EXPERIMENT:-evaluate_dadlm_cnn}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/evaluation/${EXPERIMENT}_sweep}"

# Each value list is whitespace separated. Every Cartesian-product point gets
# its own leaf because evaluation deliberately refuses to reuse an output path.
NUM_STEPS_VALUES="${NUM_STEPS_VALUES:-64 128 256}"
STRATEGY_VALUES="${STRATEGY_VALUES:-confidence}"
INFILLING_K_VALUES="${INFILLING_K_VALUES:-0}"
INFILLING_MIN_CONTEXT_VALUES="${INFILLING_MIN_CONTEXT_VALUES:-0.0}"
INFILLING_THRESHOLD_VALUES="${INFILLING_THRESHOLD_VALUES:-0.0}"

read -r -a num_steps_values <<< "${NUM_STEPS_VALUES}"
read -r -a strategy_values <<< "${STRATEGY_VALUES}"
read -r -a infilling_k_values <<< "${INFILLING_K_VALUES}"
read -r -a infilling_min_context_values <<< "${INFILLING_MIN_CONTEXT_VALUES}"
read -r -a infilling_threshold_values <<< "${INFILLING_THRESHOLD_VALUES}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for num_steps in "${num_steps_values[@]}"; do
  for strategy in "${strategy_values[@]}"; do
    for infilling_k in "${infilling_k_values[@]}"; do
      for infilling_min_context in "${infilling_min_context_values[@]}"; do
        for infilling_threshold in "${infilling_threshold_values[@]}"; do
          leaf="steps-${num_steps}_strategy-${strategy}_infill-${infilling_k}_context-${infilling_min_context}_threshold-${infilling_threshold}"
          OUTPUT_DIR="${OUTPUT_ROOT}/${leaf}" \
          EXPERIMENT="${EXPERIMENT}" \
            "${script_dir}/evaluate.sh" \
              "generation.diffusion.num_steps=${num_steps}" \
              "generation.diffusion.strategy=${strategy}" \
              "generation.diffusion.infilling_k=${infilling_k}" \
              "generation.diffusion.infilling_min_context=${infilling_min_context}" \
              "generation.diffusion.infilling_threshold=${infilling_threshold}" \
              "$@"
        done
      done
    done
  done
done
