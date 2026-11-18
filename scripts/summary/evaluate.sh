#!/usr/bin/env bash
set -eu

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to a native summary checkpoint}"
: "${CORPUS_PATH:?set CORPUS_PATH to a local datasets.save_to_disk snapshot}"

EXPERIMENT="${EXPERIMENT:-evaluate_ar_cnn}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/evaluation/${EXPERIMENT}}"

if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  set -- "startup.source_tokenizer_path=${TOKENIZER_PATH}" "$@"
fi

python main.py \
  mode=evaluate_summary \
  experiment="${EXPERIMENT}" \
  startup.path="${CHECKPOINT_PATH}" \
  data.corpus_path="${CORPUS_PATH}" \
  output_dir="${OUTPUT_DIR}" \
  "$@"
