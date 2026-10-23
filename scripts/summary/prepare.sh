#!/usr/bin/env bash
set -eu

: "${CORPUS_PATH:?set CORPUS_PATH to a local datasets.save_to_disk snapshot}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH to the local tokenizer directory}"

EXPERIMENT="${EXPERIMENT:-ar_cnn}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXPERIMENT}}"
if [[ -n "${REVISION:-}" ]]; then
  set -- "data.revision=${REVISION}" "$@"
fi

python main.py \
  mode=prepare_summary \
  experiment="${EXPERIMENT}" \
  output_dir="${OUTPUT_DIR}" \
  data.corpus_path="${CORPUS_PATH}" \
  data.tokenizer_path="${TOKENIZER_PATH}" \
  logging.backend=none \
  "$@"
