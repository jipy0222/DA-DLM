#!/usr/bin/env bash
set -eu

# Run once before training with local corpus and tokenizer snapshots.
: "${CORPUS_PATH:?set CORPUS_PATH to the local OpenWebText snapshot}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH to the local tokenizer directory}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/ar_owt}"
REVISION="${REVISION:-local-owt-snapshot}"

python main.py \
  mode=prepare \
  data=owt \
  output_dir="${OUTPUT_DIR}" \
  data.corpus_path="${CORPUS_PATH}" \
  data.revision="${REVISION}" \
  data.tokenizer_path="${TOKENIZER_PATH}" \
  logging.backend=none \
  "$@"
