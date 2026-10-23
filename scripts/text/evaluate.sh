#!/usr/bin/env bash
set -eu

MODE="${MODE:-evaluate_text}"
EXPERIMENT="${EXPERIMENT:-evaluate_ar_owt}"

# Re-score a saved generation without a checkpoint or corpus:
# MODE=score_text RECORDS_PATH=/path/to/generated bash scripts/text/evaluate.sh
case "${MODE}" in
  evaluate_text)
    : "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to a native text checkpoint}"
    : "${CORPUS_PATH:?set CORPUS_PATH to a local datasets.save_to_disk snapshot}"
    OUTPUT_DIR="${OUTPUT_DIR:-./outputs/evaluation/${EXPERIMENT}}"
    set -- \
      "startup.path=${CHECKPOINT_PATH}" \
      "data.corpus_path=${CORPUS_PATH}" \
      "$@"
    if [[ -n "${TOKENIZER_PATH:-}" ]]; then
      set -- "startup.source_tokenizer_path=${TOKENIZER_PATH}" "$@"
    fi
    if [[ -n "${REVISION:-}" ]]; then
      set -- "data.revision=${REVISION}" "$@"
    fi
    ;;
  score_text)
    : "${RECORDS_PATH:?set RECORDS_PATH to a completed text generation directory}"
    OUTPUT_DIR="${OUTPUT_DIR:-${RECORDS_PATH%/}_scores}"
    set -- "evaluation.records_path=${RECORDS_PATH}" "$@"
    ;;
  *)
    echo "MODE must be evaluate_text or score_text" >&2
    exit 2
    ;;
esac

if [[ -n "${SCORING_MODEL_PATH:-}" ]]; then
  set -- "evaluation.scoring.model_path=${SCORING_MODEL_PATH}" "$@"
fi
if [[ -n "${SCORING_TOKENIZER_PATH:-}" ]]; then
  set -- "evaluation.scoring.tokenizer_path=${SCORING_TOKENIZER_PATH}" "$@"
fi

python main.py \
  mode="${MODE}" \
  experiment="${EXPERIMENT}" \
  output_dir="${OUTPUT_DIR}" \
  "$@"
