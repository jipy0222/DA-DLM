"""Pure ROUGE scoring for generated text and references."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from progress import track

_METRICS = ("rouge1", "rouge2", "rougeL")


@dataclass(frozen=True)
class RougeResult:
    per_sample: tuple[dict[str, float], ...]
    macro: dict[str, float]


def compute_rouge(
    predictions: Sequence[str], references: Sequence[str],
) -> RougeResult:
    """Return per-sample F1 and unweighted macro means."""
    if isinstance(predictions, (str, bytes)) or isinstance(references, (str, bytes)):
        raise TypeError("predictions and references must be sequences of texts")
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        raise ValueError("ROUGE requires at least one prediction")

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(_METRICS, use_stemmer=True)
    per_sample = []
    totals = {name: 0.0 for name in _METRICS}
    pairs = zip(predictions, references)
    for prediction, reference in track(
        pairs, desc="Scoring ROUGE", total=len(predictions), unit="sample",
    ):
        prediction = "" if prediction is None else str(prediction)
        reference = "" if reference is None else str(reference)
        scores = scorer.score(reference, prediction)
        row = {name: float(scores[name].fmeasure) for name in _METRICS}
        per_sample.append(row)
        for name, value in row.items():
            totals[name] += value
    count = len(per_sample)
    return RougeResult(
        per_sample=tuple(per_sample),
        macro={name: value / count for name, value in totals.items()},
    )


def compute_record_rouge(
    records: Sequence[Mapping[str, Any]], *,
    prediction_key: str = "target_text",
    reference_key: str = "reference_text",
) -> RougeResult:
    """Score dataset-runner records in their existing order."""
    return compute_rouge(
        [record[prediction_key] for record in records],
        [record[reference_key] for record in records],
    )


__all__ = ["RougeResult", "compute_record_rouge", "compute_rouge"]
