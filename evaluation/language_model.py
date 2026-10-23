"""Causal-language-model scoring for generated text.

The scorer tokenizes complete texts once and evaluates every token after the
first with overlapping context windows.  Optional continuation boundaries are
character offsets in those same complete texts; evaluator tokens that overlap a
boundary belong to the continuation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch
import torch.nn.functional as F

from progress import track

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class CausalTextScores:
    full_nll_sums: tuple[float, ...]
    full_token_counts: tuple[int, ...]
    continuation_nll_sums: tuple[float, ...] | None
    continuation_token_counts: tuple[int, ...] | None


def load_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """Load one local evaluator tokenizer without importing Transformers early."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("evaluator tokenizer needs a PAD or EOS token for batching")
        tokenizer.pad_token = tokenizer.eos_token
    return cast('PreTrainedTokenizerBase', tokenizer)


def load_causal_lm(path: str, device: torch.device | str = "cpu") -> PreTrainedModel:
    """Load one local causal LM in its default FP32 evaluation dtype."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True,
    ).eval()
    return cast('PreTrainedModel', model.to(torch.device(device)))


def model_context_length(model: Any) -> int:
    """Read the evaluator's usable attention length from its model config."""
    for name in ("max_position_embeddings", "n_positions"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            return value
    raise ValueError("evaluator model config does not declare a context length")


def tokenize_texts(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    add_special_tokens: bool,
    max_length: int | None = None,
) -> list[list[int]]:
    """Tokenize without padding and optionally apply an explicit right cap."""
    encoded = tokenizer(
        list(texts), add_special_tokens=add_special_tokens,
        padding=False, truncation=False, return_attention_mask=False,
    )["input_ids"]
    rows = [list(map(int, values)) for values in encoded]
    if max_length is not None:
        rows = [values[:max_length] for values in rows]
    return rows


@dataclass(frozen=True)
class _TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor

    def to(self, device: torch.device) -> _TokenBatch:
        return _TokenBatch(self.input_ids.to(device), self.attention_mask.to(device),
                           self.lengths.to(device))


def _pack_token_rows(rows: Sequence[Sequence[int]], pad_token_id: int) -> _TokenBatch:
    """Prepare right padding on CPU; real tokens may have the PAD/EOS ID."""
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long, device="cpu")
    width = max(len(row) for row in rows)
    input_ids = torch.tensor(
        [list(row) + [pad_token_id] * (width - len(row)) for row in rows],
        dtype=torch.long, device="cpu",
    )
    attention_mask = (torch.arange(width, device="cpu")[None] < lengths[:, None]).long()
    return _TokenBatch(input_ids, attention_mask, lengths)


@torch.no_grad()
def featurize_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return final-valid-token hidden states, matching MAUVE featurization."""
    if not texts:
        raise ValueError("MAUVE featurization requires at least one text")
    context_length = model_context_length(model)
    if max_length > context_length:
        raise ValueError(
            f"mauve_max_length {max_length} exceeds evaluator context {context_length}"
        )
    token_rows = tokenize_texts(
        tokenizer, [str(text) for text in texts],
        add_special_tokens=True, max_length=max_length,
    )
    if any(not row for row in token_rows):
        raise ValueError("MAUVE texts must contain at least one evaluator token")

    target_device = torch.device(device)
    feature_model = getattr(model, "base_model", model)
    features = []
    batch_starts = range(0, len(token_rows), batch_size)
    for start in track(
        batch_starts, desc="Extracting MAUVE features",
        total=len(batch_starts), unit="batch",
    ):
        rows = token_rows[start:start + batch_size]
        batch = _pack_token_rows(rows, int(tokenizer.pad_token_id)).to(target_device)
        output = feature_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            return_dict=True,
            use_cache=False,
        )
        row_index = torch.arange(len(rows), device=target_device)
        features.append(
            output.last_hidden_state[row_index, batch.lengths - 1].float().cpu()
        )
    return torch.cat(features, dim=0)


def _tokenize_for_scoring(
    tokenizer: Any,
    texts: Sequence[str],
    continuation_starts: Sequence[int] | None,
    max_length: int | None,
) -> tuple[list[list[int]], list[list[tuple[int, int]]] | None]:
    kwargs = dict(
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    if continuation_starts is not None:
        kwargs["return_offsets_mapping"] = True
    try:
        encoded = tokenizer(list(texts), **kwargs)
    except (NotImplementedError, TypeError, ValueError) as error:
        if continuation_starts is None:
            raise
        raise ValueError(
            "coherence requires a fast evaluator tokenizer with offset mappings"
        ) from error

    input_ids = [list(map(int, values)) for values in encoded["input_ids"]]
    offsets = None
    if continuation_starts is not None:
        raw_offsets = encoded.get("offset_mapping")
        if raw_offsets is None:
            raise ValueError(
                "coherence requires a fast evaluator tokenizer with offset mappings"
            )
        offsets = [
            [(int(start), int(end)) for start, end in values]
            for values in raw_offsets
        ]
    if max_length is not None:
        input_ids = [values[:max_length] for values in input_ids]
        if offsets is not None:
            offsets = [values[:max_length] for values in offsets]
    return input_ids, offsets


def _windows(length: int, context_length: int, stride: int) -> Iterator[tuple[int, int, int]]:
    previous_end = 0
    for start in range(0, length, stride):
        end = min(start + context_length, length)
        if end <= previous_end:
            continue
        yield start, end, previous_end
        previous_end = end
        if end == length:
            break


@dataclass(frozen=True)
class CausalScoringInputs:
    """Unpadded IDs and optional character offsets for one scoring call."""

    token_rows: list[list[int]]
    offsets: list[list[tuple[int, int]]] | None
    continuation_starts: Sequence[int] | None


def prepare_causal_inputs(
    tokenizer: Any, texts: Sequence[str], *,
    continuation_starts: Sequence[int] | None = None,
    max_length: int | None = None,
) -> CausalScoringInputs:
    """Validate text boundaries and tokenize once for scoring and entropy."""
    if len(texts) == 0:
        raise ValueError("language-model scoring requires at least one text")
    if continuation_starts is not None and len(continuation_starts) != len(texts):
        raise ValueError("continuation boundaries must align with texts")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer or null")

    texts = [str(text) for text in texts]
    if continuation_starts is not None:
        starts = []
        for index, (text, value) in enumerate(zip(texts, continuation_starts)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"continuation_start for record {index} must be an integer")
            if value < 0 or value > len(text):
                raise ValueError(f"continuation_start for record {index} is outside full_text")
            starts.append(value)
        continuation_starts = starts

    token_rows, offsets = _tokenize_for_scoring(
        tokenizer, texts, continuation_starts, max_length,
    )
    return CausalScoringInputs(token_rows, offsets, continuation_starts)


def score_causal_texts(
    model: Any, tokenizer: Any, texts: Sequence[str], *,
    continuation_starts: Sequence[int] | None = None,
    batch_size: int = 8, max_length: int | None = None,
    stride: int = 512, device: torch.device | str = "cpu",
) -> CausalTextScores:
    """Return per-text NLL sums/counts with full-length sliding evaluation."""
    inputs = prepare_causal_inputs(
        tokenizer, texts, continuation_starts=continuation_starts, max_length=max_length,
    )
    return score_causal_inputs(
        model, tokenizer, inputs, batch_size=batch_size, stride=stride, device=device,
    )


@dataclass(frozen=True)
class _ScoringBatch:
    tokens: _TokenBatch
    sample_indices: torch.Tensor
    score_mask: torch.Tensor
    continuation_mask: torch.Tensor | None


def _pack_scoring_windows(
    inputs: CausalScoringInputs, windows: Sequence[tuple[int, int, int, int]],
    pad_token_id: int,
) -> _ScoringBatch:
    """Prepare token rows and count-once/character-boundary masks on CPU."""
    tokens = _pack_token_rows(
        [inputs.token_rows[sample][start:end] for sample, start, end, _ in windows],
        pad_token_id,
    )
    sample_indices = torch.tensor([sample for sample, _, _, _ in windows], device="cpu")
    starts = torch.tensor([start for _, start, _, _ in windows], device="cpu")
    previous_ends = torch.tensor([previous for _, _, _, previous in windows], device="cpu")
    local_targets = torch.arange(1, tokens.input_ids.shape[1], device="cpu")[None]
    score_mask = ((local_targets < tokens.lengths[:, None])
                  & (starts[:, None] + local_targets >= previous_ends[:, None]))
    continuation_mask = None
    if inputs.continuation_starts is not None:
        assert inputs.offsets is not None
        width = tokens.input_ids.shape[1]
        offset_ends = torch.tensor([
            [offset_end for _, offset_end in inputs.offsets[sample][start:end]]
            + [0] * (width - (end - start)) for sample, start, end, _ in windows
        ], dtype=torch.long, device="cpu")
        boundaries = torch.tensor(
            [inputs.continuation_starts[sample] for sample, _, _, _ in windows], device="cpu",
        )
        continuation_mask = score_mask & (offset_ends[:, 1:] > boundaries[:, None])
    return _ScoringBatch(tokens, sample_indices, score_mask, continuation_mask)


@torch.no_grad()
def score_causal_inputs(
    model: Any, tokenizer: Any, inputs: CausalScoringInputs, *,
    batch_size: int = 8, stride: int = 512, device: torch.device | str = "cpu",
) -> CausalTextScores:
    """Score inputs prepared by prepare_causal_inputs without retokenizing."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    token_rows = inputs.token_rows
    continuation_starts = inputs.continuation_starts
    context_length = model_context_length(model)
    if stride >= context_length and any(len(row) > context_length for row in token_rows):
        raise ValueError("stride must be smaller than model context length for long texts")

    windows: list[tuple[int, int, int, int]] = []
    for sample, values in enumerate(token_rows):
        for start, end, previous_end in _windows(len(values), context_length, stride):
            windows.append((sample, start, end, previous_end))

    full_sums = torch.zeros(len(token_rows), dtype=torch.float64)
    full_counts = torch.zeros(len(token_rows), dtype=torch.long)
    continuation_sums = (torch.zeros(len(token_rows), dtype=torch.float64)
                         if continuation_starts is not None else None)
    continuation_counts = (torch.zeros(len(token_rows), dtype=torch.long)
                           if continuation_starts is not None else None)
    target_device = torch.device(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("evaluator tokenizer has no pad_token_id")

    batch_starts = range(0, len(windows), batch_size)
    for batch_start in track(
        batch_starts, desc="Scoring causal LM windows",
        total=len(batch_starts), unit="batch",
    ):
        selected = windows[batch_start:batch_start + batch_size]
        width = max(end - start for _, start, end, _ in selected)
        if width < 2:
            continue
        packed = _pack_scoring_windows(inputs, selected, int(pad_token_id))
        batch = packed.tokens.to(target_device)
        score_mask = packed.score_mask.to(target_device)
        cpu_indices = packed.sample_indices
        logits = model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False,
        ).logits
        losses = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            batch.input_ids[:, 1:].reshape(-1), reduction="none",
        ).reshape(len(selected), width - 1)
        row_sums = (losses * score_mask).sum(dim=1, dtype=torch.float64)
        row_counts = score_mask.sum(dim=1)

        if packed.continuation_mask is not None:
            assert continuation_sums is not None and continuation_counts is not None
            continuation_mask = packed.continuation_mask.to(target_device)
            row_continuation_sums = (
                losses * continuation_mask
            ).sum(dim=1, dtype=torch.float64)
            row_continuation_counts = continuation_mask.sum(dim=1)
            batch_stats = torch.stack((
                row_sums, row_counts.to(torch.float64),
                row_continuation_sums, row_continuation_counts.to(torch.float64),
            ), dim=1).cpu()
            continuation_sums.index_add_(0, cpu_indices, batch_stats[:, 2])
            continuation_counts.index_add_(
                0, cpu_indices, batch_stats[:, 3].to(torch.long),
            )
        else:
            batch_stats = torch.stack(
                (row_sums, row_counts.to(torch.float64)), dim=1,
            ).cpu()
        full_sums.index_add_(0, cpu_indices, batch_stats[:, 0])
        full_counts.index_add_(
            0, cpu_indices, batch_stats[:, 1].to(torch.long),
        )

    return CausalTextScores(
        full_nll_sums=tuple(full_sums.tolist()),
        full_token_counts=tuple(full_counts.tolist()),
        continuation_nll_sums=(tuple(continuation_sums.tolist())
                               if continuation_sums is not None else None),
        continuation_token_counts=(tuple(continuation_counts.tolist())
                                   if continuation_counts is not None else None),
    )


__all__ = [
    "CausalTextScores",
    "CausalScoringInputs",
    "prepare_causal_inputs",
    "score_causal_inputs",
    "featurize_texts",
    "load_causal_lm",
    "load_tokenizer",
    "model_context_length",
    "score_causal_texts",
    "tokenize_texts",
]
