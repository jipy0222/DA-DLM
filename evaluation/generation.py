"""Task options, text decoding and evaluation records for a caller-selected generator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import replace
from os.path import commonprefix
from typing import Any, Sequence, cast

import torch

from data.tokens import NativeTokenSchema
from generation.config import (
    DADLMGenerationConfig,
    DiffusionConfig,
    SamplingConfig,
    StopConfig,
)
from generation.types import GenerationResult


def _blocked_roles(
    required: tuple[int, ...], sampling: SamplingConfig | None,
    *, allowed: tuple[int, ...] = (),
) -> tuple[int, ...]:
    extra = sampling.blocked_token_ids if sampling is not None else ()
    return tuple(dict.fromkeys(token_id for token_id in (*required, *extra)
                               if token_id not in allowed))


def _options(
    schema: NativeTokenSchema,
    *,
    generation_length: int,
    start_token_id: int | None,
    blocked_token_ids: tuple[int, ...],
    sampling: SamplingConfig | None,
    stop: StopConfig,
    diffusion: DiffusionConfig | None,
    dag: DADLMGenerationConfig | None,
) -> dict[str, Any]:
    sample = replace(sampling or SamplingConfig(), blocked_token_ids=blocked_token_ids)
    result: dict[str, Any] = {
        "generation_length": generation_length,
        "start_token_id": start_token_id,
        "sampling": sample,
        "stop": stop,
    }
    if diffusion is not None:
        result["mask_token_id"] = schema.mask_token_id
        result["diffusion"] = diffusion
    if dag is not None:
        if diffusion is None:
            raise ValueError("DADLM options require diffusion configuration")
        result.update(dag=dag, block_bos_id=schema.block_bos, block_eos_id=schema.block_eos)
    return result


def make_text_options(
    schema: NativeTokenSchema,
    *,
    generation_length: int,
    has_prompt: bool,
    sampling: SamplingConfig | None = None,
    stop: StopConfig | None = None,
    diffusion: DiffusionConfig | None = None,
    dag: DADLMGenerationConfig | None = None,
) -> dict[str, Any]:
    """Ordinary text: no default EOS stop; use BOS only without a prompt.

    Schema-forbidden roles are always present in the sampling exclusions;
    additional caller exclusions are retained.
    An explicit ``stop`` can enable an ordinary EOS or any other token. If
    ``diffusion`` is absent, no diffusion-only keyword reaches an AR generator.
    """
    return _options(
        schema, generation_length=generation_length,
        start_token_id=None if has_prompt else schema.bos_token_id,
        blocked_token_ids=_blocked_roles(schema.blocked_token_ids, sampling),
        sampling=sampling, stop=stop or StopConfig(), diffusion=diffusion, dag=dag,
    )


def make_summary_options(
    schema: NativeTokenSchema,
    *,
    generation_length: int,
    sampling: SamplingConfig | None = None,
    stop: StopConfig | None = None,
    diffusion: DiffusionConfig | None = None,
    dag: DADLMGenerationConfig | None = None,
) -> dict[str, Any]:
    """Source-conditioned summary: fixed summary BOS and default summary EOS.

    ``trim_suffix`` is an immediate stop for AR and a suffix-trimming stop for
    a diffusion block. Pass ``StopConfig()`` to disable stopping explicitly.
    The summary EOS is an allowed candidate; all other task/graph roles stay
    forbidden as ordinary target predictions.
    """
    blocked = _blocked_roles(schema.blocked_token_ids, sampling,
                             allowed=(schema.summary_eos,))
    return _options(
        schema, generation_length=generation_length,
        start_token_id=schema.summary_bos, blocked_token_ids=blocked,
        sampling=sampling,
        stop=stop or StopConfig(mode="trim_suffix", token_ids=(schema.summary_eos,)),
        diffusion=diffusion, dag=dag,
    )


def _ids(values: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError("token IDs must be one-dimensional")
        return cast('list[int]', values.tolist())
    return list(values)


def decode_target_body(
    tokenizer: Any,
    token_ids: Sequence[int] | torch.Tensor,
    *,
    start_token_id: int | None = None,
    terminal_token_ids: Sequence[int] = (),
) -> str:
    """Decode only the generated body, removing at most boundary markers.

    An EOS in the middle is never used as a second, implicit stopping rule.
    ``skip_special_tokens`` may suppress the marker itself, but later body
    tokens remain in the one decode call when stopping was disabled.
    """
    ids = _ids(token_ids)
    if ids and start_token_id is not None and ids[0] == start_token_id:
        ids = ids[1:]
    if ids and ids[-1] in terminal_token_ids:
        ids = ids[:-1]
    return cast('str', tokenizer.decode(ids, skip_special_tokens=True))


def decode_prompt_and_target(
    tokenizer: Any,
    prompt_ids: Sequence[int] | torch.Tensor,
    target_ids: Sequence[int] | torch.Tensor,
    *,
    pad_token_id: int,
) -> str:
    """Decode effective prompt plus target in one tokenizer call for MAUVE.

    Prepared summary prompts are right-padded; only trailing PAD is removed.
    Decoding two strings separately can alter byte-level/BPE boundary text.
    """
    prompt = _ids(prompt_ids)
    while prompt and prompt[-1] == pad_token_id:
        prompt.pop()
    return cast('str', tokenizer.decode(prompt + _ids(target_ids), skip_special_tokens=True))


def generate_batch_records(
    generate: Callable[..., GenerationResult],
    model: Any,
    tokenizer: Any,
    batch: Mapping[str, Any],
    *,
    generation_kwargs: Mapping[str, Any],
    ema: Any = None,
    include_prompt: bool = False,
    pad_token_id: int | None = None,
) -> tuple[GenerationResult, list[dict[str, Any]]]:
    """Run one full batch and return its raw result plus scored output rows.

    A zero-weight distributed padding copy still participates in generation;
    it is omitted only from returned records. Thus the unfiltered raw stats
    continue to report its actual work. ``batch`` accepts the source-only
    summary collator's prompt/reference fields without adapting the sampler.
    The generator owns token/length invariants; this wrapper checks row
    alignment with the batch metadata.
    """
    if include_prompt and pad_token_id is None:
        raise ValueError("pad_token_id is required for prompt-plus-target decoding")
    prompt_ids = batch["prompt_ids"]
    if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2:
        raise ValueError("batch prompt_ids must be a [batch, source] tensor")
    batch_size = prompt_ids.shape[0]
    weights = batch.get("sample_weight")
    references = batch.get("reference_texts")
    full_references = batch.get("reference_full_texts")
    prompt_texts = batch.get("prompt_texts")
    for name, values in (("sample_weight", weights),
                         ("reference_texts", references),
                         ("reference_full_texts", full_references),
                         ("prompt_texts", prompt_texts)):
        if values is not None and len(values) != batch_size:
            raise ValueError(f"{name} batch size disagrees with prompt batch")

    context = ema.average_parameters(model) if ema is not None else nullcontext()
    with context:
        result = generate(model, prompt_ids, **generation_kwargs)
    if result.token_ids.shape[0] != batch_size:
        raise ValueError("prompt and result batch sizes disagree")

    stop = generation_kwargs.get("stop", StopConfig())
    terminal = stop.token_ids if stop.mode != "none" else ()
    start = generation_kwargs.get("start_token_id")
    token_rows = result.token_ids.tolist()
    lengths = result.lengths.tolist()
    weights = weights.reshape(batch_size).tolist() if isinstance(weights, torch.Tensor) else weights
    prompt_rows = prompt_ids.tolist() if include_prompt else None
    records: list[dict[str, Any]] = []
    for row in range(batch_size):
        if weights is not None and float(weights[row]) == 0.0:
            continue
        length = int(lengths[row])
        target = token_rows[row][:length]
        record: dict[str, Any] = {
            "row_index": row,
            "token_ids": target,
            "length": length,
            "finish_reason": result.finish_reason[row],
            "target_text": decode_target_body(
                tokenizer, target, start_token_id=start,
                terminal_token_ids=terminal,
            ),
        }
        if references is not None:
            record["reference_text"] = references[row]
        if prompt_texts is not None:
            record["prompt_text"] = prompt_texts[row]
        if full_references is not None:
            record["reference_full_text"] = full_references[row]
        if include_prompt:
            record["full_text"] = decode_prompt_and_target(
                tokenizer, cast(list[list[int]], prompt_rows)[row], target, pad_token_id=cast(int, pad_token_id),
            )
            prompt = cast(list[list[int]], prompt_rows)[row]
            while prompt and prompt[-1] == pad_token_id:
                prompt.pop()
            record["prompt_ids"] = prompt
            decoded_prompt = tokenizer.decode(prompt, skip_special_tokens=True)
            # A character completed across the prompt boundary belongs to the target.
            record["continuation_start"] = len(commonprefix((decoded_prompt, record["full_text"])))
        records.append(record)
    return result, records
