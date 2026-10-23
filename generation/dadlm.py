"""Path-based diffusion with read-only DAG previews and completed-content commits."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from graph import runtime as graph
from graph.sample_path import SampledPath
from graph.viterbi_path import ViterbiPath
from kv_cache import CacheDelta, CacheSession
from models.backbone import ConditioningMode, TransitionNormalization
from progress import progress_bar, track

from .config import DADLMGenerationConfig, DiffusionConfig, SamplingConfig, StopConfig
from .dag import initialize_dag, update_dag
from .dag_reveal import reveal_dag_tokens
from .runtime import cache_dtype, generation_context, make_result, prepare_request
from .sampling import candidate_probabilities, sample_candidates
from .types import GenerationResult, GenerationStats

if TYPE_CHECKING:
    from models import DADLMModel


class _DAGLayoutArgs(TypedDict):
    dag_up_ratio: int
    grouped: bool
    content_lengths: torch.Tensor
    mask_token_id: int
    pad_token_id: int


def _decode_dag(ids: torch.Tensor, probabilities: torch.Tensor, transitions: torch.Tensor, *, mask_token_id: int,
                pad_token_id: int, dag: DADLMGenerationConfig, generator: torch.Generator | None=None) -> tuple[ViterbiPath | SampledPath, torch.Tensor, torch.Tensor]:
    """Decode original edge scores; known nodes contribute zero emission cost."""
    batch, nodes = ids.shape
    node_lengths = ids.new_full((batch,), nodes)
    candidates = ids.clone()
    confidence = probabilities.new_ones(ids.shape)
    if dag.decoder == "viterbi":
        mass, tokens = probabilities.max(dim=-1)
        unknown = ids.eq(mask_token_id)
        candidates[unknown] = tokens[unknown]
        confidence[unknown] = mass[unknown]
        node_scores = torch.where(unknown, mass.log(), 0).to(transitions.dtype)
        node_scores.masked_fill_(ids.eq(pad_token_id), -torch.inf)
        paths = graph.viterbi_path(node_scores, transitions, node_lengths,
                                  backend=dag.graph_backend)
    else:
        uniforms = torch.rand((batch, nodes), device=ids.device,
                              dtype=transitions.dtype, generator=generator)
        paths = graph.sample_path(transitions, node_lengths, uniforms,
                                 backend=dag.graph_backend)
        row, slot = (torch.arange(nodes, device=ids.device)[None] <
                     paths.lengths[:, None]).nonzero(as_tuple=True)
        node = paths.nodes[row, slot]
        unknown = ids[row, node].eq(mask_token_id)
        row, node = row[unknown], node[unknown]
        if row.numel():
            selected = probabilities[row, node]
            tokens = sample_candidates(selected, greedy=False, generator=generator)
            candidates[row, node] = tokens
            confidence[row, node] = selected.gather(-1, tokens[:, None]).squeeze(-1)
    return paths, candidates, confidence


def _validate_request(model: DADLMModel, prompt_ids: torch.Tensor, generation_length: int, start_token_id: int | None,
                      mask_token_id: int, block_bos_id: int, block_eos_id: int,
                      diffusion: DiffusionConfig, dag: DADLMGenerationConfig, sampling: SamplingConfig, stop: StopConfig) -> tuple[SamplingConfig, torch.Tensor, bool]:
    config = model.config
    if not isinstance(diffusion, DiffusionConfig) or not isinstance(dag, DADLMGenerationConfig):
        raise TypeError("diffusion and dag must be their configuration types")
    if not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig")
    if sampling.greedy != (dag.decoder == "viterbi"):
        raise ValueError("FFBS requires greedy=False; Viterbi requires greedy=True")
    crf = config.transition_normalization is TransitionNormalization.RAW_POTENTIAL
    if not config.grouped:
        if not crf and dag.decoder == "viterbi":
            raise ValueError("free local-probability DADLM does not support Viterbi")
        if diffusion.strategy in {"first_hitting", "confidence_threshold"}:
            raise ValueError(f"free DADLM does not support {diffusion.strategy}")
    if config.conditioning is not ConditioningMode.ZERO_TIME_COMPAT:
        raise ValueError("DADLM generation requires ZERO_TIME_COMPAT conditioning")
    if config.nodes_per_block - 2 < config.block_size:
        raise ValueError("DAG content capacity must fit a full block")
    roles = (mask_token_id, config.pad_token_id, block_bos_id, block_eos_id)
    if any(type(token) is not int or not 0 <= token < config.vocab_size for token in roles):
        raise ValueError("DAG special tokens must be vocabulary IDs")
    if len(set(roles)) != len(roles):
        raise ValueError("MASK, PAD and graph endpoints must have distinct IDs")
    if start_token_id in (block_bos_id, block_eos_id):
        raise ValueError("a graph endpoint cannot be an ordinary start token")
    sampling = replace(sampling, blocked_token_ids=tuple(sorted(set(
        (*sampling.blocked_token_ids, block_bos_id, block_eos_id)))))
    sampling = prepare_request(model, prompt_ids, generation_length, start_token_id,
                               sampling, stop, mask_token_id=mask_token_id)
    prompt_lengths = prompt_ids.ne(config.pad_token_id).sum(dim=-1)
    total = int(prompt_lengths.max()) + generation_length
    if total > model._content_limit:
        raise ValueError("prompt and generation budget exceed content capacity")
    attention, position = config.preview_capacity(total - 1)
    if max(prompt_ids.shape[1], attention) > config.backbone.max_attention_length:
        raise ValueError("prompt and full DAG preview exceed max_attention_length")
    if position > config.backbone.max_position_id:
        raise ValueError("full DAG preview exceeds max_position_id")
    return sampling, prompt_lengths, crf


@dataclass
class _DAGPredictions:
    """Predictions and optional CRF suffixes from the same pre-update graph."""

    probabilities: torch.Tensor | None = None
    transitions: torch.Tensor | None = None
    suffix: torch.Tensor | None = None

    def refresh(
        self, model: DADLMModel, ids: torch.Tensor, edges: torch.Tensor,
        session: CacheSession, rows: torch.Tensor, *, original_rows: torch.Tensor,
        stats: GenerationStats, sampling: SamplingConfig, dag: DADLMGenerationConfig,
        need_suffix: bool,
    ) -> None:
        batch, nodes = ids.shape
        if rows.numel() == batch:
            # Release full old predictions before allocating their replacements.
            self.probabilities = self.transitions = None
        selected_session = (session if rows.numel() == batch else
                            CacheSession(session.cache.select_rows(rows)))
        prediction = model.preview_dag(ids[rows], edges[rows], selected_session)
        stats.record("prediction", original_rows[rows])
        new_probs = candidate_probabilities(prediction.logits.float(), sampling)
        new_transitions = prediction.links.scores.float()
        if self.probabilities is None:
            self.probabilities = new_probs
            self.transitions = new_transitions
            if need_suffix and self.suffix is None:
                self.suffix = new_transitions.new_empty((batch, nodes))
        else:
            self.probabilities[rows] = new_probs
            assert self.transitions is not None
            self.transitions[rows] = new_transitions
        if need_suffix:
            assert self.suffix is not None
            self.suffix[rows] = graph.path_suffix_log_partition(
                new_transitions, ids.new_full((rows.numel(),), nodes), backend=dag.graph_backend)

    def select(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.probabilities is not None and self.transitions is not None
        if rows.numel() == self.probabilities.shape[0]:
            return self.probabilities, self.transitions
        return self.probabilities[rows], self.transitions[rows]


@dataclass
class _DAGProgress:
    """Completion state for the physical rows of one graph block."""

    done: torch.Tensor
    remaining: torch.Tensor
    contents: list[torch.Tensor | None]
    stop_at: list[int | None]

    def accept_paths(
        self, ids: torch.Tensor, rows: torch.Tensor, path_nodes: torch.Tensor,
        path_lengths: list[int], *, pad_token_id: int, mask_token_id: int,
        original_rows: torch.Tensor, block_index: int, has_start: bool,
        stop_ids: torch.Tensor | None,
    ) -> None:
        for local, row in enumerate(rows.tolist()):
            path = path_nodes[local, 1:path_lengths[local] - 1]
            path = path[ids[row, path].ne(pad_token_id)]
            tokens = ids[row, path]
            self.remaining[row] = tokens.eq(mask_token_id).sum()
            if self.remaining[row] == 0:
                if tokens.numel() == 0:
                    raise RuntimeError(f"DADLM empty content block: row={int(original_rows[row])}, block={block_index}")
                self.contents[row] = tokens
                self.done[row] = True
                if stop_ids is not None:
                    eligible = torch.isin(tokens, stop_ids)
                    if has_start:
                        eligible &= path.ne(1)
                    matches = eligible.nonzero(as_tuple=True)[0]
                    if matches.numel():
                        self.stop_at[row] = int(matches[0]) + 1


def _decoded_path_lengths(
    lengths: torch.Tensor, original_rows: torch.Tensor, rows: torch.Tensor, block_index: int,
) -> list[int]:
    path_lengths = cast(list[int], lengths.tolist())
    for local, length in enumerate(path_lengths):
        if length <= 2:
            row = int(original_rows[rows[local]])
            kind = "unreachable path" if length == 0 else "empty content block"
            raise RuntimeError(f"DADLM {kind}: row={row}, block={block_index}")
    return path_lengths


def _denoise_dag(model: DADLMModel, ids: torch.Tensor, content_lengths: torch.Tensor, session: CacheSession, *, original_rows: torch.Tensor,
                 block_index: int, mask_token_id: int, diffusion: DiffusionConfig, dag: DADLMGenerationConfig, sampling: SamplingConfig, stop: StopConfig,
                 has_start: bool, stats: GenerationStats, crf: bool, generator: torch.Generator | None) -> tuple[list[torch.Tensor], list[int | None]]:
    """Complete one graph per row; unchanged rows reuse their model predictions."""
    pad = model.config.pad_token_id
    batch, nodes = ids.shape
    terminal = ids.new_full((batch,), nodes)
    layout_args: _DAGLayoutArgs = dict(dag_up_ratio=model.config.dag_up_ratio,
                       grouped=model.config.grouped, content_lengths=content_lengths,
                       mask_token_id=mask_token_id, pad_token_id=pad)
    ids, edges = update_dag(ids, terminal_positions=terminal, **layout_args)
    progress = _DAGProgress(
        done=torch.zeros(batch, device=ids.device, dtype=torch.bool),
        remaining=ids.new_zeros(batch), contents=[None] * batch, stop_at=[None] * batch,
    )
    dirty = ~progress.done
    predictions = _DAGPredictions()
    need_suffix = crf and dag.confidence_transition_weight > 0 and diffusion.strategy in {
        "confidence", "confidence_threshold"}
    stop_ids: torch.Tensor | None = None
    position: torch.Tensor | None = None
    if stop.mode != "none":
        stop_ids = ids.new_tensor(stop.token_ids)
    if stop.mode == "trim_suffix":
        position = torch.arange(nodes, device=ids.device)

    for step in track(range(diffusion.num_steps), desc="DAG denoising", unit="update"):
        predict_rows = (dirty & ~progress.done).nonzero(as_tuple=True)[0]
        if predict_rows.numel():
            predictions.refresh(
                model, ids, edges, session, predict_rows, original_rows=original_rows,
                stats=stats, sampling=sampling, dag=dag, need_suffix=need_suffix,
            )
            dirty[predict_rows] = False

        rows = (~progress.done).nonzero(as_tuple=True)[0]
        current = ids[rows]
        active_probabilities, active_transitions = predictions.select(rows)
        paths, candidates, confidence = _decode_dag(
            current, active_probabilities, active_transitions, mask_token_id=mask_token_id,
            pad_token_id=pad, dag=dag, generator=generator)
        path_lengths = _decoded_path_lengths(paths.lengths, original_rows, rows, block_index)
        updated = reveal_dag_tokens(
            current, candidates, paths.nodes, paths.lengths, confidence, active_transitions,
            mask_token_id=mask_token_id, pad_token_id=pad, step=step, config=diffusion,
            transition_weight=dag.confidence_transition_weight, crf=crf,
            suffix_weights=None if predictions.suffix is None else predictions.suffix[rows], generator=generator)
        del active_probabilities, active_transitions
        stats.update_steps.index_add_(0, original_rows[rows], torch.ones_like(rows))

        if stop.mode == "trim_suffix":
            accepted_stop = torch.isin(updated, cast(torch.Tensor, stop_ids))
            accepted_stop &= updated.ne(mask_token_id) & updated.ne(pad)
            accepted_stop[:, 0] = accepted_stop[:, -1] = False
            if has_start:
                accepted_stop[:, 1] = False
            terminal[rows] = torch.where(accepted_stop, cast(torch.Tensor, position)[None], nodes).amin(-1)

        next_ids, next_edges = update_dag(
            updated, terminal_positions=terminal[rows],
            **cast(_DAGLayoutArgs, {**layout_args, "content_lengths": content_lengths[rows]}))
        dirty[rows] = (next_ids != current).any(-1) | (next_edges != edges[rows]).flatten(1).any(-1)
        ids[rows], edges[rows] = next_ids, next_edges

        progress.accept_paths(
            ids, rows, paths.nodes, path_lengths, pad_token_id=pad, mask_token_id=mask_token_id,
            original_rows=original_rows, block_index=block_index, has_start=has_start, stop_ids=stop_ids,
        )
        if bool(progress.done.all()):
            return cast(list[torch.Tensor], progress.contents), progress.stop_at

    failed = (~progress.done).nonzero(as_tuple=True)[0]
    raise RuntimeError(
        f"DADLM incomplete after {diffusion.num_steps} steps: "
        f"rows={original_rows[failed].tolist()}, block={block_index}, "
        f"remaining_masks={progress.remaining[failed].tolist()}")


def generate_dadlm(model: DADLMModel, prompt_ids: torch.Tensor, *, generation_length: int,
                   mask_token_id: int, block_bos_id: int, block_eos_id: int,
                   diffusion: DiffusionConfig, dag: DADLMGenerationConfig=DADLMGenerationConfig(),
                   start_token_id: int | None=None, sampling: SamplingConfig=SamplingConfig(),
                   stop: StopConfig=StopConfig(), generator: torch.Generator | None=None) -> GenerationResult:
    """Generate a target budget including the optional fixed start and kept EOS.

    Free graphs finish a full path before taking the remaining budget prefix.
    Only rows continuing to another graph re-encode and commit their content.
    Callers select task-specific starts, blocked IDs and stopping via options.
    """
    sampling, prompt_lengths, crf = _validate_request(
        model, prompt_ids, generation_length, start_token_id, mask_token_id,
        block_bos_id, block_eos_id, diffusion, dag, sampling, stop)
    batch = prompt_ids.shape[0]
    pad = model.config.pad_token_id
    stats = GenerationStats.empty(batch, prompt_ids.device)
    output = prompt_ids.new_full((batch, generation_length), pad)
    lengths = prompt_ids.new_zeros(batch)
    finished = torch.zeros(batch, device=prompt_ids.device, dtype=torch.bool)
    if start_token_id is not None and generation_length == 1:
        output[:, 0] = start_token_id
        return make_result(output, lengths + 1, finished, stats)

    with generation_context(model), progress_bar(desc="DADLM generation", unit="block") as progress:
        original_rows = torch.arange(batch, device=prompt_ids.device)
        session = model.new_cache_session(batch, dtype=cache_dtype(model))
        if bool(prompt_lengths.any()):
            prefill = model.prefill_prompt(prompt_ids, session)
            session.commit(cast(CacheDelta, prefill.cache_delta))
            del prefill
            stats.record("prefill", original_rows)
        block_index = 0
        while original_rows.numel():
            remaining = generation_length - lengths[original_rows]
            content_lengths = remaining.clamp_max(model.config.block_size)
            current = initialize_dag(
                original_rows.numel(), block_size=model.config.block_size,
                dag_up_ratio=model.config.dag_up_ratio, grouped=model.config.grouped,
                content_lengths=content_lengths, mask_token_id=mask_token_id,
                pad_token_id=pad, block_bos_id=block_bos_id, block_eos_id=block_eos_id,
                device=prompt_ids.device, start_token_id=start_token_id if block_index == 0 else None)
            contents, stops = _denoise_dag(
                model, current, content_lengths, session, original_rows=original_rows,
                block_index=block_index, mask_token_id=mask_token_id, diffusion=diffusion,
                dag=dag, sampling=sampling, stop=stop, has_start=block_index == 0 and start_token_id is not None,
                stats=stats, crf=crf, generator=generator)
            for local, (row, available) in enumerate(zip(original_rows.tolist(), remaining.tolist())):
                tokens = contents[local]
                stop_length = stops[local]
                count = min(available, tokens.numel(),
                            stop_length if stop_length is not None else tokens.numel())
                offset = generation_length - available
                output[row, offset:offset + count] = tokens[:count]
                lengths[row] += count
                finished[row] = stop_length is not None and stop_length <= count

            if progress is not None:
                progress.update()
            keep = ((lengths[original_rows] < generation_length) &
                    ~finished[original_rows]).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                break
            if keep.numel() != original_rows.numel():
                session.select_rows(keep)
            original_rows = original_rows[keep]
            continuing = [contents[i] for i in keep.tolist()]
            packed = prompt_ids.new_full((len(continuing), max(t.numel() for t in continuing)), pad)
            for row, tokens in enumerate(continuing):
                packed[row, :tokens.numel()] = tokens
            staged = model.stage_content(packed, session)
            session.commit(cast(CacheDelta, staged.cache_delta))
            del staged
            stats.record("commit", original_rows)
            block_index += 1
    return make_result(output, lengths, finished, stats)
