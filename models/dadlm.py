"""DAG diffusion adapter over the shared Transformer core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, cast

import torch
from torch import nn

from kv_cache import CacheMode, CacheSession
from models.backbone.types import AttentionLayout

from ._family import TokenFamilyModel, pad_source_boundary
from .backbone.config import (
    AttentionBackend,
    AttentionKind,
    ConditioningMode,
    TokenFamilyConfig,
    TransformerConfig,
    TransitionNormalization,
)
from .backbone.dag_link import DAGLink
from .backbone.layouts import (
    block_causal_prefill_layout,
    block_structured_layout,
    cached_generation_layout,
    positions_from_valid,
)
from .backbone.types import BackboneOutput, DADModelOutput, PositionPolicy


@dataclass(frozen=True)
class DADLMConfig:
    backbone: TransformerConfig
    vocab_size: int
    pad_token_id: int
    block_size: int
    dag_up_ratio: int
    grouped: bool
    conditioning: ConditioningMode = ConditioningMode.ZERO_TIME_COMPAT
    transition_normalization: TransitionNormalization = TransitionNormalization.LOCAL
    tie_word_embeddings: bool = False
    default_backend: AttentionBackend = AttentionBackend.FLEX
    timestep_embedding_size: int = 256
    # Total committed ordinary content, including the prompt. Preview nodes
    # spend temporary attention capacity, not this content budget. None uses
    # backbone.trained_context_length as the content limit; Core checks previews.
    max_generation_length: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, ConditioningMode):
            raise TypeError("conditioning must be a ConditioningMode")
        if not isinstance(self.default_backend, AttentionBackend):
            raise TypeError("default_backend must be an AttentionBackend")
        if not isinstance(self.transition_normalization, TransitionNormalization):
            raise TypeError(
                "transition_normalization must be a TransitionNormalization"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.dag_up_ratio < 2:
            raise ValueError("DADLM requires dag_up_ratio >= 2")
        if self.conditioning is ConditioningMode.NONE:
            raise ValueError("DADLM requires AdaLN conditioning")
        if self.max_generation_length is not None and self.max_generation_length <= 0:
            raise ValueError("max_generation_length must be positive")

    @property
    def nodes_per_block(self) -> int:
        return self.block_size * self.dag_up_ratio

    def _logical_position(self, raw_position: int) -> float:
        position = torch.tensor(raw_position, dtype=torch.long, device="cpu")
        policy = PositionPolicy(scale=1.0 / self.dag_up_ratio, grouped=self.grouped)
        return policy.logical_positions(position).item()

    def preview_capacity(self, history: int) -> tuple[int, float]:
        """Attention width and last logical position for one full DAG preview.

        ``history`` counts committed ordinary tokens, excluding physical PAD.
        Entry points validate their own history and content budgets.
        """
        return (history + self.nodes_per_block,
                self._logical_position(history * self.dag_up_ratio + self.nodes_per_block - 1))

    def with_lengths(
        self,
        *,
        context_length: int,
        current_length: Optional[int] = None,
        generation_length: Optional[int] = None,
    ) -> "DADLMConfig":
        """Derive training, temporary preview and committed-content capacities.

        context_length is clean source+target slot width (including PAD).
        current_length is the unexpanded target suffix, defaulting to the whole
        context. generation_length is total ordinary content INCLUDING prompt,
        and is not a DAG-node or max-new-token count. If omitted, it uses
        max_generation_length, then trained_context_length.

        A row with one remaining content slot may preview a FULL expanded block;
        the caller must restrict the final accepted content to its remaining
        budget. Full rows must be inactive (all PAD) or removed before preview.
        Free DADLM may commit variable-length completed blocks, not necessarily
        block_size tokens. Capacity overflow raises an error; history is retained.

        Attention capacity includes source-boundary PAD inserted before the
        target; it does not spend position/content budget. Only execution caps
        are replaced; trained_context_length and
        the original config are unchanged. Position derivation assumes default
        zero-based positions. Training widths size a shared capacity, not
        independent per-stream limits.
        """
        if current_length is None:
            current_length = context_length
        if generation_length is None:
            generation_length = self.max_generation_length
        if generation_length is None:
            generation_length = self.backbone.trained_context_length
        if not 0 < current_length <= context_length or generation_length <= 0:
            raise ValueError("lengths require 0 < current_length <= context_length and generation_length > 0")
        preview_attention, preview_position = self.preview_capacity(generation_length - 1)
        source_padding = (current_length - context_length) % self.block_size
        last_position = max(
            self._logical_position(context_length * self.dag_up_ratio - 1),
            preview_position,
        )
        return replace(
            self,
            backbone=replace(
                self.backbone,
                max_attention_length=max(
                    context_length + source_padding + current_length * self.dag_up_ratio,
                    preview_attention,
                ),
                max_position_id=last_position,
            ),
            max_generation_length=generation_length,
        )

    def shared(self) -> TokenFamilyConfig:
        return TokenFamilyConfig(
            backbone=self.backbone,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            conditioning=self.conditioning,
            default_backend=self.default_backend,
            tie_word_embeddings=self.tie_word_embeddings,
            timestep_embedding_size=self.timestep_embedding_size,
            max_generation_length=self.max_generation_length,
        )


class DADLMModel(TokenFamilyModel):
    def __init__(self, config: DADLMConfig) -> None:
        if config.max_generation_length is not None:
            attention, position = config.preview_capacity(config.max_generation_length - 1)
            if attention > config.backbone.max_attention_length:
                raise ValueError("max_generation_length must leave attention capacity for a full DAG preview")
            if position > config.backbone.max_position_id:
                raise ValueError("max_generation_length must leave RoPE positions for a full DAG preview")
        super().__init__(config.shared())
        self.config = config
        self.slot_embedding = nn.Embedding(
            config.nodes_per_block, config.backbone.hidden_size
        )
        self.dag_link = DAGLink(
            config.backbone.hidden_size,
            config.backbone.n_heads,
            config.nodes_per_block,
        )

    def _inputs(
        self,
        dag_ids: torch.Tensor,
        context_ids: torch.Tensor,
        dag_position_ids: Optional[torch.Tensor],
        context_position_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionLayout]:
        if dag_ids.ndim != 2 or context_ids.ndim != 2:
            raise ValueError("dag_ids and context_ids must have shape [batch, sequence]")
        if dag_ids.shape[0] != context_ids.shape[0]:
            raise ValueError("dag_ids and context_ids batch sizes must match")
        if dag_ids.shape[1] % self.config.dag_up_ratio:
            raise ValueError("DAG node count must be divisible by dag_up_ratio")
        current_token_count = dag_ids.shape[1] // self.config.dag_up_ratio
        if current_token_count > context_ids.shape[1]:
            raise ValueError("DAG token groups must align to a suffix of context")
        source_length = context_ids.shape[1] - current_token_count
        source_padding = (-source_length) % self.config.block_size
        padded_context_length = context_ids.shape[1] + source_padding
        total_length = dag_ids.shape[1] + padded_context_length
        if total_length > self.config.backbone.max_attention_length:
            raise ValueError(
                "DADLM DAG + context width after source-boundary padding (%d + %d = %d) exceeds max_attention_length=%d; "
                "configure the expanded capacity explicitly or use DADLMConfig.with_lengths()"
                % (dag_ids.shape[1], padded_context_length, total_length,
                   self.config.backbone.max_attention_length)
            )
        if context_position_ids is not None and context_position_ids.shape != context_ids.shape:
            raise ValueError("context_position_ids do not match context_ids")
        context_ids, context_position_ids = pad_source_boundary(
            context_ids, context_position_ids, source_length=source_length,
            padding=source_padding, pad_token_id=self.config.pad_token_id,
        )
        dag_valid = dag_ids.ne(self.config.pad_token_id)
        context_valid = context_ids.ne(self.config.pad_token_id)
        start = context_ids.shape[1] - current_token_count
        if context_position_ids is None:
            context_position_ids = positions_from_valid(
                context_valid, scale=self.config.dag_up_ratio
            )
        if dag_position_ids is None:
            base = context_position_ids[:, start:]
            offsets = torch.arange(
                self.config.dag_up_ratio, device=dag_ids.device
            )
            dag_position_ids = (
                base[:, :, None] + offsets[None, None, :]
            ).reshape_as(dag_ids)
        if dag_position_ids.shape != dag_ids.shape:
            raise ValueError("dag_position_ids do not match dag_ids")

        current_blocks = (
            start
            + torch.arange(dag_ids.shape[1], device=dag_ids.device)
            // self.config.dag_up_ratio
        ) // self.config.block_size
        context_blocks = torch.arange(
            context_ids.shape[1], device=dag_ids.device
        ) // self.config.block_size
        layout = block_structured_layout(
            current_valid=dag_valid,
            context_valid=context_valid,
            current_block_ids=current_blocks,
            context_block_ids=context_blocks,
        )
        return (
            torch.cat((dag_ids, context_ids), dim=1),
            torch.cat((dag_position_ids, context_position_ids), dim=1),
            layout,
        )

    def encode(
        self,
        dag_ids: torch.Tensor,
        context_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        *,
        dag_position_ids: Optional[torch.Tensor] = None,
        context_position_ids: Optional[torch.Tensor] = None,
        backend: Optional[AttentionBackend] = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        ids, positions, layout = self._inputs(
            dag_ids, context_ids, dag_position_ids, context_position_ids
        )
        hidden = self.token_embedding(ids)
        slot_ids = torch.arange(dag_ids.shape[1], device=dag_ids.device)
        slot_ids = slot_ids.remainder(self.config.nodes_per_block)
        hidden = torch.cat(
            (
                hidden[:, : dag_ids.shape[1]] + self.slot_embedding(slot_ids),
                hidden[:, dag_ids.shape[1] :],
            ),
            dim=1,
        )
        conditioning = self.conditioning(
            sigma, batch_size=ids.shape[0], device=ids.device
        )
        full = self.core(
            hidden,
            conditioning=conditioning,
            position_ids=positions,
            attention_layout=layout,
            backend=self.config.default_backend if backend is None else backend,
            position_policy=PositionPolicy(
                scale=1.0 / self.config.dag_up_ratio,
                grouped=self.config.grouped,
            ),
            output_hidden_states=output_hidden_states,
        )
        length = dag_ids.shape[1]
        hidden_states = None
        if full.hidden_states is not None:
            hidden_states = tuple(item[:, :length] for item in full.hidden_states)
        return BackboneOutput(
            full.last_hidden_state[:, :length],
            cache_delta=full.cache_delta,
            hidden_states=hidden_states,
        )

    def forward(
        self,
        dag_ids: torch.Tensor,
        context_ids: torch.Tensor,
        edge_valid: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        *,
        dag_position_ids: Optional[torch.Tensor] = None,
        context_position_ids: Optional[torch.Tensor] = None,
        backend: Optional[AttentionBackend] = None,
    ) -> DADModelOutput:
        if dag_ids.shape[1] % self.config.nodes_per_block:
            raise ValueError(
                "DAD training requires complete DAG blocks; construct/pad target nodes "
                "and edge_valid together before calling forward"
            )
        output = self.encode(
            dag_ids,
            context_ids,
            sigma,
            dag_position_ids=dag_position_ids,
            context_position_ids=context_position_ids,
            backend=backend,
        )
        return DADModelOutput(
            logits=self.vocabulary_head(output.last_hidden_state),
            links=self.dag_link(
                output.last_hidden_state,
                edge_valid,
                self.config.transition_normalization,
            ),
            cache_delta=output.cache_delta,
        )

    def prefill_prompt(
        self,
        prompt_ids: torch.Tensor,
        session: CacheSession,
        sigma: Optional[torch.Tensor] = None,
        *,
        block_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """Compute an external prompt delta for an empty cache, without DAG slots.

        Caller commits the returned delta. Default block IDs follow input slots;
        explicit per-sample IDs preserve other prompt layouts under padding.
        Ordinary BOS/summary_bos inside a target block belong in preview_dag.
        """

        if session.cache.width != 0:
            raise ValueError("prompt prefill requires an empty cache")
        valid = prompt_ids.ne(self.config.pad_token_id)
        request = self.cache_request(session, CacheMode.STAGE_APPEND, valid)
        if block_ids is None:
            block_ids = torch.arange(prompt_ids.shape[1], device=prompt_ids.device) // self.config.block_size
        return cast('BackboneOutput', self.core(
            self.token_embedding(prompt_ids),
            conditioning=self.conditioning(
                sigma, batch_size=prompt_ids.shape[0], device=prompt_ids.device
            ),
            position_ids=positions_from_valid(valid, scale=self.config.dag_up_ratio),
            attention_layout=block_causal_prefill_layout(valid=valid, block_ids=block_ids),
            backend=AttentionBackend.SDPA,
            position_policy=PositionPolicy(
                scale=1.0 / self.config.dag_up_ratio, grouped=self.config.grouped
            ),
            cache_request=request,
            output_hidden_states=output_hidden_states,
        ))

    def _encode_generation(
        self,
        current_ids: torch.Tensor,
        session: CacheSession,
        sigma: Optional[torch.Tensor],
        *,
        dag_nodes: bool,
    ) -> BackboneOutput:
        if dag_nodes and current_ids.shape[1] != self.config.nodes_per_block:
            raise ValueError("DAG preview must contain exactly one expanded block")
        valid = current_ids.ne(self.config.pad_token_id)
        # A preview needs one remaining content slot per active row, regardless
        # of its temporary DAG node count. Core still checks actual packed key
        # width and logical positions against the separate execution caps.
        budget_valid = (
            valid.any(dim=-1, keepdim=True)
            if dag_nodes
            else valid
        )
        request = self.cache_request(
            session,
            CacheMode.READ_ONLY if dag_nodes else CacheMode.STAGE_APPEND,
            budget_valid,
            packing_valid=valid,
        )
        cache = session.cache
        if dag_nodes:
            offsets = torch.arange(current_ids.shape[1], device=current_ids.device)
            positions = cache.lengths[:, None] * self.config.dag_up_ratio + offsets[None]
        else:
            positions = (
                cache.lengths[:, None] + valid.long().cumsum(dim=-1) - 1
            ).clamp_min(0) * self.config.dag_up_ratio
        layout = cached_generation_layout(
            request=request,
            causal_current=False,
            kind=AttentionKind.BLOCK_STRUCTURED_SELF_ATTENTION,
        )
        hidden = self.token_embedding(current_ids)
        if dag_nodes:
            slot_ids = torch.arange(current_ids.shape[1], device=current_ids.device)
            hidden = hidden + self.slot_embedding(slot_ids)
        conditioning = self.conditioning(
            sigma, batch_size=current_ids.shape[0], device=current_ids.device
        )
        return cast('BackboneOutput', self.core(
            hidden,
            conditioning=conditioning,
            position_ids=positions,
            attention_layout=layout,
            backend=AttentionBackend.SDPA,
            position_policy=PositionPolicy(
                scale=1.0 / self.config.dag_up_ratio,
                grouped=self.config.grouped,
            ),
            cache_request=request,
        ))

    def preview_dag(
        self,
        dag_ids: torch.Tensor,
        edge_valid: torch.Tensor,
        session: CacheSession,
        sigma: Optional[torch.Tensor] = None,
    ) -> DADModelOutput:
        output = self._encode_generation(
            dag_ids,
            session,
            sigma,
            dag_nodes=True,
        )
        return DADModelOutput(
            logits=self.vocabulary_head(output.last_hidden_state),
            links=self.dag_link(
                output.last_hidden_state,
                edge_valid,
                self.config.transition_normalization,
            ),
            cache_delta=None,
        )

    def stage_content(
        self,
        content_ids: torch.Tensor,
        session: CacheSession,
        sigma: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        """Stage final ordinary content from ONE completed generation block.

        Valid lengths may differ per sample (free DADLM). All current content
        is mutually visible; history is immutable. Graph sentinels/unselected
        nodes must already be removed by the caller. Use prefill_prompt for an
        external prompt, even when this call would also start with empty history.
        """

        return self._encode_generation(
            content_ids,
            session,
            sigma,
            dag_nodes=False,
        )
