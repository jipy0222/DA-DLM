"""Candidate distributions and token sampling for generation."""

from __future__ import annotations

from typing import cast

import torch

from .config import SamplingConfig


def candidate_probabilities(
    logits: torch.Tensor, config: SamplingConfig, *, ar: bool = False
) -> torch.Tensor:
    """Return normalized FP64 probabilities after output and top-k/top-p filters."""
    if logits.ndim < 2:
        raise ValueError("logits must have a vocabulary dimension and a batch dimension")
    vocab_size = logits.shape[-1]
    if vocab_size == 0 or any(token_id >= vocab_size for token_id in config.blocked_token_ids):
        raise ValueError("blocked token ID is outside the vocabulary")

    scores = logits.to(dtype=torch.float64, copy=True) if ar else logits.clone()
    if config.blocked_token_ids:
        scores[..., list(config.blocked_token_ids)] = -torch.inf
    if ar:
        probabilities = torch.softmax(scores, dim=-1)
    else:
        log_normalizer = torch.logsumexp(scores, dim=-1, keepdim=True)
        probabilities = (scores - log_normalizer).to(torch.float64).exp_()
        del log_normalizer
    del scores

    if config.top_k:
        indices = probabilities.topk(min(config.top_k, vocab_size), dim=-1).indices
        keep = torch.zeros_like(probabilities, dtype=torch.bool)
        keep.scatter_(-1, indices, True)
        probabilities = probabilities.masked_fill(~keep, 0)
    elif config.top_p < 1:
        sorted_probs, sorted_indices = probabilities.sort(dim=-1, descending=True)
        discard = sorted_probs.cumsum(dim=-1) > config.top_p
        discard[..., 1:] = discard[..., :-1].clone()
        discard[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(discard, 0)
        probabilities = torch.zeros_like(probabilities).scatter_(-1, sorted_indices, sorted_probs)

    mass = probabilities.sum(dim=-1, keepdim=True)
    if not (torch.isfinite(mass) & (mass > 0)).all():
        raise ValueError("each prediction must have at least one finite allowed logit")
    if ar and config.top_k == 0 and config.top_p == 1:
        # Softmax already normalizes an unfiltered AR distribution.
        return probabilities
    return probabilities / mass


def sample_candidates(
    probabilities: torch.Tensor,
    *,
    greedy: bool,
    generator: torch.Generator | None = None,
    ar: bool = False,
) -> torch.Tensor:
    """Choose tokens using Gumbel-max for AR or exponential racing for diffusion."""
    if greedy:
        return probabilities.argmax(dim=-1)

    uniform = torch.rand(
        probabilities.shape,
        dtype=probabilities.dtype,
        device=probabilities.device,
        generator=generator,
    )
    if ar:
        # A zero-probability category remains impossible after adding finite noise.
        uniform = uniform.clamp(min=torch.finfo(uniform.dtype).tiny, max=1 - torch.finfo(uniform.dtype).eps)
        gumbel = -torch.log(-torch.log(uniform))
        return (probabilities.log() + gumbel).argmax(dim=-1)

    exponential = 1e-10 - (uniform + 1e-10).log()
    return cast('torch.Tensor', (probabilities / exponential).argmax(dim=-1))
