"""Hydra entry point for preparation, training, validation and generation evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import hydra
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from training._types import WorkflowResult


_MODES: Final = frozenset({"bootstrap", "prepare", "pretrain", "validate",
                         "prepare_summary", "finetune", "validate_summary",
                         "evaluate_summary", "evaluate_text", "score_text"})
_GRAPH_BACKENDS: Final = frozenset({"auto", "pytorch", "cuda"})


def _mapping(config: DictConfig, name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a configuration mapping")
    return value


def _validate_graph(config: DictConfig) -> None:
    graph = _mapping(config, "graph")
    if graph.get("backend") not in _GRAPH_BACKENDS:
        raise ValueError("graph.backend must be one of auto, cuda, pytorch")
    cuda = graph.get("cuda")
    if not isinstance(cuda, Mapping):
        raise TypeError("graph.cuda must be a configuration mapping")
    if not isinstance(cuda.get("warmup_before_training"), bool):
        raise TypeError("graph.cuda.warmup_before_training must be a bool")


def validate_config(config: DictConfig) -> None:
    """Validate root dispatch, seed and graph-runtime choices.

    Each workflow validates its task inputs; typed constructors validate model,
    objective and training settings.
    """
    mode = config.get("mode")
    if mode not in _MODES:
        raise ValueError(
            f"Mode {mode!r} is unavailable from the root CLI; choose "
            "bootstrap, prepare, pretrain, validate, prepare_summary, finetune, "
            "validate_summary, evaluate_summary, evaluate_text or score_text"
        )
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _validate_graph(config)


def run(config: DictConfig) -> WorkflowResult | dict[str, Any] | None:
    """Run a safe bootstrap or dispatch an explicitly selected workflow."""
    validate_config(config)
    if config.mode == "bootstrap":
        print("DA-DLM bootstrap configuration is valid; no training was started.")
        print(OmegaConf.to_yaml(config, resolve=True))
        return None

    # Bootstrap remains importable without Lightning/data/model dependencies.
    if config.mode == "evaluate_summary":
        from evaluation.summary_workflow import evaluate_summary
        from progress import progress_reporting
        with progress_reporting(config.evaluation.progress):
            return evaluate_summary(config)
    if config.mode in {"evaluate_text", "score_text"}:
        from evaluation.text_workflow import evaluate_text, score_text
        from progress import progress_reporting
        with progress_reporting(config.evaluation.progress):
            return evaluate_text(config) if config.mode == "evaluate_text" else score_text(config)
    if config.mode in {"prepare_summary", "finetune", "validate_summary"}:
        from training.summary_workflow import run as run_workflow
    else:
        from training.workflow import run as run_workflow
    return run_workflow(config)


# Hydra supplies the optional CLI/config passthrough wrapper dynamically.
@hydra.main(version_base="1.3", config_path="configs", config_name="config")  # ty: ignore[dynamic-function-decorator-return]
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
