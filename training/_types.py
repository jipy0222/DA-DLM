"""Shared static contracts for training batches, models and saved EMA state.

Framework checkpoints and Hydra options remain open dictionaries because their
contents are owned by Lightning, configuration files and third-party plugins.
Our own checkpoint fields have fixed schemas below.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
)

from data.types import Pathish as Pathish
from data.types import TokenBatch
from models.types import NativeConfig as NativeConfig
from models.types import NativeModel as NativeModel

if TYPE_CHECKING:
    from lightning.pytorch import Callback, Trainer
    from torch import Tensor

    from objectives import LossTerms, MaskingSchedule
    from training.ar import ARTrainingModule
    from training.bd3lm import BD3LMTrainingModule
    from training.dadlm import DADLMTrainingModule
    from training.mdlm import MDLMTrainingModule


TrainingBatch = TokenBatch


TrainingModule = Union["ARTrainingModule", "MDLMTrainingModule", "BD3LMTrainingModule", "DADLMTrainingModule"]
WeightsKind = Literal["raw", "ema"]
Checkpoint = dict[str, Any]
ValidationResults = list[Mapping[str, float]]
PreparationResult = dict[str, Union[int, str]]
WorkflowResult = Union[PreparationResult, "Trainer", ValidationResults]


class TrainerCallbacks(Protocol):
    """Callbacks are populated by Lightning's callback connector at runtime."""

    callbacks: list[Callback]


class ParameterSpec(TypedDict):
    shape: tuple[int, ...]
    requires_grad: bool
    canonical: str


class EMAState(TypedDict):
    version: int
    decay: float
    num_updates: int | None
    manifest: dict[str, ParameterSpec]
    shadow_params: dict[str, Tensor]


class _FamilyContract(TypedDict, total=False):
    mask_token_id: int
    bos_token_id: int | None
    source_length: int
    diffusion: dict[str, Any]
    tokens: dict[str, int]
    assignment: str


class TrainingContract(_FamilyContract):
    blocked_token_ids: tuple[int, ...]
    optimizer: dict[str, Any]
    schedule: dict[str, Any]
    model: dict[str, Any]
    ema_decay: float | None


class TokenCheckpointState(TypedDict):
    version: int
    contract: TrainingContract
    ema: EMAState | None
    train_totals_epoch: int | None
    train_totals: Tensor | None


class PendingRetention(TypedDict):
    path: str
    score: Tensor


_Inputs = TypeVar("_Inputs", contravariant=True)


class DiffusionDraw(Protocol[_Inputs]):
    def __call__(self, inputs: _Inputs, /, *, noise: MaskingSchedule,
                 t_min: float, t_max: float, antithetic: bool) -> LossTerms: ...
