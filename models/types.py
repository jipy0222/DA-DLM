"""Model families and their configuration types, without eager model imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Union

if TYPE_CHECKING:
    from torch import Tensor

    from .ar import ARConfig, ARModel
    from .bd3lm import BD3LMConfig, BD3LMModel
    from .dadlm import DADLMConfig, DADLMModel
    from .mdlm import MDLMConfig, MDLMModel


NativeModel = Union["ARModel", "MDLMModel", "BD3LMModel", "DADLMModel"]
NativeConfig = Union["ARConfig", "MDLMConfig", "BD3LMConfig", "DADLMConfig"]
DiffusionModel = Union["MDLMModel", "BD3LMModel"]

TensorState = Mapping[str, "Tensor"]
