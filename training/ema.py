"""Exponential moving averages of named trainable parameters.

Call ``update`` once after each actual optimizer update. The caller supplies
the parameter collection and manages weight selection and checkpoint hooks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from numbers import Real
from typing import Any, Dict, Generator, Optional, Tuple, Union, cast

import torch
from torch import nn

from training._types import EMAState, ParameterSpec

NamedParameters = Union[nn.Module, Iterable[Tuple[str, nn.Parameter]]]
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _validate_decay(decay: float) -> float:
    if isinstance(decay, bool) or not isinstance(decay, Real):
        raise TypeError("decay must be a finite real number")
    value = float(decay)
    if not 0.0 <= value <= 1.0:
        raise ValueError("decay must be finite and between 0 and 1")
    return value


def _validate_dtype(dtype: Optional[torch.dtype]) -> None:
    if dtype is not None and dtype not in _FLOAT_DTYPES:
        raise TypeError("EMA dtype must be float16, bfloat16, float32, or float64")


def _collect(parameters: NamedParameters) -> tuple[dict[str, nn.Parameter], dict[str, ParameterSpec]]:
    pairs = (parameters.named_parameters(remove_duplicate=False)
             if isinstance(parameters, nn.Module) else parameters)
    named: Dict[str, nn.Parameter] = {}
    for name, parameter in pairs:
        if not isinstance(name, str) or not name:
            raise TypeError("parameter names must be nonempty strings")
        if name in named:
            raise ValueError("duplicate parameter name: %s" % name)
        if not isinstance(parameter, nn.Parameter):
            raise TypeError("%s is not an nn.Parameter" % name)
        if (parameter.layout != torch.strided
                or (parameter.requires_grad and parameter.dtype not in _FLOAT_DTYPES)):
            raise TypeError("%s must be dense and, if trainable, floating-point" % name)
        if parameter.device.type == "meta":
            raise ValueError("materialize meta parameters before creating or using EMA")
        named[name] = parameter
    names = sorted(named)
    aliases: Dict[int, list[str]] = {}
    for name in names:
        aliases.setdefault(id(named[name]), []).append(name)
    canonical = {name: group[0] for group in aliases.values() for name in group}
    # Track frozen parameters too, so changes in trainability are detected.
    manifest: dict[str, ParameterSpec] = {
        name: {"shape": tuple(named[name].shape),
               "requires_grad": named[name].requires_grad,
               "canonical": canonical[name]}
        for name in names
    }
    unique = {name: named[name] for name in names if canonical[name] == name}
    if not any(parameter.requires_grad for parameter in unique.values()):
        raise ValueError("EMA requires at least one trainable parameter")
    return unique, manifest


@torch.no_grad()
def _replace_parameter_values(parameters: Mapping[str, nn.Parameter],
                              values: Mapping[str, torch.Tensor]) -> None:
    """Copy live parameter values and invalidate cached autocast casts."""
    try:
        for name, value in values.items():
            parameters[name].copy_(value)
    finally:
        torch.clear_autocast_cache()


class ExponentialMovingAverage:
    """Maintain one shadow per initially trainable, distinct Parameter.

    ``parameters`` is a Module or an iterable of ``(name, Parameter)`` pairs.
    Iterable order is irrelevant. Include all parameters, including frozen
    ones, and expose tied aliases consistently on every call. When passing a
    Module, all aliases are discovered automatically. Its fixed manifest must
    retain names, shapes, trainability and tie relationships; changing any of
    these requires constructing a new EMA.

    Shadows initially preserve parameter device and dtype. ``shadow_dtype``
    optionally chooses FP32 (or another supported floating dtype). ``to`` can
    place the shadows explicitly; updates cast detached live values to that
    placement. Loading state preserves the receiving EMA's placement/dtype.
    Neither shadow placement nor live dtype/device is part of the manifest.

    ``copy_to`` permanently selects EMA weights. ``average_parameters`` swaps
    parameter values temporarily and restores them, including frozen parameters,
    on exit without replacing Parameters.
    Buffers are not averaged or restored. Do not mutate model structure or
    move/cast the live model while its temporary EMA context is active.
    """

    STATE_VERSION = 1

    def __init__(self, parameters: NamedParameters, decay: float,
                 use_num_updates: bool = True,
                 shadow_dtype: Optional[torch.dtype] = None) -> None:
        self._decay = _validate_decay(decay)
        if type(use_num_updates) is not bool:
            raise TypeError("use_num_updates must be bool")
        _validate_dtype(shadow_dtype)
        unique, self._manifest = _collect(parameters)
        self._shadows: dict[str, torch.Tensor] = {
            name: parameter.detach().to(dtype=shadow_dtype or parameter.dtype).clone()
            for name, parameter in unique.items() if parameter.requires_grad
        }
        self._num_updates: int | None = 0 if use_num_updates else None
        self._active = False

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def num_updates(self) -> Optional[int]:
        return self._num_updates

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """Canonical names of the tracked, trainable parameters."""
        return tuple(self._shadows)

    def _require_inactive(self) -> None:
        if self._active:
            raise RuntimeError("EMA mutation or nesting is forbidden during average_parameters")

    def _validate_parameters(self, parameters: NamedParameters) -> dict[str, nn.Parameter]:
        unique, manifest = _collect(parameters)
        if manifest != self._manifest:
            raise ValueError("EMA parameter manifest changed (names, shapes, trainability or tying)")
        return unique

    @torch.no_grad()
    def update(self, parameters: NamedParameters) -> None:
        """Update shadows once, applying count-based decay warmup when enabled."""
        self._require_inactive()
        unique = self._validate_parameters(parameters)
        # Prepare all live values before modifying shadows.
        current = {name: unique[name].detach().to(device=shadow.device, dtype=shadow.dtype)
                   for name, shadow in self._shadows.items()}
        count = None if self._num_updates is None else self._num_updates + 1
        decay = self._decay if count is None else min(self._decay, (1 + count) / (10 + count))
        for name, shadow in self._shadows.items():
            shadow.sub_((1.0 - decay) * (shadow - current[name]))
        self._num_updates = count

    def _copy_values(self, unique: Mapping[str, nn.Parameter]) -> dict[str, torch.Tensor]:
        return {name: shadow.to(device=unique[name].device, dtype=unique[name].dtype)
                for name, shadow in self._shadows.items()}

    @torch.no_grad()
    def copy_to(self, parameters: NamedParameters) -> None:
        """Permanently copy EMA values to trainable parameters only."""
        self._require_inactive()
        unique = self._validate_parameters(parameters)
        values = self._copy_values(unique)
        _replace_parameter_values(unique, values)

    @contextmanager
    def average_parameters(self, parameters: NamedParameters) -> Generator[None, None, None]:
        """Use EMA values temporarily and restore live values in ``finally``.

        This is an exclusive context: nested use, update, load, placement
        changes and permanent copy are rejected while it is active. Existing
        gradients, Parameter identities and model train/eval mode are untouched.
        """
        self._require_inactive()
        unique = self._validate_parameters(parameters)
        backup = {name: parameter.detach().clone() for name, parameter in unique.items()}
        values = self._copy_values(unique)
        self._active = True
        try:
            _replace_parameter_values(unique, values)
            yield
        finally:
            try:
                _replace_parameter_values(unique, backup)
            finally:
                self._active = False

    def to(self, device: torch.device | str | int | None = None,
           dtype: Optional[torch.dtype] = None) -> ExponentialMovingAverage:
        """Move/cast shadows only; return self."""
        self._require_inactive()
        _validate_dtype(dtype)
        converted = {name: value.to(device=device, dtype=dtype or value.dtype)
                     for name, value in self._shadows.items()}
        if any(value.device.type == "meta" for value in converted.values()):
            raise ValueError("EMA shadows must remain materialized")
        self._shadows = converted
        return self

    @torch.no_grad()
    def synchronize(self, src: int = 0) -> None:
        """Broadcast EMA values and update state in an initialized process group.

        Ranks must use matching parameter manifests and backend-compatible
        shadow devices. Call after model synchronization, before using EMA.
        """
        self._require_inactive()
        metadata: list[tuple[float, int | None]] = [(self._decay, self._num_updates)]
        device = next(iter(self._shadows.values())).device
        torch.distributed.broadcast_object_list(metadata, src=src, device=device)
        for shadow in self._shadows.values():
            torch.distributed.broadcast(shadow, src=src)
        self._decay, self._num_updates = metadata[0]

    def state_dict(self) -> EMAState:
        """Return an independent, versioned snapshot of the EMA state."""
        return {
            "version": self.STATE_VERSION,
            "decay": self._decay,
            "num_updates": self._num_updates,
            "manifest": {name: cast(ParameterSpec, dict(spec)) for name, spec in self._manifest.items()},
            "shadow_params": {name: value.detach().clone() for name, value in self._shadows.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load a named EMA state after validation and tensor conversion.

        Shape, names, trainability and tying must match. Source tensors are
        cloned and cast to this EMA's shadow dtype/device before replacing state.
        """
        self._require_inactive()
        keys = {"version", "decay", "num_updates", "manifest", "shadow_params"}
        if not isinstance(state, Mapping) or set(state) != keys:
            raise ValueError("invalid EMA state fields")
        if type(state["version"]) is not int or state["version"] != self.STATE_VERSION:
            raise ValueError("unsupported EMA state version")
        decay = _validate_decay(state["decay"])
        count = state["num_updates"]
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("num_updates must be None or a nonnegative integer")
        manifest = state["manifest"]
        if not isinstance(manifest, Mapping) or set(manifest) != set(self._manifest):
            raise ValueError("EMA state parameter manifest does not match")
        for name, expected in self._manifest.items():
            actual = manifest[name]
            if not isinstance(actual, Mapping) or set(actual) != set(expected):
                raise ValueError("invalid manifest entry for %s" % name)
            shape = actual["shape"]
            if (not isinstance(shape, (tuple, list))
                    or any(type(size) is not int for size in shape)
                    or tuple(shape) != expected["shape"]
                    or type(actual["requires_grad"]) is not bool
                    or actual["requires_grad"] != expected["requires_grad"]
                    or not isinstance(actual["canonical"], str)
                    or actual["canonical"] != expected["canonical"]):
                raise ValueError("EMA state manifest differs for %s" % name)
        shadows = state["shadow_params"]
        if not isinstance(shadows, Mapping) or set(shadows) != set(self._shadows):
            raise ValueError("EMA state shadow names do not match")
        for name, destination in self._shadows.items():
            source = shadows[name]
            if (not isinstance(source, torch.Tensor)
                    or source.dtype not in _FLOAT_DTYPES
                    or source.layout != torch.strided
                    or source.device.type == "meta"
                    or source.shape != destination.shape):
                raise ValueError("invalid EMA shadow shape, dtype or storage for %s" % name)
        converted = {
            name: shadows[name].detach().to(device=destination.device, dtype=destination.dtype).clone()
            for name, destination in self._shadows.items()
        }
        self._decay, self._num_updates, self._shadows = decay, count, converted
