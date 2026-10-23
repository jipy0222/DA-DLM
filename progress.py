"""Scoped terminal progress; library calls are silent by default."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator, Iterable, Iterator, Protocol, Sized, TypeVar, cast

from tqdm.auto import tqdm as _tqdm

_Item = TypeVar("_Item")


class ProgressBar(Protocol):
    def update(self, n: float = 1) -> bool | None: ...


_enabled = ContextVar("progress_enabled", default=False)


@contextmanager
def progress_reporting(enabled: bool=True) -> Generator[None, None, None]:
    """Enable or disable progress in this execution context."""
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


@contextmanager
def progress_bar(*, desc: str, total: int | None=None, unit: str="it", leave: bool=False) -> Generator[ProgressBar | None, None, None]:
    """Open a throttled bar, or yield None when reporting is disabled."""
    if not _enabled.get():
        yield None
        return
    with _tqdm(total=total, desc=desc, unit=unit, leave=leave,
               mininterval=1.0, dynamic_ncols=True) as bar:
        yield cast(ProgressBar, bar)


def track(iterable: Iterable[_Item], *, desc: str, total: int | None=None, unit: str="it", leave: bool=False) -> Iterable[_Item]:
    """Track consumed work without wrapping the iterator when disabled."""
    if not _enabled.get():
        return iterable
    if total is None:
        try:
            total = len(cast(Sized, iterable))
        except TypeError:
            pass

    def iterate() -> Iterator[_Item]:
        with progress_bar(desc=desc, total=total, unit=unit, leave=leave) as bar:
            for item in iterable:
                yield item
                if bar is not None:
                    bar.update()

    return iterate()
