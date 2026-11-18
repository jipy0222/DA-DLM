"""Single-writer token stores for fixed-length data rows.

Rows stream to an int64 file and are memory-mapped when read. Callers choose
the final store path and provide metadata used to check reuse.
"""

from __future__ import annotations

import fcntl
import json
import mmap
import os
import sys
import tempfile
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset

from data.types import Metadata, Pathish, TokenBatch

from .pretrain import PackingConfig, iter_token_windows
from .tokens import NativeTokenSchema

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


_FORMAT = "fixed-width-int64-le"
_DATA_FILE = "tokens.bin"
_MANIFEST = "manifest.json"
_COMPLETE = "_SUCCESS"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _require_storage_platform() -> None:
    if sys.byteorder != "little" or array("q").itemsize != 8:
        raise RuntimeError("the token format requires little-endian 64-bit integers")


def token_store_metadata(
    schema: NativeTokenSchema, config: PackingConfig, *, source: Mapping[str, Any],
    split: str, tokenizer_id: str,
) -> Metadata:
    """Describe a token store's corpus, tokenizer and packing settings."""
    return {
        "format": _FORMAT,
        "source": source,
        "split": split,
        "tokenizer": tokenizer_id,
        "special_ids": schema.to_dict(),
        "packing": config.to_metadata(),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_synced(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


class TokenDataset(Dataset[TokenBatch]):
    """A map-style, process-safe view of immutable full-token windows."""

    def __init__(self, path: Pathish, *, expected_metadata: Mapping[str, Any] | None = None) -> None:
        _require_storage_platform()
        self.path = Path(path).resolve()
        self._mapping = None
        self._mapping_pid = None
        try:
            raw = (self.path / _MANIFEST).read_bytes()
            if not (self.path / _COMPLETE).is_file():
                raise ValueError("completion marker is missing")
            manifest = json.loads(raw)
            metadata = manifest["metadata"]
            if metadata["format"] != _FORMAT or manifest["data_file"] != _DATA_FILE:
                raise ValueError("unsupported token store format")
            if expected_metadata is not None and metadata != expected_metadata:
                raise ValueError("token store metadata mismatch")
            rows = manifest["num_rows"]
            width = manifest["sequence_length"]
            if (type(rows) is not int or rows < 0
                    or type(width) is not int or width < 1
                    or width != metadata["packing"]["sequence_length"]):
                raise ValueError("invalid token store shape")
            if (self.path / _DATA_FILE).stat().st_size != rows * width * 8:
                raise ValueError("token data size does not match manifest")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("invalid or incomplete token store at %s: %s" % (self.path, exc)) from exc
        self.manifest = manifest
        self.metadata = metadata
        self.dataset_identity = _json_bytes({
            "metadata": metadata,
        }).decode("utf-8")
        self.sequence_length = width
        self._num_rows = rows

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int) -> TokenBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._mapping is None or self._mapping_pid != os.getpid():
            self.close()
            with (self.path / _DATA_FILE).open("rb") as stream:
                # Private writable pages avoid torch's read-only-buffer warning;
                # clone below prevents user tensors from modifying this mapping.
                self._mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
            self._mapping_pid = os.getpid()
        tokens = torch.frombuffer(
            self._mapping, dtype=torch.int64, count=self.sequence_length,
            offset=index * self.sequence_length * 8,
        ).clone()
        return {"input_ids": tokens}

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
            self._mapping_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_mapping"] = None
        state["_mapping_pid"] = None
        return state

    def __del__(self) -> None:
        if getattr(self, "_mapping", None) is not None:
            self.close()


def build_token_store(
    texts: Iterable[str], tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema,
    config: PackingConfig, *, store_dir: Pathish, source: Mapping[str, Any],
    split: str, tokenizer_id: str,
) -> TokenDataset:
    """Reuse a token store or atomically publish a streaming, single-writer build.

    Reusing a store does not consume ``texts``. Existing invalid entries raise rather
    than being deleted or replaced. Lock files are intentionally retained:
    unlinking them could allow different builders to lock different inodes.
    ``store_dir`` is the final token directory.
    """
    metadata = token_store_metadata(
        schema, config, source=source, split=split, tokenizer_id=tokenizer_id,
    )
    _require_storage_platform()
    target = Path(store_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / (target.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            return TokenDataset(target, expected_metadata=metadata)
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.building-", dir=str(target.parent),
        ) as directory:
            temporary = Path(directory)
            count = 0
            with (temporary / _DATA_FILE).open("xb") as stream:
                for row in iter_token_windows(texts, tokenizer, schema, config):
                    data = array("q", row)
                    stream.write(data.tobytes())
                    count += 1
                stream.flush()
                os.fsync(stream.fileno())
            manifest = {
                "metadata": metadata, "data_file": _DATA_FILE,
                "num_rows": count, "sequence_length": config.sequence_length,
            }
            raw_manifest = _json_bytes(manifest)
            _write_synced(temporary / _MANIFEST, raw_manifest)
            _write_synced(temporary / _COMPLETE, b"")
            _fsync_directory(temporary)
            if target.exists():
                raise FileExistsError("token store destination appeared during build: %s" % target)
            os.rename(str(temporary), str(target))
            _fsync_directory(target.parent)
    return TokenDataset(target, expected_metadata=metadata)
