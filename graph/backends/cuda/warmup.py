"""Optional pre-first-batch warm-up for the lazy CUDA graph extension."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

from ._extension import (
  cuda_extension_is_loaded,
  load_cuda_extension,
  visible_cuda_architectures,
)


@dataclass(frozen=True)
class CudaWarmupResult:
  """Observable result of loading or compiling the graph extension."""

  module_name: str
  architectures: tuple[str, ...]
  library_path: str
  elapsed_seconds: float
  already_loaded_in_process: bool


def warmup_cuda() -> CudaWarmupResult:
  """Compile/load the shared CUDA graph extension."""
  architectures = visible_cuda_architectures()
  already_loaded = cuda_extension_is_loaded()
  start = time.perf_counter()
  extension = load_cuda_extension()
  elapsed = time.perf_counter() - start
  return CudaWarmupResult(
    module_name=extension.__name__,
    architectures=architectures,
    library_path=str(extension.__file__),
    elapsed_seconds=elapsed,
    already_loaded_in_process=already_loaded,
  )


def main() -> None:
  """Warm the JIT cache from an allocated CUDA worker."""
  parser = argparse.ArgumentParser(
    description=(
      "JIT-build/load all DADLM graph CUDA kernels for visible GPUs; no GPU "
      "architecture argument is required"
    )
  )
  parser.add_argument(
    "--json", action="store_true", help="emit a machine-readable result")
  arguments = parser.parse_args()
  result = warmup_cuda()
  if arguments.json:
    print(json.dumps(asdict(result), sort_keys=True))
  else:
    architecture_text = ", ".join(result.architectures)
    print(
      f"loaded {result.module_name} for {architecture_text} from "
      f"{result.library_path} in {result.elapsed_seconds:.3f}s"
    )


if __name__ == "__main__":
  main()


__all__ = ["CudaWarmupResult", "warmup_cuda"]
