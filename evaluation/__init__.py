"""Thin, task-level assembly for native token generation evaluation."""

from .generation import (
    decode_prompt_and_target,
    decode_target_body,
    generate_batch_records,
    make_summary_options,
    make_text_options,
)

__all__ = [
    "decode_prompt_and_target",
    "decode_target_body",
    "generate_batch_records",
    "make_summary_options",
    "make_text_options",
]
