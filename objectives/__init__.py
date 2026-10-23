"""Task-independent objective math; callers own data and training orchestration."""

from .ar import ar_objective, prepare_ar_batch
from .mdlm import mask_tokens, mdlm_objective, prepare_mdlm_batch
from .noise import MaskingSchedule, sample_block_time, sample_mdlm_time
from .types import LossTerms

__all__ = ["LossTerms", "ar_objective", "prepare_ar_batch", "MaskingSchedule",
           "sample_mdlm_time", "sample_block_time", "mask_tokens", "mdlm_objective",
           "prepare_mdlm_batch"]
