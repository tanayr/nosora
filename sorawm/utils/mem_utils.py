# Some copy is inspired by https://github.com/vllm-project/vllm/blob/main/vllm/utils/mem_utils.py

# import contextlib
import gc

# import time
# from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cache

# import psutil
import torch

# import torch.types
from .mem_constants import GiB_bytes


@dataclass
class MemoryProfilingResult:
    # GB
    free_memory: float = 0.0
    total_memory: float = 0.0
    torch_memory: float = 0.0


def clear_gpu_memory():
    """
    Release and reset GPU memory state used by CUDA and PyTorch.

    Forces Python garbage collection, clears PyTorch's CUDA memory cache, resets CUDA peak-memory counters, and synchronizes the CUDA device so freed memory is available for subsequent operations.
    For non-CUDA devices (CPU/MPS), only garbage collection is performed.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        # MPS doesn't have the same memory management APIs
        # Just trigger garbage collection
        pass


def memory_profiling() -> MemoryProfilingResult:
    """
    Capture current device memory metrics and return them in gibibytes.

    Returns:
        MemoryProfilingResult: Dataclass containing:
            - free_memory (float): Available device memory in GiB.
            - total_memory (float): Total device memory in GiB.
            - torch_memory (float): Memory reserved by PyTorch in GiB.
    """
    clear_gpu_memory()
    
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info()
        torch_memory = torch.cuda.memory_reserved()
        result = MemoryProfilingResult(
            free_memory=free_memory / GiB_bytes,
            total_memory=total_memory / GiB_bytes,
            torch_memory=torch_memory / GiB_bytes,
        )
    else:
        # For CPU or MPS, return reasonable defaults
        # Use system memory info as a fallback
        import psutil
        mem = psutil.virtual_memory()
        result = MemoryProfilingResult(
            free_memory=mem.available / GiB_bytes,
            total_memory=mem.total / GiB_bytes,
            torch_memory=0.0,
        )
    return result

    # result = MemoryProfilingResult()

    # result.before_create = baseline_snapshot
    # # the part of memory used for holding the model weights
    # result.weights_memory = weights_memory

    # result.before_profile.measure()

    # yield result

    # gc.collect()
    # torch.cuda.empty_cache()

    # result.after_profile.measure()

    # diff_profile = result.after_profile - result.before_profile
    # diff_from_create = result.after_profile - result.before_create
    # result.torch_peak_increase = diff_profile.torch_peak
    # result.non_torch_increase = diff_from_create.non_torch_memory
    # result.profile_time = diff_profile.timestamp

    # non_torch_memory = result.non_torch_increase
    # peak_activation_memory = result.torch_peak_increase
    # result.non_kv_cache_memory = (
    #     non_torch_memory + peak_activation_memory + result.weights_memory
    # )  # noqa
