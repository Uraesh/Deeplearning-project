"""Runtime utility helpers for deterministic execution and memory cleanup."""

from __future__ import annotations

import gc
import random
from typing import Any, cast

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    cast(Any, torch).manual_seed(int(seed))


def configure_runtime(num_threads: int) -> None:
    """Configure PyTorch CPU thread usage for predictable resource consumption."""
    torch.set_num_threads(num_threads)
    interop_threads = max(1, min(4, num_threads))
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # This can fail when called too late in the process lifecycle.
        pass


def cleanup_memory() -> None:
    """Release Python and CUDA cached memory when available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
