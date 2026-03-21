"""
Configure BLAS/OpenMP before importing torch (macOS / multi-lib crash mitigation).

Import this module before ``import torch`` whenever possible:
    import core.torch_env  # noqa: F401
    import torch
    core.torch_env.limit_pytorch_threads(torch)
"""

from __future__ import annotations

import os


def apply_thread_env() -> None:
    for key, val in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ):
        os.environ.setdefault(key, val)


def limit_pytorch_threads(torch) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


apply_thread_env()
