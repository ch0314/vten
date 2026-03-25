"""vten.backend — Backend adapters.

Public API for backend selection and execution.
"""

from vten.backend.base import (
    Backend,
    BackendErrorCode,
    BackendResult,
    BatchResult,
    CmdStats,
    raise_backend_error,
)
from vten.backend.registry import (
    available_backends,
    get_backend,
    get_build_pipeline,
    resolve_backend_name,
)

__all__ = [
    "Backend",
    "BackendErrorCode",
    "BackendResult",
    "BatchResult",
    "CmdStats",
    "available_backends",
    "get_backend",
    "get_build_pipeline",
    "raise_backend_error",
    "resolve_backend_name",
]
