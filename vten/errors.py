"""VTenError hierarchy.

Spec reference: 00_data_models.md §11
"""

from __future__ import annotations


class VTenError(Exception):
    """Base error for all vTen exceptions."""

    def __init__(
        self,
        message: str,
        kernel_path: str = "",
        stage: str = "",
        context: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.kernel_path = kernel_path
        self.stage = stage
        self.context = context or {}


# ── Validation errors (Stage 0, build time) ──


class ValidationError(VTenError):
    pass


class ProtocolMismatchError(ValidationError):
    pass


class BankOverlapError(ValidationError):
    pass


class ConnectionShapeMismatchError(ValidationError):
    pass


class SpecValidationError(ValidationError):
    """Spec parsing / validation error."""

    pass


# ── Compilation errors (Stages 1-7) ──


class CompilationError(VTenError):
    pass


class ParameterResolutionError(CompilationError):
    pass


class ShapeMismatchError(CompilationError):
    pass


class SerializationError(CompilationError):
    pass


class MemoryOverflowError(CompilationError):
    pass


class BindingError(CompilationError):
    pass


class DependencyError(CompilationError):
    pass


class DependencyLimitError(CompilationError):
    pass


class ProbeError(CompilationError):
    pass


class AliasError(CompilationError):
    pass


# ── Backend errors (Execution time) ──


class BackendError(VTenError):
    pass


class TimeoutError(BackendError):
    pass


class BFMError(BackendError):
    pass


class PollTimeoutError(BackendError):
    pass


# ── Verification errors ──


class VerificationError(VTenError):
    pass
