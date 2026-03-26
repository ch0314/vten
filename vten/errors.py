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


# ── Build errors (CLI build pipeline) ──


class BuildError(VTenError):
    """Build pipeline error (codegen, compile, etc.)."""

    pass


# ── Validation errors (Stage 0, build time) ──


class ValidationError(VTenError):
    pass


class ProtocolMismatchError(ValidationError):
    pass


class BankOverlapError(ValidationError):
    pass


class ConnectionShapeMismatchError(ValidationError):
    pass


class ConnectionDtypeMismatchError(ValidationError):
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
    """Raised when HW output does not match golden reference.

    Attributes:
        tensor: Name of the tensor that failed verification.
        shape: Resolved shape of the tensor.
        max_diff: Maximum element-wise difference.
    """

    def __init__(
        self,
        message: str = "",
        *,
        tensor: str = "",
        shape: tuple[int, ...] | None = None,
        max_diff: float = 0.0,
        **kwargs,
    ) -> None:
        if not message and tensor:
            message = (
                f"Verification failed for tensor '{tensor}': "
                f"shape={shape}, max_diff={max_diff}"
            )
        super().__init__(message, **kwargs)
        self.tensor = tensor
        self.shape = shape
        self.max_diff = max_diff
