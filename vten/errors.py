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

    def __str__(self) -> str:
        msg = super().__str__()
        prefix_parts: list[str] = []
        if self.stage:
            prefix_parts.append(f"[{self.stage}]")
        if self.kernel_path:
            prefix_parts.append(f"({self.kernel_path})")
        if prefix_parts:
            return f"{' '.join(prefix_parts)} {msg}"
        return msg


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


class ProbeMismatchError(BackendError):
    """Raised when probe detects mismatch during simulation.

    Attributes:
        cmd_id: The command ID that detected the mismatch.
        beat_index: Beat index where first mismatch occurred.
        mismatches: List of mismatch detail dicts (cycle, beat, expected, actual).
    """

    def __init__(
        self,
        message: str = "",
        *,
        cmd_id: int = 0,
        beat_index: int = 0,
        mismatches: list[dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.cmd_id = cmd_id
        self.beat_index = beat_index
        self.mismatches = mismatches or []


# ── Verification errors ──


class VerificationError(VTenError):
    """Raised when HW output does not match golden reference.

    Attributes:
        tensor: Name of the tensor that failed verification.
        shape: Resolved shape of the tensor.
        max_diff: Maximum element-wise difference.
        max_lsb_err: Maximum integer-LSB error (int64 ``|hw - golden|``);
            ``0`` for floating-point comparisons.
    """

    def __init__(
        self,
        message: str = "",
        *,
        tensor: str = "",
        shape: tuple[int, ...] | None = None,
        max_diff: float = 0.0,
        max_lsb_err: int = 0,
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
        self.max_lsb_err = max_lsb_err
