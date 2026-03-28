"""Tests for VTenError hierarchy.

Spec reference: 00_data_models.md §11

Tests cover:
- Error class hierarchy (inheritance chain)
- VTenError constructor attributes (message, kernel_path, stage, context)
- All 22 error classes instantiation
- Context dict preservation
- Error code → exception mapping (backend)
"""

from __future__ import annotations

import pytest

from vten.errors import (
    AliasError,
    BackendError,
    BFMError,
    BindingError,
    BuildError,
    CompilationError,
    ConnectionShapeMismatchError,
    DependencyError,
    DependencyLimitError,
    MemoryOverflowError,
    ParameterResolutionError,
    PollTimeoutError,
    ProbeError,
    ProtocolMismatchError,
    SerializationError,
    ShapeMismatchError,
    SpecValidationError,
    BankOverlapError,
    ValidationError,
    VerificationError,
    VTenError,
)
from vten.errors import TimeoutError as VTenTimeoutError


# ═══════════════════════════════════════════════════════════════════
# §1  VTenError base class — 00_data_models.md §11
# ═══════════════════════════════════════════════════════════════════


class TestVTenErrorBase:
    """VTenError: base exception with structured attributes."""

    def test_inherits_from_exception(self):
        assert issubclass(VTenError, Exception)

    def test_message_stored(self):
        e = VTenError("something failed")
        assert str(e) == "something failed"

    def test_default_kernel_path_empty(self):
        e = VTenError("err")
        assert e.kernel_path == ""

    def test_default_stage_empty(self):
        e = VTenError("err")
        assert e.stage == ""

    def test_default_context_empty_dict(self):
        e = VTenError("err")
        assert e.context == {}

    def test_kernel_path_set(self):
        e = VTenError("err", kernel_path="kernels/conv3d")
        assert e.kernel_path == "kernels/conv3d"

    def test_stage_set(self):
        e = VTenError("err", stage="stage_4_address")
        assert e.stage == "stage_4_address"

    def test_context_dict_preserved(self):
        ctx = {"tensor": "data_in", "expected": 1024, "actual": 512}
        e = VTenError("shape mismatch", context=ctx)
        assert e.context == ctx
        assert e.context["tensor"] == "data_in"

    def test_context_none_becomes_empty_dict(self):
        e = VTenError("err", context=None)
        assert e.context == {}

    def test_all_kwargs(self):
        e = VTenError(
            "full error",
            kernel_path="kernels/npu",
            stage="stage_6_ir",
            context={"cmd_id": 5},
        )
        assert str(e) == "[stage_6_ir] (kernels/npu) full error"
        assert e.kernel_path == "kernels/npu"
        assert e.stage == "stage_6_ir"
        assert e.context["cmd_id"] == 5

    def test_catchable_as_exception(self):
        with pytest.raises(Exception):
            raise VTenError("test")


# ═══════════════════════════════════════════════════════════════════
# §2  Hierarchy — inheritance chain
# ═══════════════════════════════════════════════════════════════════


class TestErrorHierarchy:
    """All error classes inherit correctly per 00_data_models.md §11."""

    # ── Build ──

    def test_build_error_inherits_vten(self):
        assert issubclass(BuildError, VTenError)

    # ── Validation ──

    def test_validation_error_inherits_vten(self):
        assert issubclass(ValidationError, VTenError)

    def test_protocol_mismatch_inherits_validation(self):
        assert issubclass(ProtocolMismatchError, ValidationError)

    def test_bank_overlap_inherits_validation(self):
        assert issubclass(BankOverlapError, ValidationError)

    def test_connection_shape_mismatch_inherits_validation(self):
        assert issubclass(ConnectionShapeMismatchError, ValidationError)

    def test_spec_validation_inherits_validation(self):
        assert issubclass(SpecValidationError, ValidationError)

    # ── Compilation ──

    def test_compilation_error_inherits_vten(self):
        assert issubclass(CompilationError, VTenError)

    def test_parameter_resolution_inherits_compilation(self):
        assert issubclass(ParameterResolutionError, CompilationError)

    def test_shape_mismatch_inherits_compilation(self):
        assert issubclass(ShapeMismatchError, CompilationError)

    def test_serialization_inherits_compilation(self):
        assert issubclass(SerializationError, CompilationError)

    def test_memory_overflow_inherits_compilation(self):
        assert issubclass(MemoryOverflowError, CompilationError)

    def test_binding_inherits_compilation(self):
        assert issubclass(BindingError, CompilationError)

    def test_dependency_inherits_compilation(self):
        assert issubclass(DependencyError, CompilationError)

    def test_dependency_limit_inherits_compilation(self):
        assert issubclass(DependencyLimitError, CompilationError)

    def test_probe_inherits_compilation(self):
        assert issubclass(ProbeError, CompilationError)

    def test_alias_inherits_compilation(self):
        assert issubclass(AliasError, CompilationError)

    # ── Backend ──

    def test_backend_error_inherits_vten(self):
        assert issubclass(BackendError, VTenError)

    def test_timeout_inherits_backend(self):
        assert issubclass(VTenTimeoutError, BackendError)

    def test_bfm_error_inherits_backend(self):
        assert issubclass(BFMError, BackendError)

    def test_poll_timeout_inherits_backend(self):
        assert issubclass(PollTimeoutError, BackendError)

    # ── Verification ──

    def test_verification_inherits_vten(self):
        assert issubclass(VerificationError, VTenError)


# ═══════════════════════════════════════════════════════════════════
# §3  All classes carry VTenError attributes
# ═══════════════════════════════════════════════════════════════════


class TestErrorAttributes:
    """All subclasses inherit kernel_path, stage, context from VTenError."""

    ERROR_CLASSES = [
        BuildError,
        ValidationError,
        ProtocolMismatchError,
        BankOverlapError,
        ConnectionShapeMismatchError,
        SpecValidationError,
        CompilationError,
        ParameterResolutionError,
        ShapeMismatchError,
        SerializationError,
        MemoryOverflowError,
        BindingError,
        DependencyError,
        DependencyLimitError,
        ProbeError,
        AliasError,
        BackendError,
        VTenTimeoutError,
        BFMError,
        PollTimeoutError,
        VerificationError,
    ]

    @pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
    def test_subclass_accepts_all_kwargs(self, cls):
        e = cls(
            "test message",
            kernel_path="kernels/test",
            stage="stage_1",
            context={"key": "val"},
        )
        assert str(e) == "[stage_1] (kernels/test) test message"
        assert e.kernel_path == "kernels/test"
        assert e.stage == "stage_1"
        assert e.context == {"key": "val"}

    @pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
    def test_subclass_catchable_as_vten_error(self, cls):
        """All subclasses are catchable via except VTenError."""
        with pytest.raises(VTenError):
            raise cls("test")

    @pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
    def test_subclass_defaults(self, cls):
        """All subclasses have correct defaults without kwargs."""
        e = cls("minimal")
        assert e.kernel_path == ""
        assert e.stage == ""
        assert e.context == {}


# ═══════════════════════════════════════════════════════════════════
# §4  Backend error code mapping — 00_data_models.md §10.13
# ═══════════════════════════════════════════════════════════════════


class TestBackendErrorCodeMapping:
    """raise_backend_error() maps error codes to correct exception types."""

    def test_addr_unmatch_raises_bfm_error(self):
        from vten.backend.base import BackendErrorCode, raise_backend_error

        with pytest.raises(BFMError):
            raise_backend_error(BackendErrorCode.ADDR_UNMATCH, 0, "address mismatch")

    def test_poll_timeout_raises_poll_timeout_error(self):
        from vten.backend.base import BackendErrorCode, raise_backend_error

        with pytest.raises(PollTimeoutError):
            raise_backend_error(BackendErrorCode.POLL_TIMEOUT, 1, "poll timed out")

    def test_bfm_queue_error_raises_bfm_error(self):
        from vten.backend.base import BackendErrorCode, raise_backend_error

        with pytest.raises(BFMError):
            raise_backend_error(BackendErrorCode.BFM_QUEUE_ERROR, 2, "queue full")

    def test_timeout_raises_timeout_error(self):
        from vten.backend.base import BackendErrorCode, raise_backend_error

        with pytest.raises(VTenTimeoutError):
            raise_backend_error(BackendErrorCode.TIMEOUT, 3, "timed out")

    def test_unknown_code_raises_backend_error(self):
        """Unmapped error codes fall back to BackendError."""
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError):
            raise_backend_error(99, 0, "unknown")

    def test_error_message_includes_cmd_id(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError, match=r"cmd_id=7"):
            raise_backend_error(99, 7, "something")

    def test_error_context_includes_code_and_cmd_id(self):
        from vten.backend.base import raise_backend_error

        try:
            raise_backend_error(5, 3, "shm access")
        except BackendError as e:
            assert e.context["error_code"] == 5
            assert e.context["cmd_id"] == 3


# ═══════════════════════════════════════════════════════════════════
# §5  Error count — exactly 22 classes defined
# ═══════════════════════════════════════════════════════════════════


class TestErrorCompleteness:
    """Verify all expected error classes exist."""

    def test_total_error_class_count(self):
        """00_data_models.md §11 defines error classes."""
        import vten.errors as mod

        error_classes = [
            v for v in vars(mod).values()
            if isinstance(v, type) and issubclass(v, Exception) and v is not Exception
        ]
        assert len(error_classes) == 24
