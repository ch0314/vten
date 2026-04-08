"""Tests for multi-batch execution lifecycle.

Verifies:
1. Backend auto-manages session (first execute opens, subsequent reuse)
2. ExecutionContext delegates to backend.execute() uniformly
3. Backend cleanup closes active session
4. Backward compatibility (legacy backends without session support)
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from vten.backend.base import Backend, BackendResult
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.runtime.context import ExecutionContext, ExecutionResult
from vten.spec.models import (
    InterfaceSpec,
    KernelSpec,
    OpCode,
    PackingScheme,
    Protocol,
)


# ── Test Kernel ──


class StreamKernel(Kernel):
    data_in = Tensor(shape=(32,), dtype=torch.int8, interface="axis_in")
    data_out = Tensor(shape=(32,), dtype=torch.int8, interface="axis_out")

    def forward(self, **inputs):
        return self.data_in.data.clone()


def _stream_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="stream_test",
        rtl_top="rtl/stream.sv",
        interfaces={
            "axis_in": InterfaceSpec(
                name="axis_in",
                rtl_port="s_axis_in",
                protocol=Protocol.AXI4S,
                tensor="data_in",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "axis_out": InterfaceSpec(
                name="axis_out",
                rtl_port="m_axis_out",
                protocol=Protocol.AXI4S,
                tensor="data_out",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
        },
    )


# ── Mock backend with session auto-management (mirrors SimBackend) ──


class MockSessionBackend(Backend):
    """Mock backend that auto-manages session in execute()."""

    def __init__(self):
        self.calls: list[str] = []
        self._session_active = False

    @property
    def compile_target(self) -> str:
        return "sim"

    def execute(self, compiled) -> BackendResult:
        if not self._session_active:
            self.calls.append("session_start")
            self._session_active = True
        else:
            self.calls.append("batch_submit")
        self.calls.append("wait")
        return BackendResult(status=0)

    def cleanup(self) -> None:
        if self._session_active:
            self.calls.append("session_close")
            self._session_active = False
        self.calls.append("cleanup")


class MockLegacyBackend(Backend):
    """Mock backend that does NOT support sessions (one-shot execute)."""

    def __init__(self):
        self.calls: list[str] = []

    @property
    def compile_target(self) -> str:
        return "sim"

    def execute(self, compiled) -> BackendResult:
        self.calls.append("execute")
        return BackendResult(status=0)

    def cleanup(self) -> None:
        self.calls.append("cleanup")


# ── Helpers ──


def _make_ctx_and_run(backend):
    """Create ExecutionContext, record ops, call run()."""
    ctx = ExecutionContext(backend=backend, project_params={"N": 32})
    ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
    ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
    ctx.push_tensor(ki.get_tensor("data_in"))
    ctx.pull_tensor(ki.get_tensor("data_out"))
    return ctx, ctx.run()


# ── Session lifecycle tests ──


class TestSessionLifecycle:

    def test_first_execute_starts_session(self):
        """First ctx.run() → backend.execute() starts session."""
        backend = MockSessionBackend()
        _make_ctx_and_run(backend)

        assert "session_start" in backend.calls
        assert "wait" in backend.calls
        assert backend._session_active is True

    def test_second_execute_reuses_session(self):
        """Second ctx.run() on same backend reuses session."""
        backend = MockSessionBackend()

        _make_ctx_and_run(backend)
        _make_ctx_and_run(backend)

        assert backend.calls.count("session_start") == 1
        assert backend.calls.count("batch_submit") == 1
        assert backend.calls.count("wait") == 2

    def test_cleanup_closes_session(self):
        """cleanup() closes active session."""
        backend = MockSessionBackend()
        _make_ctx_and_run(backend)
        backend.cleanup()

        assert "session_close" in backend.calls
        assert backend._session_active is False

    def test_cleanup_idempotent(self):
        """Calling cleanup() multiple times is safe."""
        backend = MockSessionBackend()
        _make_ctx_and_run(backend)

        backend.cleanup()
        backend.cleanup()
        backend.cleanup()

        assert backend.calls.count("session_close") == 1
        assert backend.calls.count("cleanup") == 3

    def test_backend_context_manager(self):
        """with backend: pattern calls cleanup on exit."""
        backend = MockSessionBackend()
        with backend:
            _make_ctx_and_run(backend)

        assert "session_close" in backend.calls
        assert "cleanup" in backend.calls

    def test_ctx_close_is_noop(self):
        """ExecutionContext.close() is no-op (backend manages lifecycle)."""
        backend = MockSessionBackend()
        ctx, _ = _make_ctx_and_run(backend)
        ctx.close()

        # close() should not trigger session_close
        assert "session_close" not in backend.calls
        # session still active
        assert backend._session_active is True

    def test_ctx_context_manager_is_noop(self):
        """with ctx: pattern is no-op for backend lifecycle."""
        backend = MockSessionBackend()
        with ExecutionContext(backend=backend, project_params={"N": 32}) as ctx:
            ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
            ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
            ctx.push_tensor(ki.get_tensor("data_in"))
            ctx.pull_tensor(ki.get_tensor("data_out"))
            ctx.run()

        # Backend session still active (caller must cleanup)
        assert "session_close" not in backend.calls
        assert backend._session_active is True


# ── Backward compatibility tests ──


class TestBackwardCompat:

    def test_legacy_backend_uses_execute(self):
        """Backend without session support uses execute()."""
        backend = MockLegacyBackend()
        _make_ctx_and_run(backend)

        assert "execute" in backend.calls

    def test_no_backend_returns_done(self):
        """No backend → compile-only, returns DONE."""
        ctx = ExecutionContext(project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx.push_tensor(ki.get_tensor("data_in"))
        ctx.pull_tensor(ki.get_tensor("data_out"))
        result = ctx.run()
        assert result.status == "DONE"
