"""Tests for multi-batch session lifecycle.

Verifies:
1. Backend session protocol (open/submit/wait/close)
2. ExecutionContext session-aware run()
3. Backward compatibility (execute() still works)
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


# ── Mock session backend ──


class MockSessionBackend(Backend):
    """Mock backend that supports session protocol."""

    def __init__(self):
        self.calls: list[str] = []
        self._session_open = False

    @property
    def supports_session(self) -> bool:
        return True

    @property
    def compile_target(self) -> str:
        return "sim"

    def execute(self, compiled) -> BackendResult:
        self.calls.append("execute")
        return BackendResult(status=0)

    def open_session(self, compiled) -> None:
        self.calls.append("open_session")
        self._session_open = True

    def submit_batch(self, compiled) -> None:
        self.calls.append("submit_batch")

    def wait_batch(self) -> BackendResult:
        self.calls.append("wait_batch")
        return BackendResult(status=0)

    def close_session(self) -> None:
        self.calls.append("close_session")
        self._session_open = False

    def cleanup(self) -> None:
        self.calls.append("cleanup")


class MockLegacyBackend(Backend):
    """Mock backend that does NOT support sessions."""

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


# ── Session lifecycle tests ──


class TestSessionLifecycle:

    def test_first_run_opens_session(self):
        """First ctx.run() calls open_session + wait_batch."""
        backend = MockSessionBackend()
        ctx = ExecutionContext(backend=backend, project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx.push_tensor(ki.get_tensor("data_in"))
        ctx.pull_tensor(ki.get_tensor("data_out"))

        ctx.run()

        assert "open_session" in backend.calls
        assert "wait_batch" in backend.calls
        assert "execute" not in backend.calls
        assert ctx._session_open is True

    def test_second_run_submits_batch(self):
        """Second ctx.run() calls submit_batch + wait_batch (not open_session)."""
        backend = MockSessionBackend()

        # First run
        ctx1 = ExecutionContext(backend=backend, project_params={"N": 32})
        ki1 = ctx1.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki1.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx1.push_tensor(ki1.get_tensor("data_in"))
        ctx1.pull_tensor(ki1.get_tensor("data_out"))
        ctx1.run()

        # Second run (new context, but session state transferred)
        ctx2 = ExecutionContext(backend=backend, project_params={"N": 32})
        ctx2._session_open = ctx1._session_open  # Transfer state
        ki2 = ctx2.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki2.get_tensor("data_in").data = torch.ones(32, dtype=torch.int8)
        ctx2.push_tensor(ki2.get_tensor("data_in"))
        ctx2.pull_tensor(ki2.get_tensor("data_out"))
        ctx2.run()

        assert backend.calls.count("open_session") == 1
        assert backend.calls.count("submit_batch") == 1
        assert backend.calls.count("wait_batch") == 2
        assert "execute" not in backend.calls

    def test_close_session(self):
        """close() calls backend.close_session()."""
        backend = MockSessionBackend()
        ctx = ExecutionContext(backend=backend, project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx.push_tensor(ki.get_tensor("data_in"))
        ctx.pull_tensor(ki.get_tensor("data_out"))
        ctx.run()
        ctx.close()

        assert "close_session" in backend.calls
        assert ctx._session_open is False

    def test_close_idempotent(self):
        """Calling close() multiple times is safe."""
        backend = MockSessionBackend()
        ctx = ExecutionContext(backend=backend, project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx.push_tensor(ki.get_tensor("data_in"))
        ctx.pull_tensor(ki.get_tensor("data_out"))
        ctx.run()

        ctx.close()
        ctx.close()
        ctx.close()

        assert backend.calls.count("close_session") == 1

    def test_context_manager(self):
        """ExecutionContext works as context manager."""
        backend = MockSessionBackend()
        with ExecutionContext(backend=backend, project_params={"N": 32}) as ctx:
            ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
            ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
            ctx.push_tensor(ki.get_tensor("data_in"))
            ctx.pull_tensor(ki.get_tensor("data_out"))
            ctx.run()

        assert "close_session" in backend.calls


# ── Backward compatibility tests ──


class TestBackwardCompat:

    def test_legacy_backend_uses_execute(self):
        """Backend without supports_session uses execute()."""
        backend = MockLegacyBackend()
        ctx = ExecutionContext(backend=backend, project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx.push_tensor(ki.get_tensor("data_in"))
        ctx.pull_tensor(ki.get_tensor("data_out"))
        ctx.run()

        assert "execute" in backend.calls
        assert "open_session" not in backend.calls

    def test_no_backend_returns_done(self):
        """No backend → compile-only, returns DONE."""
        ctx = ExecutionContext(project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx.push_tensor(ki.get_tensor("data_in"))
        ctx.pull_tensor(ki.get_tensor("data_out"))
        result = ctx.run()
        assert result.status == "DONE"


