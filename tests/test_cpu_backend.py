"""Tests for CpuBackend — pure Python forward() execution."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vten.backend.cpu import CpuBackend
from vten.backend.base import BackendResult, RunContext
from vten.backend.registry import get_backend, available_backends


class TestCpuBackendRegistry:
    """CpuBackend is registered and discoverable."""

    def test_cpu_in_available_backends(self):
        assert "cpu" in available_backends()

    def test_get_backend_cpu(self):
        backend = get_backend("cpu", {})
        assert isinstance(backend, CpuBackend)


class TestCpuBackendLifecycle:
    """Context manager and cleanup."""

    def test_context_manager(self):
        backend = CpuBackend()
        with backend:
            pass  # no-op cleanup

    def test_cleanup_idempotent(self):
        backend = CpuBackend()
        backend.cleanup()
        backend.cleanup()

    def test_compile_target_is_sim(self):
        from vten.backend.base import CompileTarget
        backend = CpuBackend()
        assert backend.compile_target == CompileTarget.SIM

    def test_set_run_context(self):
        backend = CpuBackend()
        ctx = RunContext()
        backend.set_run_context(ctx)
        assert backend._run_ctx is ctx


class TestCpuBackendExecute:
    """CpuBackend.execute() runs forward() and serializes outputs."""

    def test_empty_view_returns_ok(self):
        """No kernel instance → returns status=0 with empty buffers."""
        compiled = MagicMock()
        compiled.flattened_view.sub_kernels = {}

        backend = CpuBackend()
        result = backend.execute(compiled)

        assert isinstance(result, BackendResult)
        assert result.status == 0

    def test_forward_called_on_kernel(self):
        """forward() is called via run_forward."""
        compiled = MagicMock()
        ki = MagicMock()
        ki.kernel_class_instance = MagicMock()
        compiled.flattened_view.sub_kernels = {"main": ki}
        compiled.flattened_view.exposed_tensors = {}

        with patch("vten.runtime.golden.run_forward", return_value={}) as mock_fwd:
            backend = CpuBackend()
            result = backend.execute(compiled)

            mock_fwd.assert_called_once_with(ki.kernel_class_instance)
            assert result.status == 0

    def test_output_buffers_populated(self):
        """DEV_TO_HOST tensor data appears in output_buffers."""
        from vten.spec.models import Direction, PackingScheme

        # Set up a simple exposed tensor
        exposed = MagicMock()
        exposed.direction = Direction.DEV_TO_HOST
        exposed._port_buffers = None
        exposed.top_interface = "m_axis"
        exposed.origin_tensor.dtype = torch.int8

        packing = PackingScheme(
            element_width=8,
            elements_per_beat=32,
        )
        iface = MagicMock()
        iface.packing = packing

        view = MagicMock()
        view.sub_kernels = {"main": MagicMock()}
        view.sub_kernels["main"].kernel_class_instance = MagicMock()
        view.exposed_tensors = {"output": exposed}
        view.top_spec.get_interface.return_value = iface

        compiled = MagicMock()
        compiled.flattened_view = view
        compiled.buffer_ids = {"output": 5}

        # forward() returns a small tensor
        fwd_output = torch.tensor([1, 2, 3, 4], dtype=torch.int8)

        with patch("vten.runtime.golden.run_forward", return_value={"output": fwd_output}):
            backend = CpuBackend()
            result = backend.execute(compiled)

        assert 5 in result.output_buffers
        assert len(result.output_buffers[5]) > 0
