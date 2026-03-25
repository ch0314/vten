"""Tests for vten.functional — High-level API (run_kernel, KernelExecutor).

Tests cover:
1. run_kernel() — DSL ops auto-generation, input assignment, configure flag
2. KernelExecutor — single/repeated calls, auto-alias detection
3. BatchResult.output_tensors field
4. Public exports from vten.__init__
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from vten.functional import KernelExecutor, run_kernel
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.runtime.context import BatchResult, ExecutionContext
from vten.spec.models import OpKind


# ── Test Kernel Definitions ──


class PassthroughKernel(Kernel):
    """Minimal kernel: one H2D input, one D2H output."""

    data_in = Tensor(shape=(4,), dtype=torch.uint8, interface="axi_stream_in")
    data_out = Tensor(shape=(4,), dtype=torch.uint8, interface="axi_stream_out")


class TwoInputKernel(Kernel):
    """Kernel with two H2D inputs and one D2H output."""

    ifm = Tensor(shape=(8,), dtype=torch.uint8, interface="ifm_port")
    wgt = Tensor(shape=(8,), dtype=torch.uint8, interface="wgt_port")
    ofm = Tensor(shape=(8,), dtype=torch.uint8, interface="ofm_port")


# ── Helper: capture pending ops before run() clears them ──


def _capture_run(captured_ops, captured_kernels=None):
    """Return a mock run() that captures ops and kernels, then returns BatchResult."""

    def mock_run(self):
        captured_ops.extend(self._pending_ops)
        if captured_kernels is not None:
            captured_kernels.update(self._kernels)
        # Return a minimal BatchResult without going through compile pipeline
        return BatchResult(status="DONE")

    return mock_run


# ═════════════════════════════════════════════════════════════════
# run_kernel tests
# ═════════════════════════════════════════════════════════════════


class TestRunKernel:
    """Tests for run_kernel() one-shot API."""

    def test_returns_dict(self):
        """run_kernel returns a dict (output_tensors from BatchResult)."""
        captured_ops = []
        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            result = run_kernel(
                PassthroughKernel,
                {"data_in": torch.tensor([1, 2, 3, 4], dtype=torch.uint8)},
            )
        assert isinstance(result, dict)

    def test_input_tensor_assigned(self):
        """Input tensors are assigned to kernel instance before run()."""
        captured_ops = []
        captured_ki = {}
        x = torch.tensor([10, 20, 30, 40], dtype=torch.uint8)

        with patch.object(
            ExecutionContext, "run", _capture_run(captured_ops, captured_ki)
        ):
            run_kernel(PassthroughKernel, {"data_in": x})

        ki = captured_ki["PassthroughKernel"]
        assert torch.equal(ki.get_tensor("data_in").data, x)

    def test_send_recv_ops_generated(self):
        """run_kernel generates SEND_TENSOR for inputs, RECV_TENSOR for non-inputs."""
        captured_ops = []
        x = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            run_kernel(PassthroughKernel, {"data_in": x})

        op_kinds = [op.kind for op in captured_ops]
        assert OpKind.SEND_TENSOR in op_kinds
        assert OpKind.RECV_TENSOR in op_kinds

    def test_send_recv_tensor_names(self):
        """Correct tensors assigned to send/recv ops."""
        captured_ops = []
        x = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            run_kernel(PassthroughKernel, {"data_in": x})

        send_op = next(op for op in captured_ops if op.kind == OpKind.SEND_TENSOR)
        recv_op = next(op for op in captured_ops if op.kind == OpKind.RECV_TENSOR)
        assert send_op.tensor.name == "data_in"
        assert recv_op.tensor.name == "data_out"

    def test_configure_flag_adds_op(self):
        """configure=True adds CONFIGURE op."""
        captured_ops = []
        x = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            run_kernel(PassthroughKernel, {"data_in": x}, configure=True)

        op_kinds = [op.kind for op in captured_ops]
        assert OpKind.CONFIGURE in op_kinds

    def test_no_configure_by_default(self):
        """Default: no CONFIGURE op."""
        captured_ops = []
        x = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            run_kernel(PassthroughKernel, {"data_in": x})

        op_kinds = [op.kind for op in captured_ops]
        assert OpKind.CONFIGURE not in op_kinds

    def test_multiple_inputs(self):
        """Multiple input tensors each get a send_tensor; non-inputs get recv_tensor."""
        captured_ops = []
        ifm = torch.zeros(8, dtype=torch.uint8)
        wgt = torch.ones(8, dtype=torch.uint8)

        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            run_kernel(TwoInputKernel, {"ifm": ifm, "wgt": wgt})

        send_ops = [op for op in captured_ops if op.kind == OpKind.SEND_TENSOR]
        recv_ops = [op for op in captured_ops if op.kind == OpKind.RECV_TENSOR]
        assert len(send_ops) == 2  # ifm + wgt
        assert len(recv_ops) == 1  # ofm

    def test_params_forwarded(self):
        """Runtime params are forwarded to kernel instantiation."""
        captured_ops = []
        captured_ki = {}
        x = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with patch.object(
            ExecutionContext, "run", _capture_run(captured_ops, captured_ki)
        ):
            run_kernel(PassthroughKernel, {"data_in": x}, params={"SIZE": 16})

        ki = captured_ki["PassthroughKernel"]
        assert ki.runtime_params.get("SIZE") == 16

    def test_send_depends_on_nothing_first(self):
        """First send_tensor has no dependency."""
        captured_ops = []
        x = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            run_kernel(PassthroughKernel, {"data_in": x})

        send_op = next(op for op in captured_ops if op.kind == OpKind.SEND_TENSOR)
        assert send_op.dep == []

    def test_recv_depends_on_last_send(self):
        """recv_tensor depends on the last send_tensor handle."""
        captured_ops = []
        x = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            run_kernel(PassthroughKernel, {"data_in": x})

        recv_op = next(op for op in captured_ops if op.kind == OpKind.RECV_TENSOR)
        # recv should have dependency on the send handle
        assert len(recv_op.dep) == 1
        assert recv_op.dep[0].op.kind == OpKind.SEND_TENSOR


# ═════════════════════════════════════════════════════════════════
# KernelExecutor tests
# ═════════════════════════════════════════════════════════════════


class TestKernelExecutor:
    """Tests for KernelExecutor reusable API."""

    def test_single_call_returns_dict(self):
        """Single call returns dict."""
        with patch.object(ExecutionContext, "run", _capture_run([])):
            npu = KernelExecutor(PassthroughKernel)
            result = npu(data_in=torch.tensor([1, 2, 3, 4], dtype=torch.uint8))
        assert isinstance(result, dict)

    def test_dsl_ops_generated(self):
        """KernelExecutor auto-generates send/recv ops."""
        captured_ops = []
        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            npu = KernelExecutor(PassthroughKernel)
            npu(data_in=torch.tensor([1, 2, 3, 4], dtype=torch.uint8))

        op_kinds = [op.kind for op in captured_ops]
        assert OpKind.SEND_TENSOR in op_kinds
        assert OpKind.RECV_TENSOR in op_kinds

    def test_configure_flag(self):
        """configure=True in constructor adds CONFIGURE on each call."""
        captured_ops = []
        with patch.object(ExecutionContext, "run", _capture_run(captured_ops)):
            npu = KernelExecutor(PassthroughKernel, configure=True)
            npu(data_in=torch.tensor([1, 2, 3, 4], dtype=torch.uint8))

        op_kinds = [op.kind for op in captured_ops]
        assert OpKind.CONFIGURE in op_kinds

    def test_per_call_params_override(self):
        """_params per call are merged with base params."""
        captured_ki = {}
        with patch.object(
            ExecutionContext, "run", _capture_run([], captured_ki)
        ):
            npu = KernelExecutor(PassthroughKernel, params={"A": 1})
            npu(data_in=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
                _params={"B": 2})

        ki = captured_ki["PassthroughKernel"]
        assert ki.runtime_params.get("A") == 1
        assert ki.runtime_params.get("B") == 2

    def test_auto_alias_detection(self):
        """Previous output passed as next input triggers alias."""
        alias_calls = []
        original_alias = ExecutionContext.alias

        def mock_alias(self, src, dst):
            alias_calls.append((src.name, dst.name))
            original_alias(self, src, dst)

        with patch.object(ExecutionContext, "run", _capture_run([])), \
             patch.object(ExecutionContext, "alias", mock_alias):
            npu = KernelExecutor(TwoInputKernel)
            ifm = torch.zeros(8, dtype=torch.uint8)
            wgt = torch.ones(8, dtype=torch.uint8)

            # First call: normal
            npu(ifm=ifm, wgt=wgt)

            # Simulate prev output: set up _prev_outputs with a known tensor
            fake_ofm = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.uint8)
            npu._prev_outputs = {id(fake_ofm): "ofm"}

            # Second call: pass fake_ofm as ifm → should trigger alias
            npu(ifm=fake_ofm, wgt=wgt)

        assert len(alias_calls) == 1
        assert alias_calls[0] == ("ofm", "ifm")

    def test_no_alias_first_call(self):
        """First call has no previous outputs → no alias."""
        alias_calls = []
        original_alias = ExecutionContext.alias

        def mock_alias(self, src, dst):
            alias_calls.append((src.name, dst.name))
            original_alias(self, src, dst)

        with patch.object(ExecutionContext, "run", _capture_run([])), \
             patch.object(ExecutionContext, "alias", mock_alias):
            npu = KernelExecutor(TwoInputKernel)
            npu(ifm=torch.zeros(8, dtype=torch.uint8),
                wgt=torch.ones(8, dtype=torch.uint8))

        assert len(alias_calls) == 0

    def test_no_alias_for_new_tensor(self):
        """Fresh tensors (not from prev output) don't trigger alias."""
        alias_calls = []
        original_alias = ExecutionContext.alias

        def mock_alias(self, src, dst):
            alias_calls.append((src.name, dst.name))
            original_alias(self, src, dst)

        with patch.object(ExecutionContext, "run", _capture_run([])), \
             patch.object(ExecutionContext, "alias", mock_alias):
            npu = KernelExecutor(TwoInputKernel)
            npu(ifm=torch.zeros(8, dtype=torch.uint8),
                wgt=torch.ones(8, dtype=torch.uint8))
            # New tensors have different id()
            npu(ifm=torch.zeros(8, dtype=torch.uint8),
                wgt=torch.ones(8, dtype=torch.uint8))

        assert len(alias_calls) == 0

    def test_prev_ki_tracked(self):
        """After a call, _prev_ki stores the kernel instance."""
        with patch.object(ExecutionContext, "run", _capture_run([])):
            npu = KernelExecutor(PassthroughKernel)
            assert npu._prev_ki is None
            npu(data_in=torch.tensor([1, 2, 3, 4], dtype=torch.uint8))
            assert npu._prev_ki is not None

    def test_aliased_tensor_not_assigned_data(self):
        """When alias is applied, tensor.data is NOT set (alias reuses buffer)."""
        captured_ki = {}

        def capture_run(self):
            captured_ki.update(self._kernels)
            return BatchResult(status="DONE")

        with patch.object(ExecutionContext, "run", capture_run):
            npu = KernelExecutor(TwoInputKernel)
            ifm = torch.zeros(8, dtype=torch.uint8)
            wgt = torch.ones(8, dtype=torch.uint8)
            npu(ifm=ifm, wgt=wgt)

            # Set up fake alias scenario
            fake_ofm = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.uint8)
            npu._prev_outputs = {id(fake_ofm): "ofm"}

            captured_ki.clear()
            npu(ifm=fake_ofm, wgt=wgt)

        ki = captured_ki["TwoInputKernel"]
        # ifm was aliased → data should NOT have been set to fake_ofm
        # (alias means buffer reuse, not data copy)
        ifm_tensor = ki.get_tensor("ifm")
        assert ifm_tensor.data is None

    def test_non_aliased_tensor_has_data(self):
        """Normal (non-aliased) tensors have their data assigned."""
        captured_ki = {}

        def capture_run(self):
            captured_ki.update(self._kernels)
            return BatchResult(status="DONE")

        with patch.object(ExecutionContext, "run", capture_run):
            npu = KernelExecutor(TwoInputKernel)
            ifm = torch.zeros(8, dtype=torch.uint8)
            wgt = torch.ones(8, dtype=torch.uint8)
            npu(ifm=ifm, wgt=wgt)

            fake_ofm = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.uint8)
            npu._prev_outputs = {id(fake_ofm): "ofm"}

            captured_ki.clear()
            npu(ifm=fake_ofm, wgt=wgt)

        ki = captured_ki["TwoInputKernel"]
        # wgt was not aliased → data should be set
        wgt_tensor = ki.get_tensor("wgt")
        assert torch.equal(wgt_tensor.data, wgt)


# ═════════════════════════════════════════════════════════════════
# BatchResult.output_tensors tests
# ═════════════════════════════════════════════════════════════════


class TestBatchResultOutputTensors:
    """Tests for BatchResult.output_tensors field."""

    def test_default_empty(self):
        result = BatchResult(status="DONE")
        assert result.output_tensors == {}

    def test_field_assignment(self):
        t = torch.tensor([1, 2, 3])
        result = BatchResult(status="DONE", output_tensors={"out": t})
        assert torch.equal(result.output_tensors["out"], t)

    def test_run_no_backend_returns_done(self):
        """run_kernel with no backend returns empty output_tensors."""
        with patch.object(ExecutionContext, "run", _capture_run([])):
            result = run_kernel(
                PassthroughKernel,
                {"data_in": torch.tensor([1, 2, 3, 4], dtype=torch.uint8)},
            )
        assert result == {}


# ═════════════════════════════════════════════════════════════════
# Export tests
# ═════════════════════════════════════════════════════════════════


class TestExports:
    """Verify public API exports."""

    def test_run_kernel_importable(self):
        from vten import run_kernel as rk
        assert callable(rk)

    def test_kernel_executor_importable(self):
        from vten import KernelExecutor as KE
        assert KE is not None

    def test_all_exports(self):
        import vten
        assert "run_kernel" in vten.__all__
        assert "KernelExecutor" in vten.__all__
