"""Tests for Inference API (InferenceSession, InferenceModule).

Uses example kernels (passthrough, scale_add) with mock backends
to verify the inference API flow without real FPGA hardware.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from vten.backend.base import Backend, BackendResult
from vten.inference import InferenceModule, InferenceSession
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import OpKind
from vten.runtime.context import ExecutionContext, ExecutionResult
from vten.spec.models import (
    Direction,
    InterfaceSpec,
    KernelSpec,
    PackingScheme,
    Protocol,
)

# ── Add example kernel paths ──

_PASSTHROUGH_DIR = Path(__file__).resolve().parent.parent / "examples" / "passthrough"
_SCALE_ADD_DIR = Path(__file__).resolve().parent.parent / "examples" / "scale_add"

for p in [
    str(_PASSTHROUGH_DIR / "kernels" / "passthrough"),
    str(_SCALE_ADD_DIR / "kernels" / "scale_add"),
    str(_SCALE_ADD_DIR / "kernels"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Test Kernel (inline, no spec file needed) ──


class StreamKernel(Kernel):
    """Simple passthrough kernel for unit tests."""
    data_in = Tensor(shape=(32,), dtype=torch.int8, interface="axis_in",
                     direction=Direction.HOST_TO_DEV)
    data_out = Tensor(shape=(32,), dtype=torch.int8, interface="axis_out",
                      direction=Direction.DEV_TO_HOST)

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        data = inputs.get("data_in", self.data_in.data)
        return {"data_out": data.clone()}

    def run(self, ctx) -> None:
        h_send = ctx.send_tensor(self.data_in)
        h_recv = ctx.recv_tensor(self.data_out, dep=h_send)
        ctx.verify(h_recv)


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


# ── Mock Backends ──


class MockSimBackend(Backend):
    """SIM backend mock — compile_target='sim', returns empty output."""

    def __init__(self):
        self.calls: list[str] = []
        self.last_compiled = None

    @property
    def compile_target(self) -> str:
        return "sim"

    def execute(self, compiled) -> BackendResult:
        self.calls.append("execute")
        self.last_compiled = compiled
        return BackendResult(status=0)

    def cleanup(self) -> None:
        self.calls.append("cleanup")


class MockHwBackend(Backend):
    """HW backend mock — simulates XRT persistent mode with mock BOs."""

    def __init__(self):
        self.calls: list[str] = []
        self._persistent = False
        self._buffers: dict[int, MockBO] = {}
        self.last_compiled = None

    @property
    def compile_target(self) -> str:
        return "hw"

    def get_buffer_object(self, buffer_id: int) -> object | None:
        return self._buffers.get(buffer_id)

    def inject_prebound(self, buffer_id: int, bo: object) -> None:
        self._buffers[buffer_id] = bo

    def execute(self, compiled) -> BackendResult:
        self.calls.append("execute")
        self.last_compiled = compiled

        # For each buffer, create a mock BO (simulates XRT BO allocation)
        for name, exposed in compiled.flattened_view.exposed_tensors.items():
            bid = compiled.buffer_ids.get(name)
            if bid is not None and bid not in self._buffers:
                size = exposed._serialized_size or 32
                self._buffers[bid] = MockBO(size)

        # Inject prebound buffers
        for bid, bo in compiled.prebound_buffers.items():
            self._buffers[bid] = bo

        return BackendResult(status=0)

    def cleanup(self) -> None:
        self.calls.append("cleanup")
        self._buffers.clear()


class MockBO:
    """Mock XRT buffer object."""

    def __init__(self, size: int = 32, data: bytes | None = None):
        self._size = size
        self._data = data or bytes(size)
        self._synced_to = False
        self._synced_from = False

    def write(self, data: bytes) -> None:
        self._data = bytes(data)
        self._size = len(data)

    def read(self, size: int) -> bytes:
        return self._data[:size]

    def size(self) -> int:
        return self._size

    def sync(self, direction: int) -> None:
        if direction == 1:  # TO_DEVICE
            self._synced_to = True
        elif direction == 2:  # FROM_DEVICE
            self._synced_from = True

    def address(self) -> int:
        return 0x10000 + id(self) % 0x10000


# ═══════════════════════════════════════════════════════════════════
# 1. Tensor Device State
# ═══════════════════════════════════════════════════════════════════


class TestTensorDeviceState:

    def test_on_device_default_false(self):
        t = Tensor(shape=(32,), dtype=torch.int8, interface="test")
        assert t.on_device is False

    def test_on_device_after_bind(self):
        t = Tensor(shape=(32,), dtype=torch.int8, interface="test")
        t._bind_bo(MockBO(32), 32)
        assert t.on_device is True

    def test_on_device_after_unbind(self):
        t = Tensor(shape=(32,), dtype=torch.int8, interface="test")
        t._bind_bo(MockBO(32), 32)
        t._unbind_bo()
        assert t.on_device is False

    def test_cpu_returns_host_data_when_no_bo(self):
        t = Tensor(shape=(32,), dtype=torch.int8, interface="test")
        t.data = torch.zeros(32, dtype=torch.int8)
        result = t.cpu()
        assert torch.equal(result, t.data)

    def test_cpu_raises_when_no_data(self):
        t = Tensor(shape=(32,), dtype=torch.int8, interface="test")
        with pytest.raises(RuntimeError, match="no data"):
            t.cpu()

    def test_cpu_syncs_from_device(self):
        bo = MockBO(32)
        t = Tensor(shape=(32,), dtype=torch.int8, interface="test")
        t._bind_bo(bo, 32)
        t.cpu()
        assert bo._synced_from is True

    def test_cpu_with_deserialize_fn(self):
        """cpu() applies deserialize_fn when set."""
        expected = torch.ones(4, dtype=torch.int32)
        bo = MockBO(16, data=bytes(16))

        def _deser(raw: bytes) -> torch.Tensor:
            return expected

        t = Tensor(shape=(4,), dtype=torch.int32, interface="test")
        t._bind_bo(bo, 16, deserialize_fn=_deser)
        result = t.cpu()
        assert torch.equal(result, expected)

    def test_describe_includes_on_device(self):
        t = Tensor(shape=(32,), dtype=torch.int8, interface="test")
        t.name = "test_tensor"
        t._resolved_shape = (32,)
        info = t.describe()
        assert "on_device" in info
        assert info["on_device"] is False

    def test_numpy(self):
        t = Tensor(shape=(4,), dtype=torch.float32, interface="test")
        t.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        arr = t.numpy()
        assert arr.shape == (4,)
        assert arr[0] == 1.0


# ═══════════════════════════════════════════════════════════════════
# 2. ExecutionContext Inference Mode
# ═══════════════════════════════════════════════════════════════════


class TestInferenceMode:

    def test_verify_noop_in_inference_mode(self):
        """verify() does nothing in inference mode."""
        ctx = ExecutionContext(project_params={"N": 32}, mode="inference")
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)

        # Record ops
        h_send = ctx.send_tensor(ki.get_tensor("data_in"))
        h_recv = ctx.recv_tensor(ki.get_tensor("data_out"), dep=h_send)

        # verify() should silently return (not add to _verifications)
        ctx.verify(h_recv, torch.zeros(32, dtype=torch.int8))
        assert len(ctx._verifications) == 0

    def test_verify_works_in_verification_mode(self):
        """verify() is active in verification mode (default)."""
        ctx = ExecutionContext(project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)

        h_send = ctx.send_tensor(ki.get_tensor("data_in"))
        h_recv = ctx.recv_tensor(ki.get_tensor("data_out"), dep=h_send)
        ctx.verify(h_recv, torch.zeros(32, dtype=torch.int8))
        assert len(ctx._verifications) == 1

    def test_bind_device_buffer(self):
        """bind_device_buffer() stores BO mapping."""
        ctx = ExecutionContext(project_params={"N": 32}, mode="inference")
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        bo = MockBO(32)
        ctx.bind_device_buffer(ki.get_tensor("data_in"), bo)
        assert "data_in" in ctx._bound_bos
        assert ctx._bound_bos["data_in"] is bo

    def test_send_tensor_skip_data_when_bound(self):
        """send_tensor() with bound BO sets _skip_data flag."""
        ctx = ExecutionContext(project_params={"N": 32}, mode="inference")
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)

        bo = MockBO(32)
        ctx.bind_device_buffer(ki.get_tensor("data_in"), bo)

        h = ctx.send_tensor(ki.get_tensor("data_in"))
        # The recorded op should have _skip_data=True
        assert h.op._skip_data is True

    def test_send_tensor_normal_when_not_bound(self):
        """send_tensor() without bound BO records normally."""
        ctx = ExecutionContext(project_params={"N": 32}, mode="inference")
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)

        h = ctx.send_tensor(ki.get_tensor("data_in"))
        assert h.op._skip_data is False

    def test_recv_tensor_recorded_in_inference(self):
        """recv_tensor() in inference mode records RECV_TENSOR (same as verification)."""
        ctx = ExecutionContext(project_params={"N": 32}, mode="inference")
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)

        h = ctx.recv_tensor(ki.get_tensor("data_out"))
        assert h.op.kind == OpKind.RECV_TENSOR

    def test_recv_tensor_recorded_in_verification(self):
        """recv_tensor() in verification mode records RECV_TENSOR."""
        ctx = ExecutionContext(project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)

        h = ctx.recv_tensor(ki.get_tensor("data_out"))
        assert h.op.kind == OpKind.RECV_TENSOR

    def test_prebound_injected_into_compiled(self):
        """Bound BOs are injected into CompiledResult.prebound_buffers."""
        backend = MockSimBackend()
        ctx = ExecutionContext(backend=backend, project_params={"N": 32},
                               mode="inference")
        ki = ctx.instantiate(StreamKernel, spec=_stream_spec(), N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)

        bo = MockBO(32)
        ctx.bind_device_buffer(ki.get_tensor("data_in"), bo)

        ctx.send_tensor(ki.get_tensor("data_in"))
        ctx.recv_tensor(ki.get_tensor("data_out"))
        ctx.run()

        compiled = ctx._last_compiled
        # data_in should have a prebound buffer
        bid = compiled.buffer_ids.get("data_in")
        assert bid is not None
        assert bid in compiled.prebound_buffers
        assert compiled.prebound_buffers[bid] is bo


# ═══════════════════════════════════════════════════════════════════
# 3. InferenceSession with Inline StreamKernel
# ═══════════════════════════════════════════════════════════════════


class TestInferenceSessionInline:

    def test_run_sim_backend(self):
        """session.run() with SIM backend returns Tensor with .data."""
        backend = MockSimBackend()
        session = InferenceSession(backend)

        x = torch.randint(-128, 127, (32,), dtype=torch.int8)
        result = session.run(
            StreamKernel,
            inputs={"data_in": x},
            N=32,
            _spec=_stream_spec(),
        )
        # SIM backend returns empty output_buffers, so output Tensor
        # will have .data=None (no actual SHM data), but the Tensor
        # object should exist for each DEV_TO_HOST tensor
        assert "data_out" in result
        t = result["data_out"]
        assert isinstance(t, Tensor)

    def test_run_hw_backend(self):
        """session.run() with HW backend returns Tensor(on_device)."""
        backend = MockHwBackend()
        session = InferenceSession(backend)

        x = torch.randint(-128, 127, (32,), dtype=torch.int8)
        result = session.run(
            StreamKernel,
            inputs={"data_in": x},
            N=32,
            _spec=_stream_spec(),
        )
        assert "data_out" in result
        t = result["data_out"]
        assert isinstance(t, Tensor)
        assert t.on_device is True

    def test_run_with_device_tensor_input(self):
        """session.run() with Tensor(on_device) input uses bound BO."""
        backend = MockHwBackend()
        session = InferenceSession(backend)
        assert backend._persistent is True

        # First run: host tensor input
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)
        r1 = session.run(
            StreamKernel,
            inputs={"data_in": x},
            N=32,
            _spec=_stream_spec(),
        )
        assert "data_out" in r1
        assert r1["data_out"].on_device is True

        # Second run: pass device tensor as input
        # (simulates multi-layer chaining)
        r2 = session.run(
            StreamKernel,
            inputs={"data_in": r1["data_out"]},
            N=32,
            _spec=_stream_spec(),
        )
        assert "data_out" in r2

    def test_run_enables_persistent_mode(self):
        """InferenceSession auto-sets backend._persistent = True."""
        backend = MockHwBackend()
        assert backend._persistent is False
        InferenceSession(backend)
        assert backend._persistent is True

    def test_cleanup(self):
        """session.cleanup() calls backend.cleanup()."""
        backend = MockSimBackend()
        session = InferenceSession(backend)
        session.cleanup()
        assert "cleanup" in backend.calls

    def test_base_params_merged(self):
        """base_params are merged with per-run params."""
        backend = MockSimBackend()
        session = InferenceSession(backend, base_params={"N": 32})

        x = torch.zeros(32, dtype=torch.int8)
        # N comes from base_params, no need to pass it again
        result = session.run(
            StreamKernel,
            inputs={"data_in": x},
            _spec=_stream_spec(),
        )
        assert "data_out" in result


# ═══════════════════════════════════════════════════════════════════
# 4. InferenceSession with Passthrough Example Kernel
# ═══════════════════════════════════════════════════════════════════


class TestInferenceSessionPassthrough:

    def _get_kernel_and_spec(self):
        from passthrough_kernel import PassthroughKernel
        from vten.spec.parser import parse_kernel_spec

        spec_path = _PASSTHROUGH_DIR / "kernels" / "passthrough" / "kernel_spec.yaml"
        spec = parse_kernel_spec(spec_path)
        return PassthroughKernel, spec

    def test_run_passthrough_sim(self):
        """session.run() with real passthrough kernel, SIM backend."""
        KernelClass, spec = self._get_kernel_and_spec()
        backend = MockSimBackend()
        session = InferenceSession(backend, base_params={"N": 1024})

        x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
        result = session.run(
            KernelClass,
            inputs={"data_in": x},
            _spec=spec,
        )
        assert "data_out" in result
        assert backend.calls == ["execute"]

    def test_run_passthrough_hw(self):
        """session.run() with real passthrough kernel, HW mock backend."""
        KernelClass, spec = self._get_kernel_and_spec()
        backend = MockHwBackend()
        session = InferenceSession(backend)

        x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
        result = session.run(
            KernelClass,
            inputs={"data_in": x},
            N=1024,
            _spec=spec,
        )
        assert "data_out" in result
        t = result["data_out"]
        assert t.on_device is True
        assert t._resolved_shape == (1024,)
        assert t.dtype == torch.int8

    def test_two_layer_chain(self):
        """Two sequential run() calls — output feeds into next input."""
        KernelClass, spec = self._get_kernel_and_spec()
        backend = MockHwBackend()
        session = InferenceSession(backend)

        x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
        r1 = session.run(KernelClass, inputs={"data_in": x},
                         N=1024, _spec=spec)
        r2 = session.run(KernelClass, inputs={"data_in": r1["data_out"]},
                         N=1024, _spec=spec)
        assert r2["data_out"].on_device is True
        assert backend.calls == ["execute", "execute"]

    def test_verify_not_called_in_inference(self):
        """Passthrough kernel calls ctx.verify() in run(), but it's no-op."""
        KernelClass, spec = self._get_kernel_and_spec()
        backend = MockSimBackend()
        session = InferenceSession(backend)

        x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
        # This should not raise even though the backend returns no data
        # (verify is no-op in inference mode)
        result = session.run(
            KernelClass,
            inputs={"data_in": x},
            N=1024,
            _spec=spec,
        )
        assert "data_out" in result


# ═══════════════════════════════════════════════════════════════════
# 5. InferenceSession.run_pipeline()
# ═══════════════════════════════════════════════════════════════════


class TestRunPipeline:

    def test_pipeline_3_layers(self):
        """run_pipeline() chains 3 passthrough layers."""
        from passthrough_kernel import PassthroughKernel
        from vten.spec.parser import parse_kernel_spec

        spec_path = _PASSTHROUGH_DIR / "kernels" / "passthrough" / "kernel_spec.yaml"
        spec = parse_kernel_spec(spec_path)

        backend = MockHwBackend()
        session = InferenceSession(backend)

        x = torch.randint(-128, 127, (256,), dtype=torch.int8)
        result = session.run_pipeline(
            PassthroughKernel,
            layers=[{"N": 256, "_spec": spec}] * 3,
            inputs={"data_in": x},
            chain={"data_out": "data_in"},
        )
        assert "data_out" in result
        assert result["data_out"].on_device is True
        assert backend.calls.count("execute") == 3

    def test_pipeline_empty_raises(self):
        """run_pipeline() with empty layers raises ValueError."""
        backend = MockSimBackend()
        session = InferenceSession(backend)

        with pytest.raises(ValueError, match="empty"):
            session.run_pipeline(
                StreamKernel,
                layers=[],
                inputs={"data_in": torch.zeros(32, dtype=torch.int8)},
            )


# ═══════════════════════════════════════════════════════════════════
# 6. InferenceModule (nn.Module)
# ═══════════════════════════════════════════════════════════════════


class TestInferenceModule:

    def test_module_forward(self):
        """InferenceModule.forward() runs kernel and returns Tensor."""

        class TestModule(InferenceModule):
            kernel_cls = StreamKernel
            input_name = "data_in"
            output_name = "data_out"

        backend = MockHwBackend()
        session = InferenceSession(backend)

        # Module without weight/bias
        mod = TestModule(session, N=32, _spec=_stream_spec())
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)
        y = mod(x)

        assert isinstance(y, Tensor)
        assert y.on_device is True

    def test_module_sequential(self):
        """Two modules in sequence — device tensor flows through."""

        class TestModule(InferenceModule):
            kernel_cls = StreamKernel
            input_name = "data_in"
            output_name = "data_out"

        backend = MockHwBackend()
        session = InferenceSession(backend)

        m1 = TestModule(session, N=32, _spec=_stream_spec())
        m2 = TestModule(session, N=32, _spec=_stream_spec())

        x = torch.randint(-128, 127, (32,), dtype=torch.int8)
        y = m2(m1(x))

        assert isinstance(y, Tensor)
        assert y.on_device is True
        assert backend.calls.count("execute") == 2

    def test_module_is_nn_module(self):
        """InferenceModule is a proper nn.Module subclass."""

        class TestModule(InferenceModule):
            kernel_cls = StreamKernel
            input_name = "data_in"
            output_name = "data_out"

        backend = MockSimBackend()
        session = InferenceSession(backend)
        mod = TestModule(session, N=32, _spec=_stream_spec())

        assert isinstance(mod, torch.nn.Module)


# ═══════════════════════════════════════════════════════════════════
# 7. Upload
# ═══════════════════════════════════════════════════════════════════


class TestUpload:

    def test_upload_hw_returns_device_tensor(self):
        """upload() on HW backend returns Tensor(on_device=True)."""
        backend = MockHwBackend()
        session = InferenceSession(backend)

        w = torch.randint(-128, 127, (32,), dtype=torch.int8)
        t = session.upload(w, "data_in", StreamKernel,
                           params={"N": 32, "_spec": _stream_spec()})

        assert isinstance(t, Tensor)
        assert t.name == "data_in"
        # HW backend creates mock BO in interpreter
        assert t.on_device is True

    def test_upload_sim_returns_host_tensor(self):
        """upload() on SIM backend returns Tensor with .data."""
        backend = MockSimBackend()
        session = InferenceSession(backend)

        w = torch.randint(-128, 127, (32,), dtype=torch.int8)
        t = session.upload(w, "data_in", StreamKernel,
                           params={"N": 32, "_spec": _stream_spec()})

        assert isinstance(t, Tensor)
        assert t.name == "data_in"
        assert t.on_device is False
        assert torch.equal(t.data, w)
