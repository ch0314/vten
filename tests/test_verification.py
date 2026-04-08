"""Tests for verification pipeline: VerificationError, BackendResult.read_buffer,
XsimBackend buffer reader, ExecutionContext verification loop.

Spec references:
- 02_runtime_engine.md §3 (verify, _verify_immediate, _run_deferred_verifications)
- 00_data_models.md §11.8 (Buffer Descriptor Layout)
- 00_data_models.md §12 (VerificationError)
"""

from __future__ import annotations

import struct

import pytest
import torch

from vten.errors import VerificationError, VTenError


# ═══════════════════════════════════════════════════════════════════
# §1  VerificationError — errors.py
# ═══════════════════════════════════════════════════════════════════


class TestVerificationError:
    """VerificationError with tensor, shape, max_diff fields."""

    def test_is_vten_error(self):
        assert issubclass(VerificationError, VTenError)

    def test_basic_message(self):
        err = VerificationError("mismatch")
        assert str(err) == "mismatch"

    def test_tensor_field(self):
        err = VerificationError(tensor="ofm", shape=(1, 64), max_diff=0.5)
        assert err.tensor == "ofm"

    def test_shape_field(self):
        err = VerificationError(tensor="ofm", shape=(1, 64), max_diff=0.5)
        assert err.shape == (1, 64)

    def test_max_diff_field(self):
        err = VerificationError(tensor="ofm", shape=(1, 64), max_diff=0.5)
        assert err.max_diff == 0.5

    def test_auto_message_from_fields(self):
        """When no explicit message, auto-generate from tensor/shape/max_diff."""
        err = VerificationError(tensor="ofm", shape=(1, 64), max_diff=0.5)
        assert "ofm" in str(err)
        assert "0.5" in str(err)

    def test_explicit_message_overrides_auto(self):
        err = VerificationError("custom msg", tensor="ofm", shape=(1,), max_diff=0.1)
        assert str(err) == "custom msg"

    def test_defaults(self):
        err = VerificationError("fail")
        assert err.tensor == ""
        assert err.shape is None
        assert err.max_diff == 0.0

    def test_can_be_caught_as_vten_error(self):
        with pytest.raises(VTenError):
            raise VerificationError(tensor="x", shape=(2,), max_diff=1.0)

    def test_can_be_caught_as_verification_error(self):
        with pytest.raises(VerificationError) as exc_info:
            raise VerificationError(tensor="y", shape=(3, 4), max_diff=2.5)
        assert exc_info.value.tensor == "y"
        assert exc_info.value.max_diff == 2.5


# ═══════════════════════════════════════════════════════════════════
# §2  BackendResult.read_buffer — base.py
# ═══════════════════════════════════════════════════════════════════


class TestBackendResultReadBuffer:
    """BackendResult with _shm_reader for buffer data readback."""

    def test_read_buffer_returns_empty_without_reader(self):
        from vten.backend.base import BackendResult

        result = BackendResult(status=2)
        assert result.read_buffer(0) == b""

    def test_read_buffer_with_reader(self):
        from vten.backend.base import BackendResult

        data = {0: b"\x01\x02\x03", 1: b"\x04\x05"}
        result = BackendResult(status=2, _shm_reader=lambda bid: data.get(bid, b""))
        assert result.read_buffer(0) == b"\x01\x02\x03"
        assert result.read_buffer(1) == b"\x04\x05"

    def test_read_buffer_unknown_id_returns_empty(self):
        from vten.backend.base import BackendResult

        result = BackendResult(status=2, _shm_reader=lambda bid: b"")
        assert result.read_buffer(99) == b""

    def test_shm_reader_not_in_repr(self):
        """_shm_reader field has repr=False."""
        from vten.backend.base import BackendResult

        result = BackendResult(status=2, _shm_reader=lambda bid: b"")
        assert "_shm_reader" not in repr(result)

    def test_read_buffer_multiple_calls(self):
        """Reader can be called multiple times."""
        from vten.backend.base import BackendResult

        call_count = 0
        def reader(bid):
            nonlocal call_count
            call_count += 1
            return b"\xAA" * bid

        result = BackendResult(status=2, _shm_reader=reader)
        result.read_buffer(3)
        result.read_buffer(3)
        assert call_count == 2


# ═══════════════════════════════════════════════════════════════════
# §3  XsimBackend._make_buffer_reader — xsim.py
# ═══════════════════════════════════════════════════════════════════


def _build_shm_with_buffers(
    buffers: dict[int, bytes],
    num_commands: int = 1,
) -> bytearray:
    """Build a minimal SHM image with control header, buffer descriptors, and data.

    Args:
        buffers: {buffer_id: data_bytes}
        num_commands: number of command slots
    """
    from vten.runtime.shm import (
        BUF_DESC_SIZE,
        CMD_SLOT_SIZE,
        CONTROL_SIZE,
        STATS_SLOT_SIZE,
    )

    num_buffers = len(buffers)
    cmd_region_offset = CONTROL_SIZE
    stats_region_offset = cmd_region_offset + CMD_SLOT_SIZE * num_commands
    buf_desc_offset = stats_region_offset + STATS_SLOT_SIZE * num_commands
    data_region_offset = buf_desc_offset + BUF_DESC_SIZE * num_buffers

    # Align data region to 64 bytes
    data_region_offset = (data_region_offset + 63) & ~63

    # Calculate total size
    total_data = 0
    buf_offsets: dict[int, int] = {}
    for bid in sorted(buffers.keys()):
        aligned = (total_data + 63) & ~63
        buf_offsets[bid] = aligned
        total_data = aligned + len(buffers[bid])

    total_size = data_region_offset + total_data
    total_size = (total_size + 63) & ~63

    image = bytearray(total_size)

    # Control header
    struct.pack_into("<I", image, 0x00, 0x5654454E)  # MAGIC
    struct.pack_into("<I", image, 0x04, 0x00000003)  # VERSION
    struct.pack_into("<I", image, 0x10, num_commands)
    struct.pack_into("<I", image, 0x14, num_buffers)
    struct.pack_into("<Q", image, 0x18, cmd_region_offset)
    struct.pack_into("<Q", image, 0x20, stats_region_offset)
    struct.pack_into("<Q", image, 0x28, buf_desc_offset)
    struct.pack_into("<Q", image, 0x30, data_region_offset)
    struct.pack_into("<Q", image, 0x38, total_size)

    # Buffer descriptors
    for i, bid in enumerate(sorted(buffers.keys())):
        base = buf_desc_offset + i * BUF_DESC_SIZE
        struct.pack_into("<H", image, base + 0x00, bid)
        struct.pack_into("<B", image, base + 0x02, 1)  # direction=DEV_TO_HOST
        struct.pack_into("<B", image, base + 0x03, 0)  # flags
        struct.pack_into("<I", image, base + 0x04, len(buffers[bid]))
        struct.pack_into("<Q", image, base + 0x08, buf_offsets[bid])

    # Data region
    for bid, data in buffers.items():
        start = data_region_offset + buf_offsets[bid]
        image[start:start + len(data)] = data

    return image


class TestXsimBufferReader:
    """XsimBackend._make_buffer_reader parses SHM buffer descriptors."""

    def _make_backend_with_shm(self, shm_image: bytearray):
        """Create XsimBackend with a mock SHM segment."""
        from unittest.mock import MagicMock

        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config={
            "backend": {"xsim": {"vivado_path": "", "timeout_ms": 1000}},
        })
        # Mock SHM with buf attribute
        mock_shm = MagicMock()
        mock_shm.buf = shm_image
        backend._shm = mock_shm
        return backend

    def test_reader_returns_none_without_shm(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config={
            "backend": {"xsim": {"vivado_path": ""}},
        })
        assert backend._make_buffer_reader() is None

    def test_single_buffer_readback(self):
        data = b"\x01\x02\x03\x04"
        image = _build_shm_with_buffers({0: data})
        backend = self._make_backend_with_shm(image)
        reader = backend._make_buffer_reader()
        assert reader is not None
        assert reader(0) == data

    def test_multiple_buffers(self):
        buf0 = b"\xAA" * 16
        buf1 = b"\xBB" * 32
        buf2 = b"\xCC" * 8
        image = _build_shm_with_buffers({0: buf0, 1: buf1, 2: buf2})
        backend = self._make_backend_with_shm(image)
        reader = backend._make_buffer_reader()
        assert reader(0) == buf0
        assert reader(1) == buf1
        assert reader(2) == buf2

    def test_unknown_buffer_id_returns_empty(self):
        image = _build_shm_with_buffers({0: b"\x01"})
        backend = self._make_backend_with_shm(image)
        reader = backend._make_buffer_reader()
        assert reader(99) == b""

    def test_large_buffer(self):
        """256-byte buffer (simulating a serialized tensor)."""
        data = bytes(range(256))
        image = _build_shm_with_buffers({0: data})
        backend = self._make_backend_with_shm(image)
        reader = backend._make_buffer_reader()
        assert reader(0) == data

    def test_non_sequential_buffer_ids(self):
        """Buffer IDs don't have to be sequential."""
        buf5 = b"\x55" * 4
        buf10 = b"\xAA" * 8
        image = _build_shm_with_buffers({5: buf5, 10: buf10})
        backend = self._make_backend_with_shm(image)
        reader = backend._make_buffer_reader()
        assert reader(5) == buf5
        assert reader(10) == buf10
        assert reader(0) == b""

    def test_reader_captures_shm_ref(self):
        """Reader closure captures SHM, so data is live until cleanup."""
        data = b"\xDE\xAD"
        image = _build_shm_with_buffers({0: data})
        backend = self._make_backend_with_shm(image)
        reader = backend._make_buffer_reader()
        # Modify the data in-place (simulating DUT writing output)
        from vten.runtime.shm import BUF_DESC_SIZE, CONTROL_SIZE, CMD_SLOT_SIZE, STATS_SLOT_SIZE
        data_region = struct.unpack_from("<Q", image, 0x30)[0]
        buf_desc_off = struct.unpack_from("<Q", image, 0x28)[0]
        buf_data_off = struct.unpack_from("<Q", image, buf_desc_off + 0x08)[0]
        start = data_region + buf_data_off
        image[start:start + 2] = b"\xBE\xEF"
        assert reader(0) == b"\xBE\xEF"


# ═══════════════════════════════════════════════════════════════════
# §4  ExecutionContext verification — context.py
# ═══════════════════════════════════════════════════════════════════


class TestVerificationCompare:
    """ExecutionContext._compare and _max_diff static methods."""

    def test_compare_equal_int_tensors(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 3])
        assert ExecutionContext._compare(a, b) is True

    def test_compare_unequal_int_tensors(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 4])
        assert ExecutionContext._compare(a, b) is False

    def test_compare_float_within_tolerance(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0 + 1e-7])
        assert ExecutionContext._compare(a, b) is True

    def test_compare_float_outside_tolerance(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 4.0])
        assert ExecutionContext._compare(a, b) is False

    def test_compare_shape_mismatch(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([[1, 2, 3]])
        assert ExecutionContext._compare(a, b) is False

    def test_max_diff_zero(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 3])
        assert ExecutionContext._max_diff(a, b) == 0.0

    def test_max_diff_nonzero(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1, 2, 10])
        b = torch.tensor([1, 2, 3])
        assert ExecutionContext._max_diff(a, b) == 7.0

    def test_max_diff_float(self):
        from vten.runtime.context import ExecutionContext

        a = torch.tensor([1.0, 2.5])
        b = torch.tensor([1.0, 2.0])
        assert abs(ExecutionContext._max_diff(a, b) - 0.5) < 1e-6


class TestAutoVerification:
    """Auto-verification via ctx.run(verify=True)."""

    def test_run_verify_true_triggers_auto_verify(self):
        """ctx.run(verify=True) calls _auto_verify_all internally."""
        from vten.runtime.context import ExecutionContext
        from vten.kernel.base import Kernel
        from vten.kernel.tensor import Tensor
        from vten.spec.models import InterfaceSpec, KernelSpec, PackingScheme, Protocol

        class TinyKernel(Kernel):
            x = Tensor(shape=(4,), dtype=torch.int8, interface="axis_in")
            y = Tensor(shape=(4,), dtype=torch.int8, interface="axis_out")
            def forward(self, **inputs):
                return {"y": self.x.data.clone()}

        spec = KernelSpec(
            kernel_name="tiny",
            rtl_top="rtl/tiny.sv",
            interfaces={
                "axis_in": InterfaceSpec(
                    name="axis_in", rtl_port="s_axis", protocol=Protocol.AXI4S,
                    tensor="x",
                    packing=PackingScheme(element_width=8, elements_per_beat=4),
                ),
                "axis_out": InterfaceSpec(
                    name="axis_out", rtl_port="m_axis", protocol=Protocol.AXI4S,
                    tensor="y",
                    packing=PackingScheme(element_width=8, elements_per_beat=4),
                ),
            },
        )

        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(TinyKernel, spec=spec)
        inst.x.data = torch.tensor([10, 20, 30, 40], dtype=torch.int8)
        ctx.push_tensor(inst.x)
        ctx.pull_tensor(inst.y)

        # Without backend, run(verify=True) should not raise
        result = ctx.run(verify=True)
        assert result.status == "DONE"

    def test_verifications_list_empty_by_default(self):
        """_verifications list is empty (no deferred verify API)."""
        from vten.runtime.context import ExecutionContext

        ctx = ExecutionContext()
        assert ctx._verifications == []


# ═══════════════════════════════════════════════════════════════════
# §5  BackendResult in wait() — integration
# ═══════════════════════════════════════════════════════════════════


class TestXsimWaitReturnsReader:
    """SimBackend._wait_completion() returns BackendResult with functional _shm_reader."""

    def test_wait_result_has_shm_reader(self):
        """After _wait_completion(), BackendResult._shm_reader is set."""
        from unittest.mock import MagicMock

        from vten.backend.xsim import XsimBackend
        from vten.runtime.shm import BACKEND_STATUS_DONE

        # Build SHM image with a buffer
        data = b"\x42" * 8
        image = _build_shm_with_buffers({0: data}, num_commands=1)
        # Set backend_status = DONE
        struct.pack_into("<I", image, 0x0C, BACKEND_STATUS_DONE)

        backend = XsimBackend(project_config={
            "backend": {"xsim": {"vivado_path": "", "timeout_ms": 1000}},
        })
        mock_shm = MagicMock()
        mock_shm.buf = image
        backend._shm = mock_shm

        # Mock semaphore to return immediately
        mock_sem = MagicMock()
        mock_sem.timedwait.return_value = True
        backend._sem_b2h = mock_sem

        result = backend._wait_completion()
        assert result._shm_reader is not None
        assert result.read_buffer(0) == data

    def test_wait_stub_mode_no_reader(self):
        """When SHM is None (stub mode), _shm_reader stays None."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config={
            "backend": {"xsim": {"vivado_path": ""}},
        })
        result = backend._wait_completion()
        assert result.read_buffer(0) == b""


# ═══════════════════════════════════════════════════════════════════
# §6  CLI run_test verification tracking
# ═══════════════════════════════════════════════════════════════════


class TestRunTestVerification:
    """run_test() properly catches VerificationError and tracks results."""

    def test_verification_error_sets_fail(self, tmp_path):
        """VerificationError during run causes FAIL status."""
        from vten.cli.run import run_test, TestScenario

        # Create minimal project structure
        kernel_dir = tmp_path / "kernels" / "test_k"
        tests_dir = kernel_dir / "tests"
        tests_dir.mkdir(parents=True)
        results_dir = tmp_path / "results" / "test_k" / "test_verify"
        (tmp_path / "vten.toml").write_text('[project]\nname = "t"\nversion = "0.1"\n')

        # Create kernel_spec.yaml
        (kernel_dir / "kernel_spec.yaml").write_text("""
kernel: test_k
rtl_top: test_k
interfaces:
  data_in:
    rtl_port: s_axis
    protocol: axi4_stream
    data_width: 32
    packing:
      element_width: 8
      elements_per_beat: 4
""")

        # Create test that raises VerificationError
        (tests_dir / "test_verify.py").write_text("""
from vten.cli.run import TestScenario
from vten.errors import VerificationError

class TestVerify(TestScenario):
    kernel = "test_k"
    def run(self, ctx, cfg):
        raise VerificationError(tensor="ofm", shape=(4,), max_diff=1.0)
""")

        run_test(
            project_dir=str(tmp_path),
            kernel_name="test_k",
            test_name="test_verify",
        )

        import json
        summary = json.loads((results_dir / "summary.json").read_text())
        assert summary["status"] == "FAIL"
