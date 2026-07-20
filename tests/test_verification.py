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
    from vten.backend.sim.shm_constants import (
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
        from vten.backend.sim.shm_constants import BUF_DESC_SIZE, CONTROL_SIZE, CMD_SLOT_SIZE, STATS_SLOT_SIZE
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
    """compare() and max_diff() from vten.verifier."""

    def test_compare_equal_int_tensors(self):
        from vten.verifier import compare

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 3])
        assert compare(a, b) is True

    def test_compare_unequal_int_tensors(self):
        from vten.verifier import compare

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 4])
        assert compare(a, b) is False

    def test_compare_float_within_tolerance(self):
        from vten.verifier import compare

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0 + 1e-7])
        assert compare(a, b) is True

    def test_compare_float_outside_tolerance(self):
        from vten.verifier import compare

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 4.0])
        assert compare(a, b) is False

    def test_compare_shape_mismatch(self):
        from vten.verifier import compare

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([[1, 2, 3]])
        assert compare(a, b) is False

    def test_max_diff_zero(self):
        from vten.verifier import max_diff

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 3])
        assert max_diff(a, b) == 0.0

    def test_max_diff_nonzero(self):
        from vten.verifier import max_diff

        a = torch.tensor([1, 2, 10])
        b = torch.tensor([1, 2, 3])
        assert max_diff(a, b) == 7.0

    def test_max_diff_float(self):
        from vten.verifier import max_diff

        a = torch.tensor([1.0, 2.5])
        b = torch.tensor([1.0, 2.0])
        assert abs(max_diff(a, b) - 0.5) < 1e-6


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


# ═══════════════════════════════════════════════════════════════════
# §4a  Quant-aware comparison — lsb_tol / QuantSpec reporting
# ═══════════════════════════════════════════════════════════════════


class TestCompareLsbTol:
    """compare(lsb_tol=...) — integer LSB tolerance, strictly opt-in."""

    def test_default_rejects_one_lsb_diff(self):
        """Default path unchanged: bit-exact, 1-LSB diff fails."""
        from vten.verifier import compare

        a = torch.tensor([10, 20, 30], dtype=torch.int8)
        b = torch.tensor([11, 20, 30], dtype=torch.int8)
        assert compare(a, b) is False

    def test_explicit_zero_keeps_exact_path(self):
        from vten.verifier import compare

        a = torch.tensor([10, 20, 30], dtype=torch.int8)
        b = torch.tensor([11, 20, 30], dtype=torch.int8)
        assert compare(a, b, lsb_tol=0) is False
        assert compare(a, a.clone(), lsb_tol=0) is True

    def test_tol_one_passes_one_off(self):
        from vten.verifier import compare

        a = torch.tensor([11, 19, 30], dtype=torch.int8)
        b = torch.tensor([10, 20, 30], dtype=torch.int8)
        assert compare(a, b, lsb_tol=1) is True

    def test_tol_one_fails_two_off(self):
        from vten.verifier import compare

        a = torch.tensor([12, 20, 30], dtype=torch.int8)
        b = torch.tensor([10, 20, 30], dtype=torch.int8)
        assert compare(a, b, lsb_tol=1) is False

    def test_tol_two_passes_two_off(self):
        from vten.verifier import compare

        a = torch.tensor([12, 20, 30], dtype=torch.int8)
        b = torch.tensor([10, 20, 30], dtype=torch.int8)
        assert compare(a, b, lsb_tol=2) is True

    def test_signed_wraparound_not_small(self):
        """-128 vs 127 is 255 LSBs apart (plain int64 abs diff), not 1."""
        from vten.verifier import compare

        a = torch.tensor([-128], dtype=torch.int8)
        b = torch.tensor([127], dtype=torch.int8)
        assert compare(a, b, lsb_tol=1) is False
        assert compare(a, b, lsb_tol=254) is False
        assert compare(a, b, lsb_tol=255) is True

    def test_unsigned_wraparound_not_small(self):
        """255 vs 0 (uint8) is 255 LSBs apart, not 1."""
        from vten.verifier import compare

        a = torch.tensor([255], dtype=torch.uint8)
        b = torch.tensor([0], dtype=torch.uint8)
        assert compare(a, b, lsb_tol=1) is False

    def test_float_ignores_lsb_tol(self):
        """lsb_tol applies to integers only; float rule unchanged."""
        from vten.verifier import compare

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 3.0])
        assert compare(a, b, lsb_tol=10) is False

    def test_shape_mismatch_fails_with_tol(self):
        from vten.verifier import compare

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([[1, 2, 3]])
        assert compare(a, b, lsb_tol=5) is False


class TestCheckMatchLsbTol:
    """check_match(lsb_tol=...) — pass/fail + max_lsb_err payload."""

    def test_default_raises_on_one_lsb_diff(self):
        """Default path unchanged: bit-exact still fails on 1-LSB diff."""
        from vten.verifier import check_match

        a = torch.tensor([11, 20], dtype=torch.int8)
        b = torch.tensor([10, 20], dtype=torch.int8)
        with pytest.raises(VerificationError):
            check_match("y", a, b)

    def test_exact_match_returns_zero(self):
        from vten.verifier import check_match

        a = torch.tensor([10, 20], dtype=torch.int8)
        assert check_match("y", a, a.clone()) == 0
        assert check_match("y", a, a.clone(), lsb_tol=3) == 0

    def test_tol_one_passes_and_returns_max_lsb_err(self):
        """Passing within tolerance records the max LSB error observed."""
        from vten.verifier import check_match

        a = torch.tensor([11, 20, 29], dtype=torch.int8)
        b = torch.tensor([10, 20, 30], dtype=torch.int8)
        assert check_match("y", a, b, lsb_tol=1) == 1

    def test_tol_one_fails_two_off(self):
        from vten.verifier import check_match

        a = torch.tensor([12, 21], dtype=torch.int8)
        b = torch.tensor([10, 20], dtype=torch.int8)
        with pytest.raises(VerificationError) as exc_info:
            check_match("y", a, b, lsb_tol=1)
        e = exc_info.value
        assert e.max_lsb_err == 2
        assert "max_lsb_err=2" in str(e)
        assert "lsb_tol=1" in str(e)

    def test_tol_failure_reports_only_out_of_tol_elements(self):
        """Elements within tolerance are not counted as differing."""
        from vten.verifier import check_match

        a = torch.tensor([12, 21, 30], dtype=torch.int8)  # +2, +1, exact
        b = torch.tensor([10, 20, 30], dtype=torch.int8)
        with pytest.raises(VerificationError) as exc_info:
            check_match("y", a, b, lsb_tol=1)
        assert "1/3 elements differ" in str(exc_info.value)

    def test_default_message_format_unchanged(self):
        """Without opt-in the report keeps the legacy expected=/actual= lines."""
        from vten.verifier import check_match

        a = torch.tensor([11, 20], dtype=torch.int8)
        b = torch.tensor([10, 20], dtype=torch.int8)
        with pytest.raises(VerificationError) as exc_info:
            check_match("y", a, b)
        msg = str(exc_info.value)
        assert "expected=10, actual=11" in msg
        assert "lsb_err" not in msg
        assert "lsb_tol" not in msg

    def test_float_path_unchanged(self):
        from vten.verifier import check_match

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0 + 1e-7])
        assert check_match("y", a, b) == 0
        with pytest.raises(VerificationError) as exc_info:
            check_match("y", a, torch.tensor([1.0, 3.0]))
        assert exc_info.value.max_lsb_err == 0


class TestCheckMatchQuantReport:
    """check_match(quant=...) — dequantized-domain report enrichment."""

    def _qspec(self, **kw):
        from vten.spec.models import QuantSpec

        base = {"bits": 16, "signed": True, "frac_bits": 7}
        base.update(kw)
        return QuantSpec(**base)

    def test_quant_alone_never_loosens(self):
        """QuantSpec is reporting-only: bit-exact comparison still fails."""
        from vten.verifier import check_match

        a = torch.tensor([131], dtype=torch.int16)
        b = torch.tensor([130], dtype=torch.int16)
        with pytest.raises(VerificationError):
            check_match("y", a, b, quant=self._qspec())

    def test_mismatch_lines_show_dequantized_and_lsb_err(self):
        """Q8.7: 131 → ≈1.02344, 130 → ≈1.01562, lsb_err=1."""
        from vten.verifier import check_match

        a = torch.tensor([131], dtype=torch.int16)
        b = torch.tensor([130], dtype=torch.int16)
        with pytest.raises(VerificationError) as exc_info:
            check_match("y", a, b, quant=self._qspec())
        msg = str(exc_info.value)
        assert "hw=131 (≈1.02344)" in msg
        assert "golden=130 (≈1.01562)" in msg
        assert "lsb_err=1" in msg
        assert "max_lsb_err=1" in msg

    def test_affine_dequantized_values(self):
        from vten.verifier import check_match

        qs = self._qspec(frac_bits=0, scale=0.5, zero_point=10)
        a = torch.tensor([14], dtype=torch.int16)  # (14-10)*0.5 = 2.0
        b = torch.tensor([12], dtype=torch.int16)  # (12-10)*0.5 = 1.0
        with pytest.raises(VerificationError) as exc_info:
            check_match("y", a, b, quant=qs)
        msg = str(exc_info.value)
        assert "hw=14 (≈2)" in msg
        assert "golden=12 (≈1)" in msg
        assert "lsb_err=2" in msg

    def test_quant_with_tol_passes_and_returns_err(self):
        from vten.verifier import check_match

        a = torch.tensor([131, 128], dtype=torch.int16)
        b = torch.tensor([130, 128], dtype=torch.int16)
        assert check_match("y", a, b, lsb_tol=1, quant=self._qspec()) == 1


def _make_verified_ctx(quant=None):
    """Build a compiled TinyKernel context for _auto_verify_all tests.

    Returns (ctx, inst, compiled). The kernel's forward() is identity, so
    golden y == x; tests doctor inst.y.data to create controlled HW errors.
    """
    from vten.kernel.base import Kernel
    from vten.kernel.tensor import Tensor
    from vten.runtime.context import ExecutionContext
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
                quant=quant,
            ),
        },
    )

    ctx = ExecutionContext(project_params={})
    inst = ctx.instantiate(TinyKernel, spec=spec)
    inst.x.data = torch.tensor([10, 20, 30, 40], dtype=torch.int8)
    ctx.push_tensor(inst.x)
    ctx.pull_tensor(inst.y)
    ctx.run()  # no backend: compiles and stashes _last_compiled
    return ctx, inst, ctx._last_compiled


class TestAutoVerifyLsbTolerance:
    """_auto_verify_all lsb_tolerance plumbing + QuantSpec auto-fetch."""

    def _quant(self):
        from vten.spec.models import QuantSpec

        return QuantSpec(bits=8, signed=True, frac_bits=4)

    def test_default_fails_on_one_lsb(self):
        """Default behavior unchanged: 1-LSB diff still fails."""
        ctx, inst, compiled = _make_verified_ctx()
        inst.y.data = torch.tensor([11, 20, 30, 40], dtype=torch.int8)
        with pytest.raises(VerificationError):
            ctx._auto_verify_all(compiled, {"y": inst.y})

    def test_scalar_tolerance_passes_with_max_lsb_err(self):
        ctx, inst, compiled = _make_verified_ctx()
        inst.y.data = torch.tensor([11, 20, 30, 40], dtype=torch.int8)
        count, results = ctx._auto_verify_all(
            compiled, {"y": inst.y}, lsb_tolerance=1,
        )
        assert count == 1
        assert results[0].passed is True
        assert results[0].max_lsb_err == 1

    def test_dict_tolerance_by_tensor_name(self):
        ctx, inst, compiled = _make_verified_ctx()
        inst.y.data = torch.tensor([11, 20, 30, 40], dtype=torch.int8)
        count, results = ctx._auto_verify_all(
            compiled, {"y": inst.y}, lsb_tolerance={"y": 1},
        )
        assert results[0].passed is True
        assert results[0].max_lsb_err == 1

    def test_dict_tolerance_unlisted_tensor_stays_exact(self):
        ctx, inst, compiled = _make_verified_ctx()
        inst.y.data = torch.tensor([11, 20, 30, 40], dtype=torch.int8)
        with pytest.raises(VerificationError) as exc_info:
            ctx._auto_verify_all(
                compiled, {"y": inst.y}, lsb_tolerance={"other": 5},
            )
        assert exc_info.value.max_lsb_err == 1

    def test_quant_auto_fetched_for_reporting(self):
        """Declared interface quant enriches the report without loosening."""
        ctx, inst, compiled = _make_verified_ctx(quant=self._quant())
        inst.y.data = torch.tensor([11, 20, 30, 40], dtype=torch.int8)
        with pytest.raises(VerificationError) as exc_info:
            ctx._auto_verify_all(compiled, {"y": inst.y})  # no tolerance
        msg = str(exc_info.value)
        # Q4.4 dequantized values: 11/16 = 0.6875, 10/16 = 0.625
        assert "hw=11 (≈0.6875)" in msg
        assert "golden=10 (≈0.625)" in msg
        assert "lsb_err=1" in msg

    def test_failing_result_carries_max_lsb_err(self):
        ctx, inst, compiled = _make_verified_ctx()
        inst.y.data = torch.tensor([13, 20, 30, 40], dtype=torch.int8)
        with pytest.raises(VerificationError) as exc_info:
            ctx._auto_verify_all(compiled, {"y": inst.y}, lsb_tolerance=1)
        results = exc_info.value.context["verification_results"]
        assert results[0].passed is False
        assert results[0].max_lsb_err == 3


# ═══════════════════════════════════════════════════════════════════
# §5  BackendResult in wait() — integration
# ═══════════════════════════════════════════════════════════════════


class TestXsimWaitReturnsReader:
    """SimBackend._wait_completion() returns BackendResult with functional _shm_reader."""

    def test_wait_result_has_shm_reader(self):
        """After _wait_completion(), BackendResult._shm_reader is set."""
        from unittest.mock import MagicMock

        from vten.backend.xsim import XsimBackend
        from vten.backend.sim.shm_constants import BACKEND_STATUS_DONE

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
        """VerificationError in execute_batch → FAIL status in summary."""
        from unittest.mock import MagicMock, patch

        from vten.cli.run import run_test
        from vten.cli.scenario import TestScenario
        from vten.execution import BatchResult, ConfigResult

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

        # Create test scenario
        (tests_dir / "test_verify.py").write_text("""
from vten.cli.scenario import TestScenario

class TestVerify(TestScenario):
    kernel = "test_k"
""")

        # Mock execute_batch to return VerificationError
        fail_batch = BatchResult(configs=[
            ConfigResult(
                config_index=0,
                error=VerificationError(tensor="ofm", shape=(4,), max_diff=1.0),
            ),
        ])

        with patch("vten.cli.run.get_backend") as mock_get_backend, \
             patch("vten.cli.run.execute_batch", return_value=fail_batch), \
             patch.object(TestScenario, "_discover_kernel_class",
                          return_value=MagicMock()):
            mock_backend = MagicMock()
            mock_backend.__exit__ = MagicMock(
                side_effect=lambda *a: mock_backend.cleanup()
            )
            mock_get_backend.return_value = mock_backend
            mock_backend.working_directory.return_value = tmp_path

            run_test(
                project_dir=str(tmp_path),
                kernel_name="test_k",
                test_name="test_verify",
            )

        import json
        summary = json.loads((results_dir / "summary.json").read_text())
        assert summary["status"] == "FAIL"


# ═══════════════════════════════════════════════════════════════════
# §7  Scenario / execute_batch lsb_tolerance plumbing
# ═══════════════════════════════════════════════════════════════════


class TestScenarioLsbTolerancePlumbing:
    """TestScenario.lsb_tolerance (int or dict) flows into execute_batch."""

    def _run_scenario(self, tmp_path, scenario_attrs: str):
        """run_test() a minimal scenario; return the execute_batch mock."""
        from unittest.mock import MagicMock, patch

        from vten.cli.run import run_test
        from vten.cli.scenario import TestScenario
        from vten.execution import BatchResult, ConfigResult

        kernel_dir = tmp_path / "kernels" / "test_k"
        tests_dir = kernel_dir / "tests"
        tests_dir.mkdir(parents=True)
        (tmp_path / "vten.toml").write_text('[project]\nname = "t"\nversion = "0.1"\n')
        (kernel_dir / "kernel_spec.yaml").write_text("""
kernel: test_k
rtl_top: test_k
interfaces:
  data_in:
    rtl_port: s_axis
    protocol: axi4_stream
    packing:
      element_width: 8
      elements_per_beat: 4
""")
        (tests_dir / "test_tol.py").write_text(f"""
from vten.cli.scenario import TestScenario

class TestTol(TestScenario):
    kernel = "test_k"
{scenario_attrs}
""")

        ok_batch = BatchResult(configs=[
            ConfigResult(config_index=0, result=MagicMock(
                status="DONE", total_cycles=0, verification_count=0,
                verification_results=[], per_command_stats=[],
            )),
        ])

        with patch("vten.cli.run.get_backend") as mock_get_backend, \
             patch("vten.cli.run.execute_batch",
                   return_value=ok_batch) as mock_eb, \
             patch.object(TestScenario, "_discover_kernel_class",
                          return_value=MagicMock()):
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend
            mock_backend.working_directory.return_value = tmp_path
            run_test(
                project_dir=str(tmp_path),
                kernel_name="test_k",
                test_name="test_tol",
            )
        return mock_eb

    def test_scenario_default_is_zero(self):
        from vten.cli.scenario import TestScenario

        assert TestScenario().lsb_tolerance == 0

    def test_default_forwards_zero(self, tmp_path):
        mock_eb = self._run_scenario(tmp_path, "")
        assert mock_eb.call_args.kwargs["lsb_tolerance"] == 0

    def test_int_form_forwarded(self, tmp_path):
        mock_eb = self._run_scenario(tmp_path, "    lsb_tolerance = 2\n")
        assert mock_eb.call_args.kwargs["lsb_tolerance"] == 2

    def test_dict_form_forwarded(self, tmp_path):
        mock_eb = self._run_scenario(
            tmp_path, "    lsb_tolerance = {'ofm': 1, 'psum': 2}\n",
        )
        assert mock_eb.call_args.kwargs["lsb_tolerance"] == {"ofm": 1, "psum": 2}


class TestExecuteBatchLsbTolerance:
    """End-to-end execute_batch(lsb_tolerance=...) over a 1-LSB-off backend."""

    def _run(self, **kwargs):
        from vten.backend.cpu import CpuBackend
        from vten.execution import execute_batch
        from vten.kernel.base import Kernel
        from vten.kernel.tensor import Tensor
        from vten.spec.models import (
            InterfaceSpec, KernelSpec, PackingScheme, Protocol,
        )

        class OffByOneBackend(CpuBackend):
            """CPU backend whose outputs are all 1 LSB above golden."""

            def execute(self, compiled):
                result = super().execute(compiled)
                if result._forward_tensors:
                    result._forward_tensors = {
                        k: v + 1 for k, v in result._forward_tensors.items()
                    }
                return result

        class TinyKernel(Kernel):
            x = Tensor(shape=(4,), dtype=torch.int8, interface="axis_in")
            y = Tensor(shape=(4,), dtype=torch.int8, interface="axis_out")

            def forward(self, **inputs):
                return {"y": self.x.data.clone()}

            def run(self, ctx):
                ctx.push_tensor(self.x)
                ctx.pull_tensor(self.y)

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

        return execute_batch(
            backend=OffByOneBackend(),
            kernel_class=TinyKernel,
            configs=[{}],
            spec=spec,
            inputs={"x": torch.tensor([10, 20, 30, 40], dtype=torch.int8)},
            verify=True,
            on_error="raise",
            **kwargs,
        )

    def test_default_bit_exact_fails(self):
        """1-LSB HW error fails without opt-in (default unchanged)."""
        with pytest.raises(VerificationError):
            self._run()

    def test_lsb_tolerance_one_passes(self):
        batch = self._run(lsb_tolerance=1)
        assert batch.all_passed
        results = batch.single().verification_results
        assert results[0].passed is True
        assert results[0].max_lsb_err == 1

    def test_lsb_tolerance_dict_passes(self):
        batch = self._run(lsb_tolerance={"y": 1})
        assert batch.single().verification_results[0].max_lsb_err == 1
