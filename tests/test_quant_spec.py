"""Tests for QuantSpec — declarative quantization on kernel interfaces (M1.2).

Covers:
  - QuantSpec dataclass: derived helpers (qmin/qmax/lsb) + every validation
    error path (bits, signed, frac_bits, overflow, rounding, affine-xor-qformat)
  - kernel_spec.yaml ``quant:`` block parsing round-trip + parser error paths
    (missing keys, unknown keys, bits != element_width, packing requirements)
  - Stage-0 cross-check of quant.signed against the kernel Tensor torch dtype
  - Serializer signedness: a declared quant.signed overrides the legacy
    torch-dtype inference on deserialize (fast and slow paths)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from vten.errors import SpecValidationError
from vten.kernel.tensor import Tensor
from vten.runtime.flatten import check_quant_dtype
from vten.runtime.serializer import StreamSerializer
from vten.spec.models import (
    InterfaceSpec,
    KernelSpec,
    PackingScheme,
    Protocol,
    QuantSpec,
)
from vten.spec.parser import parse_kernel_spec


# ── Helpers ────────────────────────────────────────────────────────


def _write_spec(tmp_path: Path, data: dict, name: str = "test.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return p


def _spec_with_quant(
    quant: dict | object | None,
    packing: dict | None = None,
) -> dict:
    """Minimal single-interface spec dict with an optional quant block."""
    iface: dict = {
        "rtl_port": "s_axis",
        "protocol": "axi4_stream",
        "tensor": "data_in",
    }
    if packing is not None:
        iface["packing"] = packing
    elif packing is None and quant is not None:
        iface["packing"] = {"element_width": 8, "elements_per_beat": 4}
    if quant is not None:
        iface["quant"] = quant
    return {
        "kernel": "quant_test",
        "rtl_top": "rtl/q.sv",
        "interfaces": {"axis_in": iface},
    }


# ═══════════════════════════════════════════════════════════════════
# §1  QuantSpec dataclass — defaults, derived helpers
# ═══════════════════════════════════════════════════════════════════


class TestQuantSpecModel:

    def test_defaults(self):
        qs = QuantSpec(bits=8, signed=True)
        assert qs.frac_bits == 0
        assert qs.overflow == "saturate"
        assert qs.rounding == "trunc"
        assert qs.scale is None
        assert qs.zero_point == 0
        assert not qs.is_affine

    def test_qmin_qmax_signed_8(self):
        qs = QuantSpec(bits=8, signed=True)
        assert qs.qmin == -128
        assert qs.qmax == 127

    def test_qmin_qmax_unsigned_8(self):
        qs = QuantSpec(bits=8, signed=False)
        assert qs.qmin == 0
        assert qs.qmax == 255

    def test_qmin_qmax_signed_4(self):
        qs = QuantSpec(bits=4, signed=True)
        assert qs.qmin == -8
        assert qs.qmax == 7

    def test_qmin_qmax_unsigned_16(self):
        qs = QuantSpec(bits=16, signed=False)
        assert qs.qmin == 0
        assert qs.qmax == 65535

    def test_lsb_integer_default(self):
        assert QuantSpec(bits=8, signed=True).lsb == 1.0

    def test_lsb_qformat(self):
        assert QuantSpec(bits=8, signed=True, frac_bits=4).lsb == 1.0 / 16

    def test_lsb_affine(self):
        qs = QuantSpec(bits=8, signed=False, scale=0.02, zero_point=3)
        assert qs.lsb == 0.02
        assert qs.is_affine

    def test_affine_valid(self):
        qs = QuantSpec(bits=8, signed=True, scale=0.5, zero_point=-1)
        assert qs.scale == 0.5
        assert qs.zero_point == -1


# ═══════════════════════════════════════════════════════════════════
# §2  QuantSpec validation error paths
# ═══════════════════════════════════════════════════════════════════


class TestQuantSpecValidation:

    def test_bits_zero(self):
        with pytest.raises(SpecValidationError, match=r"quant\.bits"):
            QuantSpec(bits=0, signed=True)

    def test_bits_negative(self):
        with pytest.raises(SpecValidationError, match=r"quant\.bits"):
            QuantSpec(bits=-8, signed=True)

    def test_bits_too_large(self):
        with pytest.raises(SpecValidationError, match=r"quant\.bits"):
            QuantSpec(bits=65, signed=True)

    def test_bits_not_int(self):
        with pytest.raises(SpecValidationError, match=r"quant\.bits"):
            QuantSpec(bits="8", signed=True)

    def test_bits_bool_rejected(self):
        with pytest.raises(SpecValidationError, match=r"quant\.bits"):
            QuantSpec(bits=True, signed=True)

    def test_signed_not_bool(self):
        with pytest.raises(SpecValidationError, match=r"quant\.signed"):
            QuantSpec(bits=8, signed=1)

    def test_frac_bits_negative(self):
        with pytest.raises(SpecValidationError, match=r"quant\.frac_bits"):
            QuantSpec(bits=8, signed=True, frac_bits=-1)

    def test_frac_bits_equal_bits(self):
        with pytest.raises(SpecValidationError, match=r"quant\.frac_bits"):
            QuantSpec(bits=8, signed=True, frac_bits=8)

    def test_frac_bits_not_int(self):
        with pytest.raises(SpecValidationError, match=r"quant\.frac_bits"):
            QuantSpec(bits=8, signed=True, frac_bits=1.5)

    def test_frac_bits_max_valid(self):
        qs = QuantSpec(bits=8, signed=True, frac_bits=7)
        assert qs.lsb == 2.0 ** -7

    def test_invalid_overflow(self):
        with pytest.raises(SpecValidationError, match=r"quant\.overflow"):
            QuantSpec(bits=8, signed=True, overflow="clamp")

    def test_invalid_rounding(self):
        with pytest.raises(SpecValidationError, match=r"quant\.rounding"):
            QuantSpec(bits=8, signed=True, rounding="nearest")

    def test_affine_and_qformat_conflict(self):
        with pytest.raises(SpecValidationError, match="ONE quantization model"):
            QuantSpec(bits=8, signed=True, frac_bits=4, scale=0.1)

    def test_zero_point_without_scale(self):
        with pytest.raises(SpecValidationError, match=r"quant\.zero_point"):
            QuantSpec(bits=8, signed=True, zero_point=3)

    def test_scale_nonpositive(self):
        with pytest.raises(SpecValidationError, match=r"quant\.scale"):
            QuantSpec(bits=8, signed=True, scale=0.0)

    def test_scale_not_number(self):
        with pytest.raises(SpecValidationError, match=r"quant\.scale"):
            QuantSpec(bits=8, signed=True, scale="0.5")

    def test_zero_point_out_of_range(self):
        with pytest.raises(SpecValidationError, match=r"quant\.zero_point"):
            QuantSpec(bits=8, signed=True, scale=0.1, zero_point=200)

    def test_zero_point_not_int(self):
        with pytest.raises(SpecValidationError, match=r"quant\.zero_point"):
            QuantSpec(bits=8, signed=True, scale=0.1, zero_point=1.5)


# ═══════════════════════════════════════════════════════════════════
# §3  Cross-check against the kernel Tensor torch dtype (Stage 0)
# ═══════════════════════════════════════════════════════════════════


def _kernel_spec_with_quant(signed: bool) -> KernelSpec:
    packing = PackingScheme(element_width=8, elements_per_beat=4)
    quant = QuantSpec(bits=8, signed=signed)
    packing.signed_override = quant.signed
    iface = InterfaceSpec(
        name="axis_in",
        rtl_port="s_axis",
        protocol=Protocol.AXI4S,
        tensor="data_in",
        packing=packing,
        quant=quant,
    )
    return KernelSpec(
        kernel_name="k", rtl_top="rtl/k.sv", interfaces={"axis_in": iface}
    )


def _tensor(dtype: torch.dtype) -> Tensor:
    t = Tensor(shape=(4,), dtype=dtype, interface="axis_in")
    t.name = "data_in"
    return t


class TestQuantDtypeCrossCheck:

    def test_uint8_dtype_signed_true_raises(self):
        with pytest.raises(SpecValidationError, match="signed=true"):
            check_quant_dtype(_tensor(torch.uint8), _kernel_spec_with_quant(True))

    def test_error_names_interface_and_tensor(self):
        with pytest.raises(
            SpecValidationError, match=r"Interface 'axis_in' \(tensor 'data_in'\)"
        ):
            check_quant_dtype(_tensor(torch.uint8), _kernel_spec_with_quant(True))

    def test_int8_dtype_signed_true_ok(self):
        check_quant_dtype(_tensor(torch.int8), _kernel_spec_with_quant(True))

    def test_uint8_dtype_signed_false_ok(self):
        check_quant_dtype(_tensor(torch.uint8), _kernel_spec_with_quant(False))

    def test_int8_dtype_signed_false_ok(self):
        # A signed container carrying unsigned codes is allowed.
        check_quant_dtype(_tensor(torch.int8), _kernel_spec_with_quant(False))

    def test_no_quant_block_no_check(self):
        spec = _kernel_spec_with_quant(True)
        spec.interfaces["axis_in"].quant = None
        check_quant_dtype(_tensor(torch.uint8), spec)

    def test_unknown_interface_no_check(self):
        t = Tensor(shape=(4,), dtype=torch.uint8, interface="missing")
        t.name = "x"
        check_quant_dtype(t, _kernel_spec_with_quant(True))

    def test_validate_against_dtype_direct(self):
        qs = QuantSpec(bits=8, signed=True)
        with pytest.raises(SpecValidationError, match="unsigned"):
            qs.validate_against_dtype(torch.uint8)
        qs.validate_against_dtype(torch.int8)  # ok


# ═══════════════════════════════════════════════════════════════════
# §4  YAML parsing — quant block round-trip
# ═══════════════════════════════════════════════════════════════════


class TestQuantYamlParsing:

    def test_full_round_trip(self, tmp_path):
        data = _spec_with_quant(
            quant={
                "bits": 8,
                "signed": True,
                "frac_bits": 4,
                "overflow": "wrap",
                "rounding": "half_up",
            },
        )
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        q = spec.interfaces["axis_in"].quant
        assert isinstance(q, QuantSpec)
        assert q.bits == 8
        assert q.signed is True
        assert q.frac_bits == 4
        assert q.overflow == "wrap"
        assert q.rounding == "half_up"
        assert q.scale is None
        assert q.zero_point == 0

    def test_defaults_applied(self, tmp_path):
        data = _spec_with_quant(quant={"bits": 8, "signed": False})
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        q = spec.interfaces["axis_in"].quant
        assert q.frac_bits == 0
        assert q.overflow == "saturate"
        assert q.rounding == "trunc"

    def test_affine_round_trip(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 8, "signed": False, "scale": 0.02, "zero_point": 3},
        )
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        q = spec.interfaces["axis_in"].quant
        assert q.is_affine
        assert q.scale == 0.02
        assert q.zero_point == 3

    def test_signed_override_set_true(self, tmp_path):
        data = _spec_with_quant(quant={"bits": 8, "signed": True})
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.interfaces["axis_in"].packing.signed_override is True

    def test_signed_override_set_false(self, tmp_path):
        data = _spec_with_quant(quant={"bits": 8, "signed": False})
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.interfaces["axis_in"].packing.signed_override is False

    def test_no_quant_block(self, tmp_path):
        data = _spec_with_quant(
            quant=None, packing={"element_width": 8, "elements_per_beat": 4}
        )
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        iface = spec.interfaces["axis_in"]
        assert iface.quant is None
        assert iface.packing.signed_override is None

    def test_wider_element_width(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 16, "signed": True, "frac_bits": 8},
            packing={"element_width": 16, "elements_per_beat": 2},
        )
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.interfaces["axis_in"].quant.bits == 16


# ═══════════════════════════════════════════════════════════════════
# §5  YAML parsing — quant error paths
# ═══════════════════════════════════════════════════════════════════


class TestQuantYamlErrors:

    def _parse(self, tmp_path, data):
        return parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_missing_bits(self, tmp_path):
        data = _spec_with_quant(quant={"signed": True})
        with pytest.raises(SpecValidationError, match=r"quant\.bits is required"):
            self._parse(tmp_path, data)

    def test_missing_signed(self, tmp_path):
        data = _spec_with_quant(quant={"bits": 8})
        with pytest.raises(SpecValidationError, match=r"quant\.signed is required"):
            self._parse(tmp_path, data)

    def test_unknown_key(self, tmp_path):
        data = _spec_with_quant(quant={"bits": 8, "signed": True, "frac": 4})
        with pytest.raises(SpecValidationError, match="unknown quant key"):
            self._parse(tmp_path, data)

    def test_quant_not_mapping(self, tmp_path):
        data = _spec_with_quant(quant="int8")
        with pytest.raises(SpecValidationError, match="must be a mapping"):
            self._parse(tmp_path, data)

    def test_quant_without_packing(self, tmp_path):
        data = _spec_with_quant(quant={"bits": 8, "signed": True})
        del data["interfaces"]["axis_in"]["packing"]
        with pytest.raises(SpecValidationError, match="requires a 'packing'"):
            self._parse(tmp_path, data)

    def test_bits_element_width_mismatch(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 16, "signed": True},
            packing={"element_width": 8, "elements_per_beat": 4},
        )
        with pytest.raises(
            SpecValidationError,
            match=r"quant\.bits \(16\) must equal packing\.element_width \(8\)",
        ):
            self._parse(tmp_path, data)

    def test_quant_with_custom_packing(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 8, "signed": True},
            packing={
                "mode": "custom",
                "fields": [{"name": "data", "bits": [0, 7]}],
            },
        )
        with pytest.raises(SpecValidationError, match="custom packing"):
            self._parse(tmp_path, data)

    def test_invalid_overflow_has_interface_context(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 8, "signed": True, "overflow": "clip"}
        )
        with pytest.raises(
            SpecValidationError, match=r"Interface 'axis_in'.*quant\.overflow"
        ):
            self._parse(tmp_path, data)

    def test_invalid_rounding_via_yaml(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 8, "signed": True, "rounding": "ceil"}
        )
        with pytest.raises(SpecValidationError, match=r"quant\.rounding"):
            self._parse(tmp_path, data)

    def test_frac_bits_out_of_range_via_yaml(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 8, "signed": True, "frac_bits": 8}
        )
        with pytest.raises(SpecValidationError, match=r"quant\.frac_bits"):
            self._parse(tmp_path, data)

    def test_affine_xor_qformat_via_yaml(self, tmp_path):
        data = _spec_with_quant(
            quant={"bits": 8, "signed": True, "frac_bits": 4, "scale": 0.1}
        )
        with pytest.raises(SpecValidationError, match="ONE quantization model"):
            self._parse(tmp_path, data)

    def test_signed_not_bool_via_yaml(self, tmp_path):
        data = _spec_with_quant(quant={"bits": 8, "signed": 1})
        with pytest.raises(SpecValidationError, match=r"quant\.signed"):
            self._parse(tmp_path, data)


# ═══════════════════════════════════════════════════════════════════
# §6  Serializer signedness — declared quant.signed drives deserialize
# ═══════════════════════════════════════════════════════════════════


class TestSerializerSignednessOverride:
    """quant.signed (via PackingScheme.signed_override) beats dtype inference.

    The parser rejects uint dtype + signed:true at Stage 0, so the
    interesting direction is a signed torch dtype whose interface declares
    signed:false — codes must NOT be sign-extended on deserialize.
    """

    def test_fast_path_unsigned_override(self):
        # ew=8 → vectorized fast path. int16 dtype would infer signed.
        packing = PackingScheme(
            element_width=8, elements_per_beat=4, signed_override=False
        )
        s = StreamSerializer(packing)
        raw = s.serialize(torch.tensor([-1, -128, 127, 5], dtype=torch.int8))

        out = s.deserialize(raw, 4, (4,), dtype=torch.int16)
        assert out.tolist() == [255, 128, 127, 5]

        # Fast and slow paths agree under the override.
        fast = s._deserialize_fast(raw, 4, (4,), torch.int16)
        slow = s._deserialize_slow(raw, 4, (4,), torch.int16)
        assert fast is not None
        assert torch.equal(fast, slow)

    def test_fast_path_without_override_sign_extends(self):
        packing = PackingScheme(element_width=8, elements_per_beat=4)
        s = StreamSerializer(packing)
        raw = s.serialize(torch.tensor([-1, -128, 127, 5], dtype=torch.int8))
        out = s.deserialize(raw, 4, (4,), dtype=torch.int16)
        assert out.tolist() == [-1, -128, 127, 5]

    def test_slow_path_unsigned_override(self):
        # ew=12 is not byte-aligned → reference slow path.
        packing = PackingScheme(
            element_width=12, elements_per_beat=2, signed_override=False
        )
        s = StreamSerializer(packing)
        raw = s.serialize(torch.tensor([-1, -2048], dtype=torch.int16))
        out = s.deserialize(raw, 2, (2,), dtype=torch.int16)
        assert out.tolist() == [4095, 2048]

    def test_slow_path_without_override_sign_extends(self):
        packing = PackingScheme(element_width=12, elements_per_beat=2)
        s = StreamSerializer(packing)
        raw = s.serialize(torch.tensor([-1, -2048], dtype=torch.int16))
        out = s.deserialize(raw, 2, (2,), dtype=torch.int16)
        assert out.tolist() == [-1, -2048]

    def test_signed_override_true_forces_sign_extension(self):
        # dtype=None defaults to signed already; use an unsigned dtype
        # container wide enough to hold the sign-extended value check via
        # int32 (no wraparound ambiguity).
        packing = PackingScheme(
            element_width=8, elements_per_beat=4, signed_override=True
        )
        s = StreamSerializer(packing)
        raw = s.serialize(torch.tensor([255, 128, 127, 5], dtype=torch.int16))
        out = s.deserialize(raw, 4, (4,), dtype=torch.int32)
        assert out.tolist() == [-1, -128, 127, 5]

    def test_parsed_spec_end_to_end(self, tmp_path):
        """quant block parsed from YAML → serializer honors signed:false."""
        data = _spec_with_quant(quant={"bits": 8, "signed": False})
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        s = StreamSerializer(spec.interfaces["axis_in"].packing)
        raw = s.serialize(torch.tensor([-1, -2, 3, 4], dtype=torch.int8))
        out = s.deserialize(raw, 4, (4,), dtype=torch.int16)
        assert out.tolist() == [255, 254, 3, 4]
