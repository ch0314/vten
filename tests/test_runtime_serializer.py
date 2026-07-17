"""Runtime pipeline tests — Stage 3: Tensor Serialization.

Spec reference: 02_runtime_engine.md §8, 00_data_models.md §6.1
NPU 3D patterns: npu_3d_analysis.md §5

Tests StreamSerializer (serialize/deserialize), MultiPortSerializer,
and beat_index_to_coords utility.
"""

from __future__ import annotations

import math
import struct

import pytest
import torch


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def serializer_cls():
    from vten.runtime.serializer import StreamSerializer
    return StreamSerializer


@pytest.fixture()
def packing_cls():
    from vten.spec.models import PackingScheme
    return PackingScheme


@pytest.fixture()
def multi_port_cls():
    from vten.runtime.serializer import MultiPortSerializer
    return MultiPortSerializer


def _make_packing(packing_cls, element_width=8, elements_per_beat=32,
                  bit_order="lsb_first", byte_order="little",
                  alignment="packed"):
    return packing_cls(
        element_width=element_width,
        elements_per_beat=elements_per_beat,
        bit_order=bit_order,
        byte_order=byte_order,
        alignment=alignment,
    )


# ═══════════════════════════════════════════════════════════════════
# §8.2 — Basic serialize: C-contiguous, LSB-first
# ═══════════════════════════════════════════════════════════════════


class TestBasicSerialize:
    """StreamSerializer.serialize() produces correct byte streams."""

    def test_single_beat_int8(self, serializer_cls, packing_cls):
        """4 elements, 8-bit each = 32 bits = 4 bytes per beat."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        data = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        result = s.serialize(data)
        # LSB-first: elem0 at bits[7:0], elem1 at [15:8], etc.
        expected = struct.pack("<I", 0x04030201)
        assert result == expected

    def test_two_beats_int8(self, serializer_cls, packing_cls):
        """8 elements / 4 per beat = 2 beats."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int8)
        result = s.serialize(data)
        assert len(result) == 8  # 2 beats × 4 bytes
        # First beat
        assert result[:4] == struct.pack("<I", 0x04030201)
        # Second beat
        assert result[4:] == struct.pack("<I", 0x08070605)

    def test_npu_3d_int8_256bit(self, serializer_cls, packing_cls):
        """NPU 3D: 32 int8 elements per 256-bit beat."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        s = serializer_cls(packing)
        data = torch.arange(32, dtype=torch.int8)
        result = s.serialize(data)
        assert len(result) == 32  # 256 bits = 32 bytes
        # Element 0 at byte 0 (LSB-first)
        assert result[0] == 0

    def test_npu_3d_int32_bias(self, serializer_cls, packing_cls):
        """NPU bias: 8 int32 elements per 256-bit beat."""
        packing = _make_packing(packing_cls, element_width=32,
                                elements_per_beat=8)
        s = serializer_cls(packing)
        data = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800],
                            dtype=torch.int32)
        result = s.serialize(data)
        assert len(result) == 32  # 8 × 32-bit = 256-bit = 32 bytes

    def test_partial_last_beat(self, serializer_cls, packing_cls):
        """Last beat may have fewer elements than elements_per_beat."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        data = torch.tensor([10, 20, 30], dtype=torch.int8)
        result = s.serialize(data)
        # 3 elements → 1 beat (padded with zeros)
        assert len(result) == 4
        assert result[0] == 10
        assert result[1] == 20
        assert result[2] == 30
        assert result[3] == 0  # zero-padded


# ═══════════════════════════════════════════════════════════════════
# §8.2 — Signed int8 handling
# ═══════════════════════════════════════════════════════════════════


class TestSignedValues:
    """Negative values are masked to element_width bits."""

    def test_negative_int8(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        data = torch.tensor([-1, -128, 127, 0], dtype=torch.int8)
        result = s.serialize(data)
        # -1 → 0xFF masked to 8 bits
        assert result[0] == 0xFF
        # -128 → 0x80
        assert result[1] == 0x80
        # 127 → 0x7F
        assert result[2] == 0x7F
        assert result[3] == 0x00


# ═══════════════════════════════════════════════════════════════════
# §8.2 — MSB-first bit order
# ═══════════════════════════════════════════════════════════════════


class TestMSBFirst:
    """bit_order='msb_first' places element 0 at MSB."""

    def test_msb_first_4elem(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4, bit_order="msb_first")
        s = serializer_cls(packing)
        data = torch.tensor([0xAA, 0xBB, 0xCC, 0xDD], dtype=torch.uint8)
        result = s.serialize(data)
        # MSB-first: elem0 at highest bits
        # elem0 at [31:24], elem1 at [23:16], elem2 at [15:8], elem3 at [7:0]
        expected = struct.pack("<I", 0xAABBCCDD)
        assert result == expected


# ═══════════════════════════════════════════════════════════════════
# §8.2 — Big-endian byte order
# ═══════════════════════════════════════════════════════════════════


class TestByteOrder:
    """byte_order affects final byte serialization."""

    def test_big_endian(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4, byte_order="big")
        s = serializer_cls(packing)
        data = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        result = s.serialize(data)
        # LSB-first packing: beat_val = 0x04030201
        # Big-endian bytes: 04 03 02 01
        expected = struct.pack(">I", 0x04030201)
        assert result == expected


# ═══════════════════════════════════════════════════════════════════
# §8.2 — Multidimensional tensor (C-contiguous order)
# ═══════════════════════════════════════════════════════════════════


class TestMultidimensional:
    """Tensors are flattened C-contiguous before serialization."""

    def test_2d_tensor(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        data = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
        result = s.serialize(data)
        # flatten() → [1, 2, 3, 4]
        expected = struct.pack("<I", 0x04030201)
        assert result == expected

    def test_npu_3d_tiled_layout(self, serializer_cls, packing_cls):
        """NPU 3D IFM: (D, C_pkt, H, W, Ti) = (1, 1, 2, 2, 32)."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        s = serializer_cls(packing)
        shape = (1, 1, 2, 2, 32)
        data = torch.arange(math.prod(shape), dtype=torch.int8).reshape(shape)
        result = s.serialize(data)
        total_elements = math.prod(shape)  # 128
        num_beats = math.ceil(total_elements / 32)  # 4
        assert len(result) == num_beats * 32


# ═══════════════════════════════════════════════════════════════════
# §8.2 — Deserialize (roundtrip)
# ═══════════════════════════════════════════════════════════════════


class TestDeserialize:
    """serialize → deserialize roundtrip preserves data."""

    def test_roundtrip_int8(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        original = torch.tensor([10, 20, 30, 40], dtype=torch.int8)
        serialized = s.serialize(original)
        restored = s.deserialize(serialized, num_elements=4, shape=(4,))
        assert torch.equal(original, restored)

    def test_roundtrip_int32(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=32,
                                elements_per_beat=8)
        s = serializer_cls(packing)
        original = torch.tensor([100, -200, 300, -400, 500, -600, 700, -800],
                                dtype=torch.int32)
        serialized = s.serialize(original)
        restored = s.deserialize(serialized, num_elements=8, shape=(8,))
        assert torch.equal(original, restored)

    def test_roundtrip_multidim(self, serializer_cls, packing_cls):
        """2D tensor roundtrip with shape restoration."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        original = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]],
                                dtype=torch.int8)
        serialized = s.serialize(original)
        restored = s.deserialize(serialized, num_elements=8, shape=(2, 4))
        assert torch.equal(original, restored)

    def test_roundtrip_npu_3d_256bit(self, serializer_cls, packing_cls):
        """NPU 3D: 64 int8 elements → 2 beats → roundtrip."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        s = serializer_cls(packing)
        original = torch.randint(-128, 127, (64,), dtype=torch.int8)
        serialized = s.serialize(original)
        restored = s.deserialize(serialized, num_elements=64, shape=(64,))
        assert torch.equal(original, restored)


# ═══════════════════════════════════════════════════════════════════
# §8.2 — beat_index_to_coords
# ═══════════════════════════════════════════════════════════════════


class TestBeatIndexToCoords:
    """Utility for debug: beat index → tensor coordinates."""

    def test_first_beat_1d(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        coords = s.beat_index_to_coords(0, (8,))
        assert coords == [(0,), (1,), (2,), (3,)]

    def test_second_beat_1d(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        coords = s.beat_index_to_coords(1, (8,))
        assert coords == [(4,), (5,), (6,), (7,)]

    def test_last_beat_partial(self, serializer_cls, packing_cls):
        """Last beat with fewer elements than elements_per_beat."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        coords = s.beat_index_to_coords(1, (6,))
        # Elements 4, 5 only (total=6)
        assert coords == [(4,), (5,)]

    def test_2d_shape(self, serializer_cls, packing_cls):
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        coords = s.beat_index_to_coords(0, (2, 4))
        # flat indices 0-3 → (0,0),(0,1),(0,2),(0,3)
        assert coords == [(0, 0), (0, 1), (0, 2), (0, 3)]

    def test_npu_3d_tiled_coords(self, serializer_cls, packing_cls):
        """NPU tiled shape: (D, C_pkt, H, W, Ti)."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        s = serializer_cls(packing)
        shape = (1, 1, 2, 2, 32)
        coords = s.beat_index_to_coords(0, shape)
        # First beat: flat indices 0..31 → all have D=0,C=0,H=0,W=0,Ti=0..31
        assert len(coords) == 32
        assert coords[0] == (0, 0, 0, 0, 0)
        assert coords[31] == (0, 0, 0, 0, 31)


# ═══════════════════════════════════════════════════════════════════
# §8.2 — serialized_size calculation
# ═══════════════════════════════════════════════════════════════════


class TestSerializedSize:
    """Verify serialized byte count matches expected."""

    def test_exact_fit(self, serializer_cls, packing_cls):
        """Element count is exact multiple of elements_per_beat."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        s = serializer_cls(packing)
        data = torch.zeros(64, dtype=torch.int8)
        result = s.serialize(data)
        assert len(result) == 64  # 2 beats × 32 bytes

    def test_non_exact_fit(self, serializer_cls, packing_cls):
        """Element count not multiple — last beat zero-padded."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        s = serializer_cls(packing)
        data = torch.zeros(33, dtype=torch.int8)
        result = s.serialize(data)
        # 33 elements → 2 beats → 64 bytes
        assert len(result) == 64

    def test_npu_3d_ifm_size(self, serializer_cls, packing_cls):
        """NPU IFM: (4, 2, 8, 8, 32) int8 → bytes."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        s = serializer_cls(packing)
        total_elements = 4 * 2 * 8 * 8 * 32  # 16384
        data = torch.zeros(total_elements, dtype=torch.int8)
        result = s.serialize(data)
        num_beats = total_elements // 32  # 512
        assert len(result) == num_beats * 32  # 16384 bytes


# ═══════════════════════════════════════════════════════════════════
# §8.3 — MultiPortSerializer (split interface)
# ═══════════════════════════════════════════════════════════════════


class TestMultiPortSerializer:
    """Split tensor data across multiple ports (HBM banks)."""

    def test_channel_interleave_2_ports(self, multi_port_cls, packing_cls):
        from vten.spec.models import SplitSpec, PortDef, InterleaveSpec

        splitter = multi_port_cls()
        split_spec = SplitSpec(
            mode="channel_interleave",
            ports=[PortDef(name="port0", base_addr=0),
                   PortDef(name="port1", base_addr=0)],
            interleave=InterleaveSpec(unit=32),
        )
        # 128 bytes → alternating 32-byte chunks
        data = bytes(range(128))
        result = splitter.split_tensor(data, split_spec)
        assert len(result) == 2
        assert len(result["port0"]) == 64
        assert len(result["port1"]) == 64
        # port0 gets chunks 0,2 (bytes 0-31, 64-95)
        assert result["port0"][:32] == data[0:32]
        assert result["port0"][32:64] == data[64:96]

    def test_npu_3d_32_port_hbm(self, multi_port_cls, packing_cls):
        """NPU 3D weight: 32 HBM ports, channel interleave."""
        from vten.spec.models import SplitSpec, PortDef, InterleaveSpec

        splitter = multi_port_cls()
        ports = [PortDef(name=f"hbm_m{i:02d}_axi", base_addr=0)
                 for i in range(32)]
        split_spec = SplitSpec(
            mode="channel_interleave",
            ports=ports,
            interleave=InterleaveSpec(unit=32),
        )
        # 1024 bytes = 32 chunks × 32 bytes → each port gets 1 chunk
        data = bytes(range(256)) * 4  # 1024 bytes
        result = splitter.split_tensor(data, split_spec)
        assert len(result) == 32
        for port in ports:
            assert len(result[port.name]) == 32  # 1024/32 = 32 bytes each

    def test_block_split_2_ports(self, multi_port_cls, packing_cls):
        """Block split: first half to port0, second half to port1."""
        from vten.spec.models import SplitSpec, PortDef

        splitter = multi_port_cls()
        split_spec = SplitSpec(
            mode="block_split",
            ports=[PortDef(name="ddr0", base_addr=0),
                   PortDef(name="ddr1", base_addr=0)],
        )
        data = bytes(range(64))
        result = splitter.split_tensor(data, split_spec)
        assert len(result) == 2
        assert result["ddr0"] == data[:32]
        assert result["ddr1"] == data[32:]


# ═══════════════════════════════════════════════════════════════════
# §8.1 — Direction inference (serialized vs allocated-only)
# ═══════════════════════════════════════════════════════════════════


class TestDirectionHandling:
    """HOST_TO_DEV tensors are serialized; DEV_TO_HOST only get size."""

    def test_host_to_dev_has_data(self, serializer_cls, packing_cls):
        """Input tensor: serialize() returns bytes."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=4)
        s = serializer_cls(packing)
        data = torch.ones(4, dtype=torch.int8)
        result = s.serialize(data)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_size_calculation(self, packing_cls):
        """Output tensor: size = num_beats × bytes_per_beat."""
        packing = _make_packing(packing_cls, element_width=8,
                                elements_per_beat=32)
        element_count = 128
        num_beats = math.ceil(element_count / packing.elements_per_beat)
        expected_size = num_beats * (packing.bus_width // 8)
        assert expected_size == 128  # 4 beats × 32 bytes


# ═══════════════════════════════════════════════════════════════════
# bus_width property
# ═══════════════════════════════════════════════════════════════════


class TestBusWidth:
    """PackingScheme.bus_width derived property."""

    def test_packed_bus_width(self, packing_cls):
        p = _make_packing(packing_cls, element_width=8,
                          elements_per_beat=32)
        assert p.bus_width == 256

    def test_int32_bus_width(self, packing_cls):
        p = _make_packing(packing_cls, element_width=32,
                          elements_per_beat=8)
        assert p.bus_width == 256

    def test_small_bus(self, packing_cls):
        p = _make_packing(packing_cls, element_width=8,
                          elements_per_beat=4)
        assert p.bus_width == 32


# ═══════════════════════════════════════════════════════════════════
# §9  CustomFieldSerializer — custom packing mode
# ═══════════════════════════════════════════════════════════════════


class TestCustomFieldSerializer:
    """Custom packing: named fields at specific bit positions."""

    @staticmethod
    def _custom_packing():
        from vten.spec.models import CustomField, PackingScheme
        return PackingScheme(
            element_width=0, elements_per_beat=0,
            mode="custom",
            custom_fields=[
                CustomField(name="data_a", bits=(0, 23)),    # 24 bits
                CustomField(name="data_b", bits=(24, 47)),   # 24 bits
                CustomField(name="valid_mask", bits=(48, 49)),  # 2 bits
                CustomField(name="reserved", bits=(50, 63)),    # 14 bits
            ],
        )

    @staticmethod
    def _simple_packing():
        from vten.spec.models import CustomField, PackingScheme
        return PackingScheme(
            element_width=0, elements_per_beat=0,
            mode="custom",
            custom_fields=[
                CustomField(name="lo", bits=(0, 7)),
                CustomField(name="hi", bits=(8, 15)),
            ],
        )

    def test_serialize_single_beat(self):
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._simple_packing())
        result = ser.serialize_beat({"lo": 0xAB, "hi": 0xCD})
        assert len(result) == 2
        val = int.from_bytes(result, "little")
        assert val & 0xFF == 0xAB
        assert (val >> 8) & 0xFF == 0xCD

    def test_deserialize_single_beat(self):
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._simple_packing())
        raw = (0xCD << 8 | 0xAB).to_bytes(2, "little")
        fields = ser.deserialize_beat(raw)
        assert fields["lo"] == 0xAB
        assert fields["hi"] == 0xCD

    def test_round_trip_single_beat(self):
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._simple_packing())
        original = {"lo": 42, "hi": 200}
        raw = ser.serialize_beat(original)
        restored = ser.deserialize_beat(raw)
        assert restored == original

    def test_64bit_custom_fields(self):
        """64-bit bus with 4 named fields."""
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._custom_packing())
        beat = {"data_a": 0x123456, "data_b": 0xABCDEF, "valid_mask": 3, "reserved": 0}
        raw = ser.serialize_beat(beat)
        assert len(raw) == 8  # 64 bits = 8 bytes
        restored = ser.deserialize_beat(raw)
        assert restored["data_a"] == 0x123456
        assert restored["data_b"] == 0xABCDEF
        assert restored["valid_mask"] == 3

    def test_serialize_multiple_beats(self):
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._simple_packing())
        beats = [{"lo": 1, "hi": 2}, {"lo": 3, "hi": 4}]
        raw = ser.serialize_beats(beats)
        assert len(raw) == 4  # 2 beats × 2 bytes

    def test_deserialize_multiple_beats(self):
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._simple_packing())
        beats = [{"lo": 10, "hi": 20}, {"lo": 30, "hi": 40}]
        raw = ser.serialize_beats(beats)
        restored = ser.deserialize_beats(raw, num_beats=2)
        assert restored == beats

    def test_missing_field_defaults_to_zero(self):
        """Missing fields in input dict default to 0."""
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._simple_packing())
        result = ser.serialize_beat({"lo": 0xFF})
        fields = ser.deserialize_beat(result)
        assert fields["lo"] == 0xFF
        assert fields["hi"] == 0  # default

    def test_field_value_masking(self):
        """Values exceeding field width are masked."""
        from vten.runtime.serializer import CustomFieldSerializer
        ser = CustomFieldSerializer(self._simple_packing())
        result = ser.serialize_beat({"lo": 0x1FF, "hi": 0})  # 9 bits → mask to 8
        fields = ser.deserialize_beat(result)
        assert fields["lo"] == 0xFF  # masked to 8 bits

    def test_requires_custom_mode(self):
        """Non-custom packing raises ValueError."""
        from vten.runtime.serializer import CustomFieldSerializer
        from vten.spec.models import PackingScheme
        packing = PackingScheme(element_width=8, elements_per_beat=4)
        with pytest.raises(ValueError, match="custom"):
            CustomFieldSerializer(packing)

    def test_bus_width_matches_field_range(self):
        """bus_width computed from custom fields' max bit position."""
        packing = self._custom_packing()
        assert packing.bus_width == 64  # bits 0-63


# ═══════════════════════════════════════════════════════════════════
# Differential: vectorized fast path == reference slow path
# ═══════════════════════════════════════════════════════════════════

import itertools  # noqa: E402


# Width-matched signed dtype + a bounded, negatives-included value range.
_EW_DTYPE = {
    8: (torch.int8, -128, 128),
    16: (torch.int16, -(2**15), 2**15),
    32: (torch.int32, -(2**31), 2**31),
    64: (torch.int64, -(2**62), 2**62),
}

# Grid: element width × elements/beat × bit order × byte order × padding.
_DIFF_GRID = list(itertools.product(
    (8, 16, 32, 64),          # element_width (byte-aligned fast widths)
    (1, 3, 7, 32),            # elements_per_beat (incl. odd, narrow, wide)
    ("lsb_first", "msb_first"),
    ("little", "big"),
    (0, 8, 16),               # extra padding bits on the bus (>0 → padded beat)
))


def _make_diff_packing(packing_cls, ew, epb, bit_order, byte_order, pad_bits):
    p = packing_cls(
        element_width=ew,
        elements_per_beat=epb,
        bit_order=bit_order,
        byte_order=byte_order,
    )
    if pad_bits:
        # Widen the bus beyond epb*ew → each beat gets trailing zero padding.
        p._explicit_bus_width = ew * epb + pad_bits
    return p


class TestFastVsSlowDifferential:
    """Fast (numpy) and slow (reference loop) paths must agree exactly.

    Sweeps a grid of PackingSchemes with randomized data and asserts, for
    both serialize and deserialize, that the vectorized fast path is (a)
    actually taken and (b) byte/element-identical to the reference loop.
    Also confirms round-trip fidelity for the width-matched signed dtype.
    """

    @pytest.mark.parametrize("ew,epb,bit_order,byte_order,pad_bits", _DIFF_GRID)
    def test_serialize_deserialize_match(
        self, serializer_cls, packing_cls,
        ew, epb, bit_order, byte_order, pad_bits,
    ):
        packing = _make_diff_packing(
            packing_cls, ew, epb, bit_order, byte_order, pad_bits
        )
        s = serializer_cls(packing)
        dtype, lo, hi = _EW_DTYPE[ew]

        gen = torch.Generator().manual_seed(1234 + ew * 131 + epb)
        # Exact multiple of epb, and a partial final beat (remainder < epb).
        n_values = [5 * epb]
        if epb > 1:
            n_values.append(5 * epb + max(1, epb // 2))

        for n in n_values:
            data = torch.randint(
                lo, hi, (n,), dtype=dtype, generator=gen
            )

            # ---- serialize: fast path taken and == slow ----
            fast_bytes = s._serialize_fast(data)
            slow_bytes = s._serialize_slow(data)
            assert fast_bytes is not None, (
                f"fast serialize unexpectedly declined for ew={ew} epb={epb}"
            )
            assert fast_bytes == slow_bytes, (
                f"serialize mismatch ew={ew} epb={epb} {bit_order} "
                f"{byte_order} pad={pad_bits} n={n}"
            )
            assert s.serialize(data) == slow_bytes

            # ---- deserialize (signed): fast path taken and == slow ----
            f_signed = s._deserialize_fast(fast_bytes, n, (n,), dtype)
            sl_signed = s._deserialize_slow(fast_bytes, n, (n,), dtype)
            assert f_signed is not None
            assert torch.equal(f_signed, sl_signed), (
                f"deserialize(signed) mismatch ew={ew} epb={epb} n={n}"
            )
            # Round-trip fidelity (width-matched signed dtype).
            assert torch.equal(f_signed, data)

            # ---- deserialize (unsigned interpretation): fast == slow ----
            u_dtype = torch.uint8 if ew == 8 else getattr(
                torch, f"uint{ew}", None
            )
            if u_dtype is not None:
                f_uns = s._deserialize_fast(fast_bytes, n, (n,), u_dtype)
                sl_uns = s._deserialize_slow(fast_bytes, n, (n,), u_dtype)
                if f_uns is not None:  # ew==64 unsigned falls back by design
                    assert torch.equal(f_uns, sl_uns)
                # Public entry point agrees with the reference regardless.
                assert torch.equal(
                    s.deserialize(fast_bytes, n, (n,), u_dtype), sl_uns
                )

    @pytest.mark.parametrize("ew", (8, 16, 32))
    @pytest.mark.parametrize("bit_order", ("lsb_first", "msb_first"))
    @pytest.mark.parametrize("byte_order", ("little", "big"))
    def test_wide_data_narrow_ew_masking(
        self, serializer_cls, packing_cls, ew, bit_order, byte_order,
    ):
        """int32 data packed into a narrower element width → masking path."""
        packing = _make_diff_packing(
            packing_cls, ew, 5, bit_order, byte_order, pad_bits=0
        )
        s = serializer_cls(packing)
        gen = torch.Generator().manual_seed(99)
        data = torch.randint(
            -(2**31), 2**31, (23,), dtype=torch.int32, generator=gen
        )
        fast_bytes = s._serialize_fast(data)
        slow_bytes = s._serialize_slow(data)
        assert fast_bytes is not None
        assert fast_bytes == slow_bytes

    def test_fallback_for_non_byte_aligned_width(
        self, serializer_cls, packing_cls
    ):
        """Odd (non-byte) element widths must decline the fast path."""
        packing = packing_cls(element_width=12, elements_per_beat=4)
        s = serializer_cls(packing)
        data = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
        assert s._serialize_fast(data) is None
        # Public path still works via the reference loop.
        assert s.serialize(data) == s._serialize_slow(data)
