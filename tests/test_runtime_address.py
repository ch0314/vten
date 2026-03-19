"""Phase 2 tests — Stage 4: Address Allocation.

Spec reference: 02_runtime_engine.md §10, 00_data_models.md §6.5
NPU 3D patterns: npu_3d_analysis.md §6

Tests AddressAllocator: sequential allocation, alignment, memory
region bounds checking, and user-override handling.
"""

from __future__ import annotations

import pytest

from vten.errors import MemoryOverflowError


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def allocator_cls():
    from vten.runtime.address import AddressAllocator
    return AddressAllocator


@pytest.fixture()
def memory_region_cls():
    from vten.spec.models import MemoryRegion
    return MemoryRegion


@pytest.fixture()
def ddr_region(memory_region_cls):
    """NPU 3D DDR: 4GB, 4096-byte alignment."""
    return memory_region_cls(
        name="ddr", base=0x0000_0000,
        size=0x1_0000_0000, alignment=4096,
    )


@pytest.fixture()
def hbm_region(memory_region_cls):
    """NPU 3D HBM per-bank: 4GB, default alignment."""
    return memory_region_cls(
        name="hbm", base=0x0000_0000,
        size=0x1_0000_0000, alignment=4096,
    )


@pytest.fixture()
def small_region(memory_region_cls):
    """Small region for overflow testing."""
    return memory_region_cls(
        name="small", base=0x1000,
        size=0x1000, alignment=64,
    )


# ═══════════════════════════════════════════════════════════════════
# Basic allocation
# ═══════════════════════════════════════════════════════════════════


class TestBasicAllocation:
    """Sequential address allocation with alignment."""

    def test_first_allocation_at_base(self, allocator_cls, ddr_region):
        alloc = allocator_cls(ddr_region)
        addr = alloc.allocate("ifm", size=16384)
        assert addr == 0x0000_0000  # base is already aligned

    def test_sequential_aligned(self, allocator_cls, ddr_region):
        """Second tensor gets next aligned address."""
        alloc = allocator_cls(ddr_region)
        addr1 = alloc.allocate("ifm", size=16384)
        addr2 = alloc.allocate("ofm", size=8192)
        # addr2 = align_up(0 + 16384, 4096)
        assert addr2 >= addr1 + 16384
        assert addr2 % 4096 == 0

    def test_alignment_padding(self, allocator_cls, memory_region_cls):
        """Non-aligned sizes get padding before next allocation."""
        region = memory_region_cls(name="test", base=0, size=0x10000,
                                   alignment=64)
        alloc = allocator_cls(region)
        addr1 = alloc.allocate("t1", size=100)  # 100 is not 64-aligned
        addr2 = alloc.allocate("t2", size=50)
        assert addr1 == 0
        # next_addr after t1 = 100, align_up(100, 64) = 128
        assert addr2 == 128

    def test_three_tensors_npu(self, allocator_cls, ddr_region):
        """NPU 3D: IFM, OFM, concat in same DDR region."""
        alloc = allocator_cls(ddr_region)
        ifm_addr = alloc.allocate("ifm", size=16384)
        ofm_addr = alloc.allocate("ofm", size=8192)
        concat_addr = alloc.allocate("concat", size=4096)
        # All distinct and aligned
        assert ifm_addr % 4096 == 0
        assert ofm_addr % 4096 == 0
        assert concat_addr % 4096 == 0
        # Non-overlapping
        assert ofm_addr >= ifm_addr + 16384
        assert concat_addr >= ofm_addr + 8192


# ═══════════════════════════════════════════════════════════════════
# §10 — Non-zero base address
# ═══════════════════════════════════════════════════════════════════


class TestNonZeroBase:
    """Regions starting at non-zero base."""

    def test_base_offset(self, allocator_cls, memory_region_cls):
        region = memory_region_cls(name="sram", base=0x8000_0000,
                                   size=0x0010_0000, alignment=256)
        alloc = allocator_cls(region)
        addr = alloc.allocate("buf", size=1024)
        assert addr == 0x8000_0000

    def test_sequential_with_offset_base(self, allocator_cls,
                                         memory_region_cls):
        region = memory_region_cls(name="sram", base=0x8000_0000,
                                   size=0x0010_0000, alignment=256)
        alloc = allocator_cls(region)
        a1 = alloc.allocate("a", size=512)
        a2 = alloc.allocate("b", size=512)
        assert a1 == 0x8000_0000
        # next_addr = 0x80000000 + 512 = 0x80000200, already 256-aligned
        assert a2 == 0x8000_0200
        assert a2 % 256 == 0


# ═══════════════════════════════════════════════════════════════════
# §16.2 V8 — Memory overflow
# ═══════════════════════════════════════════════════════════════════


class TestMemoryOverflow:
    """MemoryOverflowError when allocation exceeds region."""

    def test_single_large_tensor(self, allocator_cls, small_region):
        alloc = allocator_cls(small_region)
        with pytest.raises(MemoryOverflowError):
            alloc.allocate("huge", size=0x2000)  # > 0x1000

    def test_cumulative_overflow(self, allocator_cls, small_region):
        alloc = allocator_cls(small_region)
        alloc.allocate("t1", size=0x800)  # fits
        with pytest.raises(MemoryOverflowError):
            alloc.allocate("t2", size=0x900)  # exceeds remaining

    def test_alignment_causes_overflow(self, allocator_cls,
                                       memory_region_cls):
        """Alignment padding pushes allocation beyond region."""
        region = memory_region_cls(name="tight", base=0,
                                   size=200, alignment=128)
        alloc = allocator_cls(region)
        alloc.allocate("t1", size=100)
        # next_addr=100, align_up(100,128)=128, 128+100=228 > 200
        with pytest.raises(MemoryOverflowError):
            alloc.allocate("t2", size=100)


# ═══════════════════════════════════════════════════════════════════
# §10 — Stream interfaces (no address needed)
# ═══════════════════════════════════════════════════════════════════


class TestStreamInterface:
    """AXI4-Stream tensors skip address allocation."""

    def test_stream_no_memory_region(self):
        """Stream interfaces have no memory_region → skip allocation.
        This is tested at pipeline level, not allocator level."""
        # Allocator is never called for stream interfaces.
        # Just verify the concept: stream interface has memory_region=None.
        from vten.spec.models import InterfaceSpec, Protocol
        iface = InterfaceSpec(
            name="axis_in", rtl_port="s_axis",
            protocol=Protocol.AXI4S,
        )
        assert iface.memory_region is None


# ═══════════════════════════════════════════════════════════════════
# NPU 3D realistic scenarios
# ═══════════════════════════════════════════════════════════════════


class TestNPU3DScenarios:
    """Realistic allocation patterns from NPU 3D accelerator."""

    def test_conv3d_layer_ddr_layout(self, allocator_cls, ddr_region):
        """Single conv3d layer: IFM + OFM + concat + bias in DDR."""
        alloc = allocator_cls(ddr_region)
        # IFM: (4, 2, 8, 8, 32) int8 = 16384 bytes
        ifm = alloc.allocate("ifm", 16384)
        # OFM: (4, 2, 8, 8, 32) int8 = 16384 bytes
        ofm = alloc.allocate("ofm", 16384)
        # Bias: 64 int32 = 256 bytes
        bias = alloc.allocate("bias", 256)
        assert ifm == 0
        assert ofm % 4096 == 0
        assert bias % 4096 == 0
        # Verify non-overlapping ranges
        assert ofm >= ifm + 16384
        assert bias >= ofm + 16384

    def test_large_feature_map(self, allocator_cls, ddr_region):
        """Large layer: (16, 8, 32, 32, 32) = 4MB IFM."""
        alloc = allocator_cls(ddr_region)
        size = 16 * 8 * 32 * 32 * 32  # 4,194,304 bytes
        addr = alloc.allocate("large_ifm", size)
        assert addr == 0
        # Should still fit in 4GB DDR
        addr2 = alloc.allocate("large_ofm", size)
        assert addr2 >= size
        assert addr2 % 4096 == 0

    def test_hbm_per_bank_allocation(self, allocator_cls, hbm_region):
        """Each HBM bank gets independent allocation."""
        alloc = allocator_cls(hbm_region)
        # Weight chunk per bank: ~1024 bytes
        addr = alloc.allocate("wgt_bank0", 1024)
        assert addr == 0


# ═══════════════════════════════════════════════════════════════════
# align_up utility
# ═══════════════════════════════════════════════════════════════════


class TestAlignUp:
    """_align_up method behavior."""

    def test_already_aligned(self, allocator_cls, memory_region_cls):
        region = memory_region_cls(name="t", base=0, size=0x10000,
                                   alignment=64)
        alloc = allocator_cls(region)
        assert alloc._align_up(64, 64) == 64
        assert alloc._align_up(128, 64) == 128

    def test_needs_alignment(self, allocator_cls, memory_region_cls):
        region = memory_region_cls(name="t", base=0, size=0x10000,
                                   alignment=64)
        alloc = allocator_cls(region)
        assert alloc._align_up(1, 64) == 64
        assert alloc._align_up(63, 64) == 64
        assert alloc._align_up(65, 64) == 128

    def test_power_of_two_alignment(self, allocator_cls, memory_region_cls):
        region = memory_region_cls(name="t", base=0, size=0x100000,
                                   alignment=4096)
        alloc = allocator_cls(region)
        assert alloc._align_up(1, 4096) == 4096
        assert alloc._align_up(4095, 4096) == 4096
        assert alloc._align_up(4096, 4096) == 4096
        assert alloc._align_up(4097, 4096) == 8192
