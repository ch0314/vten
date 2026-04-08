"""Tests for vten.kernel.tensor — Tensor descriptor class.

Spec reference: 00_data_models.md §2 (Tensor)
NPU 3D patterns: npu_3d_analysis.md §8 (Tensor Data Layout)

Tensor shapes from NPU 3D:
  IFM:    (IN_CH, IN_DEPTH, IN_HEIGHT, IN_WIDTH)  — 4D, int8/uint8
  Weight: (OUT_CH, IN_CH, 3, 3, 3)                — 5D, int8
  Bias:   (OUT_CH,)                                — 1D, int32
  OFM:    (OUT_CH, OUT_D, OUT_H, OUT_W)            — 4D, int8/uint8
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from vten.kernel.tensor import Tensor


# ── Resolver stub ──────────────────────────────────────────────────


class _FakeResolver:
    """ParameterResolver stub."""

    def __init__(self, mapping: dict[str, int]):
        self._map = mapping

    def resolve(self, dim):
        if isinstance(dim, int):
            return dim
        key = dim.strip("${}")
        if key in self._map:
            return self._map[key]
        raise KeyError(f"Unresolved parameter: {dim}")


# NPU 3D 파라미터 — U-Net L0: 1→32ch, 3×3×3 conv
_NPU_PARAMS = {
    "IN_CH": 1, "OUT_CH": 32,
    "IN_DEPTH": 16, "IN_HEIGHT": 16, "IN_WIDTH": 16,
}

# U-Net L2: 32→64ch, stride-2 downsample
_NPU_PARAMS_L2 = {
    "IN_CH": 32, "OUT_CH": 64,
    "IN_DEPTH": 16, "IN_HEIGHT": 16, "IN_WIDTH": 16,
}


# ═══════════════════════════════════════════════════════════════════
# §1  Construction — NPU 3D tensor shapes
# ═══════════════════════════════════════════════════════════════════


class TestTensorConstruction:

    def test_ifm_4d_parametric(self):
        """IFM: (IN_CH, D, H, W) — 4D 파라미터 shape."""
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        assert t.shape == ("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}")
        assert t.dtype == torch.int8
        assert t.interface == "ddr"

    def test_weight_5d_mixed(self):
        """Weight: (OUT_CH, IN_CH, 3, 3, 3) — 정수+파라미터 혼합."""
        t = Tensor(
            shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
            dtype=torch.int8, interface="hbm",
        )
        assert t.shape == ("${OUT_CH}", "${IN_CH}", 3, 3, 3)

    def test_bias_1d_int32(self):
        """Bias: (OUT_CH,) — 1D int32 텐서."""
        t = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
        assert t.dtype == torch.int32

    def test_name_defaults_empty(self):
        t = Tensor(shape=(1,), dtype=torch.int8, interface="x")
        assert t.name == ""

    def test_unresolved_initial_state(self):
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        assert t._resolved_shape is None
        assert t._element_count == 0
        assert t.data is None
        assert t._address is None


# ═══════════════════════════════════════════════════════════════════
# §2  _resolve_shape — NPU 3D 파라미터 해석
# ═══════════════════════════════════════════════════════════════════


class TestResolveShape:

    def test_ifm_resolve(self):
        """IFM (1, 16, 16, 16) — U-Net L0."""
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        t._resolve_shape(_FakeResolver(_NPU_PARAMS))
        assert t._resolved_shape == (1, 16, 16, 16)
        assert t._element_count == 1 * 16 * 16 * 16

    def test_weight_resolve(self):
        """Weight (32, 1, 3, 3, 3) — 5D mixed shape."""
        t = Tensor(
            shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
            dtype=torch.int8, interface="hbm",
        )
        t._resolve_shape(_FakeResolver(_NPU_PARAMS))
        assert t._resolved_shape == (32, 1, 3, 3, 3)
        assert t._element_count == 32 * 1 * 27

    def test_bias_resolve(self):
        """Bias (32,) — 1D."""
        t = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
        t._resolve_shape(_FakeResolver(_NPU_PARAMS))
        assert t._resolved_shape == (32,)
        assert t._element_count == 32

    def test_all_int_shape(self):
        """고정 shape — spatial kernel (3,3,3)."""
        t = Tensor(shape=(3, 3, 3), dtype=torch.int8, interface="x")
        t._resolve_shape(_FakeResolver({}))
        assert t._resolved_shape == (3, 3, 3)
        assert t._element_count == 27

    def test_unresolved_param_raises(self):
        t = Tensor(shape=("${UNKNOWN}",), dtype=torch.int8, interface="x")
        with pytest.raises(KeyError):
            t._resolve_shape(_FakeResolver({}))

    def test_large_channel_resolve(self):
        """U-Net bottleneck: 128→320ch."""
        params = {"IN_CH": 128, "OUT_CH": 320, "IN_DEPTH": 4, "IN_HEIGHT": 4, "IN_WIDTH": 4}
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        t._resolve_shape(_FakeResolver(params))
        assert t._resolved_shape == (128, 4, 4, 4)
        assert t._element_count == 128 * 64


# ═══════════════════════════════════════════════════════════════════
# §3  fill_random — NPU 3D dtype patterns
# ═══════════════════════════════════════════════════════════════════


class TestFillRandom:

    def test_fill_random_requires_resolved_shape(self):
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        with pytest.raises(RuntimeError, match="shape not resolved"):
            t.fill_random()

    def test_ifm_int8_fill(self):
        """IFM: int8 — signed [-128, 127]."""
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        t._resolve_shape(_FakeResolver(_NPU_PARAMS))
        t.fill_random()
        assert t.data.shape == (1, 16, 16, 16)
        assert t.data.dtype == torch.int8

    def test_bias_int32_fill(self):
        """Bias: int32 — NPU bias values."""
        t = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
        t._resolve_shape(_FakeResolver({"OUT_CH": 64}))
        t.fill_random()
        assert t.data.shape == (64,)
        assert t.data.dtype == torch.int32

    def test_weight_int8_fill(self):
        """Weight: int8 — 5D tensor."""
        t = Tensor(
            shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
            dtype=torch.int8, interface="hbm",
        )
        t._resolve_shape(_FakeResolver(_NPU_PARAMS_L2))
        t.fill_random()
        assert t.data.shape == (64, 32, 3, 3, 3)
        assert t.data.dtype == torch.int8

    def test_deterministic_with_seed(self):
        """동일 seed → 동일 IFM 데이터."""
        def _make():
            t = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ddr")
            t._resolve_shape(_FakeResolver({"IN_CH": 256}))
            t.fill_random(generator=torch.Generator().manual_seed(42))
            return t.data

        assert torch.equal(_make(), _make())

    def test_different_seeds_differ(self):
        def _make(seed):
            t = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ddr")
            t._resolve_shape(_FakeResolver({"IN_CH": 256}))
            t.fill_random(generator=torch.Generator().manual_seed(seed))
            return t.data

        assert not torch.equal(_make(1), _make(2))


# ═══════════════════════════════════════════════════════════════════
# §4  to_float — golden reference 계산용
# ═══════════════════════════════════════════════════════════════════


class TestToFloat:

    def test_no_data_raises(self):
        t = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
        with pytest.raises(RuntimeError, match="no data"):
            t.to_float()

    def test_int8_ifm_to_float(self):
        """IFM int8 → float32 (F.conv3d 입력용)."""
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        t._resolve_shape(_FakeResolver(_NPU_PARAMS))
        t.fill_random()
        result = t.to_float()
        assert result.dtype == torch.float32
        assert result.shape == (1, 16, 16, 16)

    def test_int32_bias_to_float(self):
        """Bias int32 → float32."""
        t = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
        t._resolve_shape(_FakeResolver({"OUT_CH": 32}))
        t.fill_random()
        result = t.to_float()
        assert result.dtype == torch.float32


# ═══════════════════════════════════════════════════════════════════
# §5  set_address / numel — NPU 3D 메모리 할당
# ═══════════════════════════════════════════════════════════════════


class TestAddressAndNumel:

    def test_set_address_ddr(self):
        """DDR base address (64-bit)."""
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        t.set_address(0x0000_0000_0001_0000)
        assert t._address == 0x0000_0000_0001_0000

    def test_numel_requires_resolved(self):
        t = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ddr")
        with pytest.raises(RuntimeError, match="shape not resolved"):
            t.numel()

    def test_numel_ifm(self):
        """IFM numel: 1 × 16 × 16 × 16 = 4096."""
        t = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        t._resolve_shape(_FakeResolver(_NPU_PARAMS))
        assert t.numel() == 4096

    def test_numel_weight_5d(self):
        """Weight numel: 64 × 32 × 3 × 3 × 3 = 55296."""
        t = Tensor(
            shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
            dtype=torch.int8, interface="hbm",
        )
        t._resolve_shape(_FakeResolver(_NPU_PARAMS_L2))
        assert t.numel() == 64 * 32 * 27


# ═══════════════════════════════════════════════════════════════════
# §6  Instance isolation — copy.copy (KernelInstance.initialize 패턴)
# ═══════════════════════════════════════════════════════════════════


class TestShallowCopy:
    """copy.copy — KernelInstance가 Tensor descriptor를 복사할 때 사용."""

    def test_shallow_copy_shares_immutable(self):
        """shape, dtype, interface, name — 공유."""
        orig = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        orig.name = "ifm"
        clone = copy.copy(orig)

        assert clone.shape is orig.shape
        assert clone.dtype is orig.dtype
        assert clone.interface is orig.interface
        assert clone.name == orig.name

    def test_shallow_copy_independent_mutable(self):
        """data, _resolved_shape, _address — 독립."""
        orig = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        orig._resolve_shape(_FakeResolver(_NPU_PARAMS))
        orig.fill_random()
        orig.set_address(0x1_0000)

        clone = copy.copy(orig)
        clone._resolved_shape = None
        clone._element_count = 0
        clone.data = None
        clone._address = None

        # orig 무영향
        assert orig._resolved_shape == (1, 16, 16, 16)
        assert orig.data is not None
        assert orig._address == 0x1_0000

    def test_two_kernel_instances_independent(self):
        """두 커널 인스턴스가 같은 descriptor에서 복사되어도 독립 resolve."""
        desc = Tensor(
            shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr",
        )
        desc.name = "bias"

        inst1 = copy.copy(desc)
        inst1._resolve_shape(_FakeResolver({"OUT_CH": 32}))
        inst1.fill_random()

        inst2 = copy.copy(desc)
        inst2._resolve_shape(_FakeResolver({"OUT_CH": 128}))
        inst2.fill_random()

        assert inst1._resolved_shape == (32,)
        assert inst2._resolved_shape == (128,)
        assert inst1.data.shape != inst2.data.shape


# ═══════════════════════════════════════════════════════════════════
# §7  Verification state — golden, verified, max_diff, verify()
# ═══════════════════════════════════════════════════════════════════


class TestTensorVerification:

    def _make_tensor(self, shape=(4,), dtype=torch.int8) -> Tensor:
        t = Tensor(shape=shape, dtype=dtype, interface="ddr")
        t.name = "ofm"
        t._resolve_shape(_FakeResolver({}))
        return t

    def test_golden_field_default_none(self):
        t = self._make_tensor()
        assert t.golden is None
        assert t.verified is False
        assert t.max_diff == 0.0

    def test_golden_field_set_get(self):
        t = self._make_tensor()
        golden = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        t.golden = golden
        assert torch.equal(t.golden, golden)

    def test_verify_pass(self):
        """Matching data → verified=True, max_diff=0."""
        t = self._make_tensor()
        data = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        t.data = data.clone()
        t.golden = data.clone()
        t.verify()
        assert t.verified is True
        assert t.max_diff == 0.0

    def test_verify_with_explicit_golden(self):
        """Pass golden to verify() directly."""
        t = self._make_tensor()
        data = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        t.data = data.clone()
        t.verify(golden=data.clone())
        assert t.verified is True
        assert torch.equal(t.golden, data)

    def test_verify_fail_raises(self):
        """Mismatching data raises VerificationError."""
        from vten.errors import VerificationError

        t = self._make_tensor()
        t.data = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        t.golden = torch.tensor([1, 2, 3, 99], dtype=torch.int8)
        with pytest.raises(VerificationError, match="ofm"):
            t.verify()
        # max_diff should still be set even though verify raised
        assert t.max_diff == pytest.approx(95.0)
        assert t.verified is True

    def test_verify_no_golden_raises(self):
        """No golden → VerificationError."""
        from vten.errors import VerificationError

        t = self._make_tensor()
        t.data = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        with pytest.raises(VerificationError, match="no golden"):
            t.verify()

    def test_verify_no_data_raises(self):
        """No data and no BO → VerificationError."""
        from vten.errors import VerificationError

        t = self._make_tensor()
        t.golden = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        with pytest.raises(VerificationError, match="no data"):
            t.verify()

    def test_verify_float_tolerance(self):
        """Float tensors use tolerance-based comparison."""
        t = self._make_tensor(shape=(4,), dtype=torch.float32)
        t.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t.golden = torch.tensor([1.0, 2.0, 3.0, 4.0 + 1e-8])
        t.verify()
        assert t.verified is True
        assert t.max_diff < 1e-6
