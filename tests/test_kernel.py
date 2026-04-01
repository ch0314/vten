"""Tests for vten.kernel.base — Kernel, RegisterHandle, __init_subclass__.

Spec reference: 00_data_models.md §3 (Kernel), 01_kernel_and_dsl.md §2
NPU 3D patterns: npu_3d_analysis.md §4 (Per-IP Register Maps), §11.1

NPU 3D sub-kernels as unit Kernel examples:
  FmapIOKernel:       3 tensors (ifm, ofm, concat) + 1 register handle
  BiasLoaderKernel:   1 tensor (bias) + 1 register handle
  WeightLoaderKernel: 1 tensor (weight) + 1 register handle
  MacAtuKernel:       3 tensors (ifm, weight, psum) + 1 register handle
  PsumBufferKernel:   0 external tensors + 1 register handle
  ActQuantKernel:     0 external tensors + 1 register handle
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from vten.kernel.tensor import Tensor
from vten.kernel.base import Kernel, RegisterHandle, register


# ── NPU 3D sub-kernel definitions ─────────────────────────────────
# These mirror the actual IP structure from npu_3d_analysis.md §3


class FmapIOKernel(Kernel):
    """fmapIO — IFM/OFM DDR transfer + AXIS internal."""
    spec = "design/fmapIO/rtl/fmapIO_top.yaml"
    ifm = Tensor(
        shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
        dtype=torch.int8, interface="ddr",
    )
    ofm = Tensor(
        shape=("${OUT_CH}", "${OUT_DEPTH}", "${OUT_HEIGHT}", "${OUT_WIDTH}"),
        dtype=torch.int8, interface="ddr",
    )
    concat = Tensor(
        shape=("${CONCAT_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
        dtype=torch.int8, interface="ddr",
    )
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.ifm.fill_random(generator=torch.Generator().manual_seed(seed or 0))

    def forward(self, **inputs):
        return self.ifm.to_float()


class BiasLoaderKernel(Kernel):
    """bias_loader — int32 bias, DDR single port."""
    spec = "design/bias_loader/rtl/bias_loader_top.yaml"
    bias = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.bias.fill_random(generator=torch.Generator().manual_seed(seed or 0))

    def forward(self, **inputs):
        return self.bias.to_float()


class WeightLoaderKernel(Kernel):
    """weight_loader — int8 weight, HBM 32-bank split."""
    spec = "design/weight_loader/rtl/weight_loader_top.yaml"
    weight = Tensor(
        shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
        dtype=torch.int8, interface="hbm",
    )
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.weight.fill_random(generator=torch.Generator().manual_seed(seed or 0))

    def forward(self, **inputs):
        return self.weight.to_float()


class MacAtuKernel(Kernel):
    """mac_atu — AXI4-Lite only, AXIS internal I/O."""
    spec = "design/mac/rtl/mac_atu.yaml"
    ifm = Tensor(
        shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
        dtype=torch.int8, interface="ifm_in",
    )
    weight = Tensor(
        shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
        dtype=torch.int8, interface="wgt_in",
    )
    psum = Tensor(
        shape=("${OUT_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
        dtype=torch.int32, interface="psum_out",
    )
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.ifm.fill_random(generator=torch.Generator().manual_seed(seed or 0))
        self.weight.fill_random(generator=torch.Generator().manual_seed((seed or 0) + 1))

    def forward(self, **inputs):
        return self.ifm.to_float()


class PsumBufferKernel(Kernel):
    """psum_buffer — AXI4-Lite only, no external tensor."""
    spec = "design/psum_buffer/rtl/psum_buffer_top.yaml"
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass

    def forward(self, **inputs):
        return torch.tensor(0)


class ActQuantKernel(Kernel):
    """act_quant — AXI4-Lite only, no external tensor."""
    spec = "design/activation/rtl/act_quant_top.yaml"
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass

    def forward(self, **inputs):
        return torch.tensor(0)


# ═══════════════════════════════════════════════════════════════════
# §1  RegisterHandle — NPU 3D has 6 ctrl interfaces
# ═══════════════════════════════════════════════════════════════════


class TestRegisterHandle:

    def test_register_handle_creation(self):
        rh = RegisterHandle("ctrl")
        assert rh.interface_name == "ctrl"

    def test_register_helper(self):
        rh = register("ctrl")
        assert isinstance(rh, RegisterHandle)
        assert rh.interface_name == "ctrl"


# ═══════════════════════════════════════════════════════════════════
# §2  __init_subclass__ — NPU 3D sub-kernel auto-registration
# ═══════════════════════════════════════════════════════════════════


class TestInitSubclass:

    def test_fmapio_descriptors(self):
        """fmapIO: 3 tensors (ifm, ofm, concat) + 1 register handle."""
        assert set(FmapIOKernel._tensor_descriptors.keys()) == {"ifm", "ofm", "concat"}
        assert "ctrl" in FmapIOKernel._register_handles

    def test_bias_loader_descriptors(self):
        """bias_loader: 1 tensor (bias) + 1 register handle."""
        assert set(BiasLoaderKernel._tensor_descriptors.keys()) == {"bias"}
        assert BiasLoaderKernel._tensor_descriptors["bias"].dtype == torch.int32

    def test_weight_loader_descriptors(self):
        """weight_loader: 1 tensor (weight, 5D) + 1 register handle."""
        assert set(WeightLoaderKernel._tensor_descriptors.keys()) == {"weight"}
        wt = WeightLoaderKernel._tensor_descriptors["weight"]
        assert len(wt.shape) == 5  # (OUT_CH, IN_CH, 3, 3, 3)
        assert wt.interface == "hbm"

    def test_mac_atu_descriptors(self):
        """mac_atu: 3 tensors + 1 register handle."""
        assert set(MacAtuKernel._tensor_descriptors.keys()) == {"ifm", "weight", "psum"}
        assert MacAtuKernel._tensor_descriptors["psum"].dtype == torch.int32

    def test_psum_buffer_no_tensors(self):
        """psum_buffer: no tensors, 1 register handle only."""
        assert len(PsumBufferKernel._tensor_descriptors) == 0
        assert "ctrl" in PsumBufferKernel._register_handles

    def test_tensor_name_auto_set(self):
        """__init_subclass__가 tensor.name을 attribute name으로 설정."""
        for name, tensor in FmapIOKernel._tensor_descriptors.items():
            assert tensor.name == name

    def test_no_inheritance_between_subkernels(self):
        """각 sub-kernel은 독립적인 descriptor를 가진다."""
        assert "bias" not in FmapIOKernel._tensor_descriptors
        assert "ifm" not in BiasLoaderKernel._tensor_descriptors
        assert "weight" not in FmapIOKernel._tensor_descriptors

    def test_non_tensor_attrs_ignored(self):
        """일반 클래스 속성(spec 등)은 등록되지 않는다."""
        assert "spec" not in FmapIOKernel._tensor_descriptors


# ═══════════════════════════════════════════════════════════════════
# §3  Instance methods — tensors(), get_tensor(), verify()
# ═══════════════════════════════════════════════════════════════════


class TestKernelInstanceMethods:

    def test_tensors_returns_list(self):
        k = FmapIOKernel()
        ts = k.tensors()
        assert isinstance(ts, list)
        names = {t.name for t in ts}
        assert names == {"ifm", "ofm", "concat"}

    def test_get_tensor_existing(self):
        k = BiasLoaderKernel()
        t = k.get_tensor("bias")
        assert isinstance(t, Tensor)
        assert t.name == "bias"
        assert t.dtype == torch.int32

    def test_get_tensor_missing_raises(self):
        k = FmapIOKernel()
        with pytest.raises(AttributeError, match="No tensor 'nonexistent'"):
            k.get_tensor("nonexistent")

    def test_generate_inputs_base_raises(self):
        """Base Kernel.generate_inputs는 NotImplementedError."""
        class BareBoneKernel(Kernel):
            spec = "x.yaml"
        k = BareBoneKernel()
        with pytest.raises(NotImplementedError):
            k.generate_inputs()

    def test_forward_base_raises(self):
        class BareBoneKernel(Kernel):
            spec = "x.yaml"
        k = BareBoneKernel()
        with pytest.raises(NotImplementedError):
            k.forward()



# ═══════════════════════════════════════════════════════════════════
# §4  generate_inputs / forward — NPU 3D sub-kernel patterns
# ═══════════════════════════════════════════════════════════════════


class TestSubKernelInputsForward:
    """실제 NPU 3D sub-kernel의 generate_inputs/forward 동작."""

    def _resolve_kernel(self, kernel_cls, params):
        """Helper: descriptor 복사 + shape resolve."""
        k = kernel_cls()
        for name in k.__class__._tensor_descriptors:
            class_t = k.__class__._tensor_descriptors[name]
            inst_t = copy.copy(class_t)
            setattr(k, name, inst_t)
            resolved = []
            for dim in inst_t.shape:
                if isinstance(dim, int):
                    resolved.append(dim)
                else:
                    key = dim.strip("${}")
                    resolved.append(params[key])
            inst_t._resolved_shape = tuple(resolved)
            inst_t._element_count = math.prod(inst_t._resolved_shape)
        return k

    def test_fmapio_generate_inputs(self):
        """fmapIO: IFM int8 (1, 16, 16, 16)."""
        k = self._resolve_kernel(FmapIOKernel, {
            "IN_CH": 1, "IN_DEPTH": 16, "IN_HEIGHT": 16, "IN_WIDTH": 16,
            "OUT_CH": 32, "OUT_DEPTH": 16, "OUT_HEIGHT": 16, "OUT_WIDTH": 16,
            "CONCAT_CH": 0,
        })
        k.generate_inputs(seed=42)
        assert k.ifm.data is not None
        assert k.ifm.data.shape == (1, 16, 16, 16)
        assert k.ifm.data.dtype == torch.int8

    def test_bias_loader_generate_inputs(self):
        """bias_loader: bias int32 (64,)."""
        k = self._resolve_kernel(BiasLoaderKernel, {"OUT_CH": 64})
        k.generate_inputs(seed=42)
        assert k.bias.data.shape == (64,)
        assert k.bias.data.dtype == torch.int32

    def test_weight_loader_generate_inputs(self):
        """weight_loader: weight int8 (64, 32, 3, 3, 3)."""
        k = self._resolve_kernel(WeightLoaderKernel, {"OUT_CH": 64, "IN_CH": 32})
        k.generate_inputs(seed=42)
        assert k.weight.data.shape == (64, 32, 3, 3, 3)
        assert k.weight.data.dtype == torch.int8

    def test_deterministic_seed(self):
        """같은 seed → 동일 weight 데이터."""
        def _make():
            k = self._resolve_kernel(WeightLoaderKernel, {"OUT_CH": 32, "IN_CH": 32})
            k.generate_inputs(seed=42)
            return k.weight.data
        assert torch.equal(_make(), _make())


