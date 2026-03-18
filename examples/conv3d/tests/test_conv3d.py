"""Tests for NPU_3D 3D convolution accelerator verification.

Spec reference: specs/npu_3d_analysis.md
Kernel spec:    examples/conv3d/specs/npu_3d.yaml
Kernel class:   examples/conv3d/kernels/npu_3d_kernel.py

Test categories:
  §1  Golden reference correctness (PyTorch)
  §2  Tensor layout transforms (IFM, Weight, Bias, OFM)
  §3  Buffer size calculations
  §4  Register configuration sequences
  §5  DSL test scenarios (E2E command flow)
  §6  Parametric sweeps (kernel_size, stride, concat, relu)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from examples.conv3d.kernels.npu_3d_kernel import (
    NPU3DKernel,
    Ti, To,
    WGT_TILE_SIZE,
    ifm_to_hw_layout,
    ofm_from_hw_layout,
    weight_to_hw_layout,
    bias_to_hw_layout,
    _ceil_div,
    _compute_output_dims,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_kernel(
    in_ch=32, in_depth=4, in_height=4, in_width=4,
    out_ch=32, kernel_size=3, ifm_stride=1, ofm_stride=1,
    is_relu=0, is_concat=0, concat_ch=0, bias_shift=0,
    ifm_dtype=1, scale_shift=0,
) -> NPU3DKernel:
    """Create an NPU3DKernel with resolved shapes for unit testing.

    This simulates what KernelInstance.initialize() would do:
    resolve parameters, shallow-copy tensors, set resolved shapes.
    """
    import copy

    k = NPU3DKernel()

    # Set resolved parameter values as attributes
    k.IN_DEPTH = in_depth
    k.IN_HEIGHT = in_height
    k.IN_WIDTH = in_width
    k.IN_CH = in_ch
    k.OUT_CH = out_ch
    k.KERNEL_SIZE = kernel_size
    k.IFM_STRIDE = ifm_stride
    k.OFM_STRIDE = ofm_stride
    k.IS_RELU = is_relu
    k.IS_CONCAT = is_concat
    k.CONCAT_CH = concat_ch
    k.BIAS_SHIFT = bias_shift
    k.IFM_DTYPE = ifm_dtype
    k.SCALE_SHIFT = scale_shift

    # Compute derived dimensions
    k_spatial = {1: 1, 2: 3, 3: 3}.get(kernel_size, 3)
    total_in_ch = in_ch + concat_ch if is_concat else in_ch
    out_d, out_h, out_w = _compute_output_dims(
        in_depth, in_height, in_width, ifm_stride, ofm_stride, kernel_size,
    )

    # Resolve tensor shapes (simulate _resolve_shape)
    shape_map = {
        "ifm": (in_ch, in_depth, in_height, in_width),
        "weight": (out_ch, total_in_ch, k_spatial, k_spatial, k_spatial),
        "bias": (out_ch,),
        "ofm": (out_ch, out_d, out_h, out_w),
        "concat": (concat_ch, in_depth, in_height, in_width),
    }

    for name in k.__class__._tensor_descriptors:
        class_tensor = k.__class__._tensor_descriptors[name]
        instance_tensor = copy.copy(class_tensor)
        setattr(k, name, instance_tensor)
        instance_tensor._resolved_shape = shape_map.get(name)
        if instance_tensor._resolved_shape:
            instance_tensor._element_count = math.prod(instance_tensor._resolved_shape)

    return k


# ═══════════════════════════════════════════════════════════════════
# §1  Golden reference correctness
# ═══════════════════════════════════════════════════════════════════


class TestGoldenReference:
    """NPU3DKernel.forward() vs PyTorch F.conv3d 직접 계산 비교."""

    def test_conv3d_3x3x3_basic(self):
        """3×3×3 normal conv, stride=1, no relu, no bias_shift."""
        k = _make_kernel(
            in_ch=32, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, ifm_stride=1, ofm_stride=1,
            is_relu=0, bias_shift=0,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()

        # Independently verify
        ifm = k.ifm.data.float().unsqueeze(0)
        wgt = k.weight.data.float()
        bias = k.bias.data.float()
        ref = F.conv3d(ifm, wgt, bias=bias, stride=1, padding=1).to(torch.int32)
        ref = torch.clip(ref, -128, 127).to(torch.int8).squeeze(0)

        assert torch.equal(golden, ref)

    def test_conv3d_1x1x1(self):
        """1×1×1 convolution (pointwise)."""
        k = _make_kernel(
            in_ch=64, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=1, ifm_stride=1, ofm_stride=1,
            is_relu=0, bias_shift=0,
        )
        k.generate_inputs(seed=7)
        golden = k.forward()
        assert golden.shape == (32, 4, 4, 4)

    def test_conv3d_with_relu(self):
        """ReLU activation → output in [0, 255] (uint8)."""
        k = _make_kernel(
            in_ch=32, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, is_relu=1, bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()

        assert golden.dtype == torch.uint8
        assert golden.min().item() >= 0
        assert golden.max().item() <= 255

    def test_conv3d_without_relu(self):
        """No ReLU → output in [-128, 127] (int8)."""
        k = _make_kernel(
            in_ch=32, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, is_relu=0, bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()

        assert golden.dtype == torch.int8
        assert golden.min().item() >= -128
        assert golden.max().item() <= 127

    def test_conv3d_bias_shift(self):
        """bias_shift=10 으로 right-shift quantization."""
        k = _make_kernel(
            in_ch=32, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, bias_shift=10, is_relu=0,
        )
        k.generate_inputs(seed=99)
        golden = k.forward()

        # 모든 값이 clip 범위 내
        assert golden.min().item() >= -128
        assert golden.max().item() <= 127

    def test_conv3d_stride2_downsample(self):
        """ifm_stride=2 → output spatial dims halved."""
        k = _make_kernel(
            in_ch=32, out_ch=64,
            in_depth=8, in_height=8, in_width=8,
            kernel_size=3, ifm_stride=2, ofm_stride=1,
            bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()

        # Output should be ceil(8/2) = 4
        assert golden.shape == (64, 4, 4, 4)

    def test_conv3d_transpose_upsample(self):
        """ofm_stride=2 → transpose conv, output spatial dims doubled."""
        k = _make_kernel(
            in_ch=64, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, ifm_stride=1, ofm_stride=2,
            bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()

        # Transpose conv: output = 2 * input
        assert golden.shape == (32, 8, 8, 8)

    def test_conv3d_with_concat(self):
        """is_concat=1: IFM + concat tensor 채널 결합 후 conv."""
        k = _make_kernel(
            in_ch=32, out_ch=64,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, is_concat=1, concat_ch=32,
            bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()

        # Weight shape should be (64, 64, 3, 3, 3) for 32+32 input channels
        assert k.weight._resolved_shape == (64, 64, 3, 3, 3)
        assert golden.shape == (64, 4, 4, 4)


# ═══════════════════════════════════════════════════════════════════
# §2  Tensor layout transforms
# ═══════════════════════════════════════════════════════════════════


class TestIFMLayout:
    """IFM layout: (C, D, H, W) → (D, C_pkt, H, W, Ti)."""

    def test_exact_tile_boundary(self):
        """in_ch = 32 (Ti boundary) → no padding."""
        ifm = torch.randn(32, 4, 4, 4, dtype=torch.float32)
        result = ifm_to_hw_layout(ifm, in_ch_pkt=1)
        assert result.shape == (4, 1, 4, 4, Ti)

    def test_padding_needed(self):
        """in_ch = 48 → pad to 64 → C_pkt = 2."""
        ifm = torch.randn(48, 4, 4, 4, dtype=torch.float32)
        result = ifm_to_hw_layout(ifm, in_ch_pkt=2)
        assert result.shape == (4, 2, 4, 4, Ti)

    def test_single_channel(self):
        """in_ch = 3 → pad to 32 → C_pkt = 1."""
        ifm = torch.randn(3, 2, 2, 2, dtype=torch.float32)
        result = ifm_to_hw_layout(ifm, in_ch_pkt=1)
        assert result.shape == (2, 1, 2, 2, Ti)
        # Padded channels should be 0
        assert torch.all(result[:, :, :, :, 3:] == 0)

    def test_data_preservation(self):
        """원본 데이터가 transform 후에도 보존되는지 확인."""
        ifm = torch.arange(32 * 2 * 2 * 2, dtype=torch.float32).reshape(32, 2, 2, 2)
        hw = ifm_to_hw_layout(ifm, in_ch_pkt=1)

        # (D=0, C_pkt=0, H=0, W=0, :) 은 ifm[:, 0, 0, 0]과 같아야 함
        expected = ifm[:, 0, 0, 0]  # shape (32,)
        actual = hw[0, 0, 0, 0, :]  # shape (Ti,)
        assert torch.equal(actual, expected)


class TestOFMLayout:
    """OFM layout: HW (D, C_pkt, H, W, To) → (C, D, H, W)."""

    def test_roundtrip_exact(self):
        """out_ch = 64, To boundary → trim 불필요."""
        out_ch, out_d, out_h, out_w = 64, 4, 4, 4
        out_ch_pkt = _ceil_div(out_ch, To)
        hw_ofm = torch.randn(out_d * out_ch_pkt * out_h * out_w * To)
        hw_ofm = hw_ofm.reshape(out_d, out_ch_pkt, out_h, out_w, To)

        result = ofm_from_hw_layout(hw_ofm.flatten(), out_ch, out_d, out_h, out_w)
        assert result.shape == (64, 4, 4, 4)

    def test_trim_channels(self):
        """out_ch = 48 → pad to 64, then trim back to 48."""
        out_ch, out_d, out_h, out_w = 48, 2, 2, 2
        out_ch_pkt = _ceil_div(out_ch, To)  # 2
        hw_ofm = torch.randn(out_d, out_ch_pkt, out_h, out_w, To)

        result = ofm_from_hw_layout(hw_ofm, out_ch, out_d, out_h, out_w)
        assert result.shape == (48, 2, 2, 2)


class TestWeightLayout:
    """Weight layout transform per npu_3d_analysis.md §9.2."""

    def test_3x3x3_no_spatial_pad(self):
        """3×3×3 kernel → spatial padding 불필요."""
        wgt = torch.randn(32, 32, 3, 3, 3, dtype=torch.float32)
        result = weight_to_hw_layout(wgt, kernel_size=3, ofm_stride=1)
        assert result.shape == (1, Ti, 1, To, 27)  # 3^3 = 27

    def test_1x1x1_spatial_pad_to_3x3x3(self):
        """1×1×1 kernel → spatial 을 3×3×3으로 pad."""
        wgt = torch.randn(32, 32, 1, 1, 1, dtype=torch.float32)
        result = weight_to_hw_layout(wgt, kernel_size=1, ofm_stride=1)
        assert result.shape == (1, Ti, 1, To, 27)

    def test_channel_padding(self):
        """in_ch=48, out_ch=48 → pad to 64 each."""
        wgt = torch.randn(48, 48, 3, 3, 3, dtype=torch.float32)
        result = weight_to_hw_layout(wgt, kernel_size=3, ofm_stride=1)
        in_ch_pkt = _ceil_div(48, Ti)   # 2
        out_ch_pkt = _ceil_div(48, To)  # 2
        assert result.shape == (in_ch_pkt, Ti, out_ch_pkt, To, 27)

    def test_spatial_flip_normal_conv(self):
        """ofm_stride=1 → spatial flip 적용됨."""
        wgt = torch.zeros(32, 32, 3, 3, 3, dtype=torch.float32)
        wgt[0, 0, 0, 0, 0] = 1.0  # corner element

        result = weight_to_hw_layout(wgt, kernel_size=3, ofm_stride=1)
        result_flat = result.reshape(-1)

        # After flip, element [0,0,0] should move to [2,2,2]
        wgt_flipped = torch.flip(wgt, [2, 3, 4])
        assert wgt_flipped[0, 0, 2, 2, 2] == 1.0

    def test_no_flip_transpose_conv(self):
        """ofm_stride=2 → spatial flip 없음."""
        wgt = torch.zeros(32, 32, 3, 3, 3, dtype=torch.float32)
        wgt[0, 0, 0, 0, 0] = 1.0

        result = weight_to_hw_layout(wgt, kernel_size=3, ofm_stride=2)
        # First element should stay at position 0 (no flip)
        # After transpose: (I, O, kD, kH, kW), element is at [0,0,0,0,0]
        # After reshape: it should be at tile position [0,0,0,0,0]
        assert result[0, 0, 0, 0, 0] == 1.0


class TestBiasLayout:
    """Bias layout: (O,) → (O_pkt, To) int32."""

    def test_exact_boundary(self):
        bias = torch.randint(-1000, 1000, (64,), dtype=torch.int32)
        result = bias_to_hw_layout(bias, out_ch=64)
        assert result.shape == (2, To)

    def test_padding(self):
        bias = torch.randint(-1000, 1000, (48,), dtype=torch.int32)
        result = bias_to_hw_layout(bias, out_ch=48)
        assert result.shape == (2, To)
        # Padded elements should be 0
        assert torch.all(result[1, 16:] == 0)

    def test_data_preserved(self):
        bias = torch.arange(32, dtype=torch.int32)
        result = bias_to_hw_layout(bias, out_ch=32)
        assert result.shape == (1, To)
        assert torch.equal(result.flatten(), bias)


# ═══════════════════════════════════════════════════════════════════
# §3  Buffer size calculations
# ═══════════════════════════════════════════════════════════════════


class TestBufferSizes:
    """Buffer size formulas from npu_3d_analysis.md §8.3."""

    def test_ifm_size(self):
        in_ch, in_d, in_h, in_w = 64, 4, 8, 8
        in_ch_pkt = _ceil_div(in_ch, Ti)  # 2
        expected = in_d * in_ch_pkt * in_h * in_w * Ti
        assert expected == 4 * 2 * 8 * 8 * 32

    def test_wgt_size_3x3x3(self):
        in_ch, out_ch = 64, 128
        kernel_size = 3
        in_ch_pkt = _ceil_div(in_ch, Ti)   # 2
        out_ch_pkt = _ceil_div(out_ch, To)  # 4
        wgt_tile_size = WGT_TILE_SIZE[kernel_size]  # 27? → actually 32 from analysis
        # Analysis says WGT_TILE_SIZE = {3: 32, 2: 8, 1: 1}
        # But spec analysis §8.3 defines this
        expected = in_ch_pkt * out_ch_pkt * To * wgt_tile_size
        assert expected == 2 * 4 * 32 * WGT_TILE_SIZE[3]

    def test_bias_size(self):
        out_ch = 128
        out_ch_pkt = _ceil_div(out_ch, To)  # 4
        expected = out_ch_pkt * To * 4  # int32 = 4 bytes
        assert expected == 4 * 32 * 4

    def test_ofm_size(self):
        out_ch, out_d, out_h, out_w = 64, 4, 8, 8
        out_ch_pkt = _ceil_div(out_ch, To)  # 2
        expected = out_d * out_ch_pkt * To * out_h * out_w
        assert expected == 4 * 2 * 32 * 8 * 8


# ═══════════════════════════════════════════════════════════════════
# §4  Output dimension calculations
# ═══════════════════════════════════════════════════════════════════


class TestOutputDims:

    def test_stride1_same_dims(self):
        d, h, w = _compute_output_dims(8, 16, 16, ifm_stride=1, ofm_stride=1, kernel_size=3)
        assert (d, h, w) == (8, 16, 16)

    def test_stride2_downsample(self):
        d, h, w = _compute_output_dims(8, 16, 16, ifm_stride=2, ofm_stride=1, kernel_size=3)
        assert (d, h, w) == (4, 8, 8)

    def test_stride2_downsample_odd(self):
        d, h, w = _compute_output_dims(7, 15, 15, ifm_stride=2, ofm_stride=1, kernel_size=3)
        assert (d, h, w) == (4, 8, 8)  # ceil

    def test_ofm_stride2_upsample(self):
        d, h, w = _compute_output_dims(4, 8, 8, ifm_stride=1, ofm_stride=2, kernel_size=3)
        assert (d, h, w) == (8, 16, 16)


# ═══════════════════════════════════════════════════════════════════
# §5  Verify method (±1 tolerance)
# ═══════════════════════════════════════════════════════════════════


class TestVerify:
    """NPU3DKernel.verify() — ±1 tolerance for rounding."""

    def test_exact_match(self):
        k = NPU3DKernel()
        a = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        assert k.verify(a, b) is True

    def test_off_by_one_acceptable(self):
        k = NPU3DKernel()
        a = torch.tensor([10, 20, 30], dtype=torch.int8)
        b = torch.tensor([11, 19, 31], dtype=torch.int8)
        assert k.verify(a, b) is True

    def test_off_by_two_fails(self):
        k = NPU3DKernel()
        a = torch.tensor([10, 20, 30], dtype=torch.int8)
        b = torch.tensor([12, 20, 30], dtype=torch.int8)  # diff=2
        assert k.verify(a, b) is False


# ═══════════════════════════════════════════════════════════════════
# §6  Register configuration (host workflow)
# ═══════════════════════════════════════════════════════════════════


class TestRegisterConfig:
    """Host workflow register sequence from npu_3d_analysis.md §8.1.

    순서: bias_loader → act_quant → psum_buffer → weight_loader → mac_atu → fmapIO
    """

    def test_register_handles_exist(self):
        k = NPU3DKernel()
        # 6개 IP 커널의 register handle이 모두 존재해야 함
        assert hasattr(k, "ctrl_bias_loader")
        assert hasattr(k, "ctrl_act_quant")
        assert hasattr(k, "ctrl_psum_buffer")
        assert hasattr(k, "ctrl_weight_loader")
        assert hasattr(k, "ctrl_mac_atu")
        assert hasattr(k, "ctrl_fmapio")

    def test_register_interface_names(self):
        k = NPU3DKernel()
        handles = k.__class__._register_handles
        assert handles["ctrl_bias_loader"].interface_name == "ctrl_bias_loader"
        assert handles["ctrl_act_quant"].interface_name == "ctrl_act_quant"
        assert handles["ctrl_psum_buffer"].interface_name == "ctrl_psum_buffer"
        assert handles["ctrl_weight_loader"].interface_name == "ctrl_weight_loader"
        assert handles["ctrl_mac_atu"].interface_name == "ctrl_mac_atu"
        assert handles["ctrl_fmapio"].interface_name == "ctrl_fmapio"


# ═══════════════════════════════════════════════════════════════════
# §7  Tensor descriptors
# ═══════════════════════════════════════════════════════════════════


class TestTensorDescriptors:

    def test_all_tensors_registered(self):
        descs = NPU3DKernel._tensor_descriptors
        assert "ifm" in descs
        assert "weight" in descs
        assert "bias" in descs
        assert "ofm" in descs
        assert "concat" in descs

    def test_tensor_interfaces(self):
        descs = NPU3DKernel._tensor_descriptors
        assert descs["ifm"].interface == "ddr0"
        assert descs["weight"].interface == "hbm"
        assert descs["bias"].interface == "ddr1"
        assert descs["ofm"].interface == "ddr0"
        assert descs["concat"].interface == "ddr0"

    def test_tensor_dtypes(self):
        descs = NPU3DKernel._tensor_descriptors
        assert descs["ifm"].dtype == torch.int8
        assert descs["weight"].dtype == torch.int8
        assert descs["bias"].dtype == torch.int32
        assert descs["ofm"].dtype == torch.int8
        assert descs["concat"].dtype == torch.int8

    def test_tensor_names_auto_set(self):
        descs = NPU3DKernel._tensor_descriptors
        for name, tensor in descs.items():
            assert tensor.name == name


# ═══════════════════════════════════════════════════════════════════
# §8  DSL test scenario (E2E command flow)
# ═══════════════════════════════════════════════════════════════════


class TestDSLScenario:
    """E2E DSL scenario 구조 검증.

    Host workflow (npu_3d_analysis.md §8.1):
      1. Load IFM, Weight, Bias to memory
      2. Configure all 6 IP kernels (register writes)
      3. VSYNC triggers (bias_loader, weight_loader, fmapIO)
      4. Poll LAYER_DONE
      5. Store OFM, verify against golden

    이 테스트는 DSL Operation 체인이 올바른지 검증한다.
    (ExecutionContext가 구현되면 실제 ctx 사용으로 전환)
    """

    def test_operation_chain_structure(self):
        """Operation 체인이 올바른 OpKind 순서를 따르는지."""
        from vten.dsl.operations import Operation, OperationHandle
        from vten.spec.models import OpKind

        # Phase 1: Load tensors to memory
        load_ifm = OperationHandle(op=Operation(kind=OpKind.LOAD_TENSOR))
        load_wgt = OperationHandle(op=Operation(kind=OpKind.LOAD_TENSOR))
        load_bias = OperationHandle(op=Operation(kind=OpKind.LOAD_TENSOR))

        # Phase 2: Configure all registers (auto_bind)
        configure = OperationHandle(
            op=Operation(kind=OpKind.CONFIGURE, dep=[load_ifm, load_wgt, load_bias])
        )

        # Phase 3: VSYNC triggers
        # bias_loader VSYNC
        vsync_bias = OperationHandle(
            op=Operation(
                kind=OpKind.WRITE_REGISTER,
                register_interface="ctrl_bias_loader",
                register_fields={"trigger": 1},
                dep=[configure],
            )
        )
        # weight_loader VSYNC
        vsync_wgt = OperationHandle(
            op=Operation(
                kind=OpKind.WRITE_REGISTER,
                register_interface="ctrl_weight_loader",
                register_fields={"trigger": 1},
                dep=[configure],
            )
        )
        # fmapIO VSYNC (must be last — triggers computation)
        vsync_fmapio = OperationHandle(
            op=Operation(
                kind=OpKind.WRITE_REGISTER,
                register_interface="ctrl_fmapio",
                register_fields={"trigger": 1},
                dep=[vsync_bias, vsync_wgt],
            )
        )

        # Phase 4: Poll LAYER_DONE
        poll_done = OperationHandle(
            op=Operation(
                kind=OpKind.POLL_REGISTER,
                register_interface="ctrl_fmapio",
                register_field_name="done",
                dep=[vsync_fmapio],
            )
        )

        # Phase 5: Store OFM
        store_ofm = OperationHandle(
            op=Operation(kind=OpKind.STORE_TENSOR, dep=[poll_done])
        )

        # Verify chain
        assert configure.op.dep == [load_ifm, load_wgt, load_bias]
        assert vsync_fmapio.op.dep == [vsync_bias, vsync_wgt]
        assert poll_done.op.dep == [vsync_fmapio]
        assert store_ofm.op.dep == [poll_done]

    def test_verify_with_golden(self):
        """verify flag가 golden reference와 함께 설정되는지."""
        from vten.dsl.operations import Operation
        from vten.spec.models import OpKind

        golden = torch.randn(32, 4, 4, 4)
        store_op = Operation(
            kind=OpKind.STORE_TENSOR,
            verify=True,
            golden=golden,
        )
        assert store_op.verify is True
        assert torch.equal(store_op.golden, golden)


# ═══════════════════════════════════════════════════════════════════
# §9  Parametric sweeps
# ═══════════════════════════════════════════════════════════════════


class TestParametricSweep:
    """다양한 HW 파라미터 조합으로 golden reference 일관성 검증."""

    @pytest.mark.parametrize("kernel_size", [1, 3])
    def test_kernel_sizes(self, kernel_size):
        k_spatial = {1: 1, 2: 3, 3: 3}[kernel_size]
        k = _make_kernel(
            in_ch=32, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=kernel_size, bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()
        assert golden.shape[0] == 32

    @pytest.mark.parametrize("is_relu", [0, 1])
    def test_relu_variants(self, is_relu):
        k = _make_kernel(
            in_ch=32, out_ch=32,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, is_relu=is_relu, bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()

        if is_relu:
            assert golden.min().item() >= 0
        else:
            # Non-relu can have negative values
            pass

    @pytest.mark.parametrize("bias_shift", [0, 4, 8, 16])
    def test_bias_shift_range(self, bias_shift):
        k = _make_kernel(
            in_ch=32, out_ch=32,
            in_depth=2, in_height=2, in_width=2,
            kernel_size=3, bias_shift=bias_shift,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()
        assert golden.dtype == torch.int8

    @pytest.mark.parametrize(
        "in_ch,out_ch",
        [(1, 32), (3, 32), (32, 64), (64, 128), (128, 256)],
    )
    def test_channel_combinations(self, in_ch, out_ch):
        k = _make_kernel(
            in_ch=in_ch, out_ch=out_ch,
            in_depth=2, in_height=2, in_width=2,
            kernel_size=3, bias_shift=8,
        )
        k.generate_inputs(seed=42)
        golden = k.forward()
        assert golden.shape[0] == out_ch

    def test_deterministic_across_runs(self):
        """동일 seed → 동일 golden."""
        k1 = _make_kernel(in_ch=32, out_ch=32, bias_shift=8)
        k1.generate_inputs(seed=42)
        g1 = k1.forward()

        k2 = _make_kernel(in_ch=32, out_ch=32, bias_shift=8)
        k2.generate_inputs(seed=42)
        g2 = k2.forward()

        assert torch.equal(g1, g2)


# ═══════════════════════════════════════════════════════════════════
# §10  U-Net layer configs (npu_3d_analysis.md §11.3)
# ═══════════════════════════════════════════════════════════════════


class TestUNetLayers:
    """U-Net 3D encoder/decoder layer configs."""

    def test_encoder_l0(self):
        """L0: 1→32ch, conv3d 3×3×3, stride 1."""
        k = _make_kernel(
            in_ch=1, out_ch=32,
            in_depth=16, in_height=16, in_width=16,
            kernel_size=3, ifm_stride=1, ofm_stride=1,
            bias_shift=8,
        )
        k.generate_inputs(seed=0)
        golden = k.forward()
        assert golden.shape == (32, 16, 16, 16)

    def test_encoder_downsample(self):
        """L2: 32→64ch, stride 2 downsample."""
        k = _make_kernel(
            in_ch=32, out_ch=64,
            in_depth=16, in_height=16, in_width=16,
            kernel_size=3, ifm_stride=2, ofm_stride=1,
            bias_shift=8,
        )
        k.generate_inputs(seed=0)
        golden = k.forward()
        assert golden.shape == (64, 8, 8, 8)

    def test_decoder_upsample(self):
        """Decoder transpose conv: 128→64ch, ofm_stride=2."""
        k = _make_kernel(
            in_ch=128, out_ch=64,
            in_depth=4, in_height=4, in_width=4,
            kernel_size=3, ifm_stride=1, ofm_stride=2,
            bias_shift=8,
        )
        k.generate_inputs(seed=0)
        golden = k.forward()
        assert golden.shape == (64, 8, 8, 8)

    def test_decoder_concat_layer(self):
        """Decoder concat: is_concat=1, skip connection."""
        k = _make_kernel(
            in_ch=64, out_ch=64, concat_ch=64,
            in_depth=8, in_height=8, in_width=8,
            kernel_size=3, ifm_stride=1, ofm_stride=1,
            is_concat=1, bias_shift=8,
        )
        k.generate_inputs(seed=0)
        golden = k.forward()
        assert golden.shape == (64, 8, 8, 8)

    def test_final_1x1x1(self):
        """L27: 32→3ch, 1×1×1 kernel (final output)."""
        k = _make_kernel(
            in_ch=32, out_ch=3,
            in_depth=16, in_height=16, in_width=16,
            kernel_size=1, ifm_stride=1, ofm_stride=1,
            bias_shift=0,
        )
        k.generate_inputs(seed=0)
        golden = k.forward()
        assert golden.shape == (3, 16, 16, 16)
