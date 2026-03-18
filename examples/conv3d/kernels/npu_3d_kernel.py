"""NPU 3D Conv3D Kernel — vTen verification kernel.

Golden reference and tensor layout transformation for NPU_3D accelerator.
Based on: specs/npu_3d_analysis.md §9 (Tensor Layout), §10 (Golden Reference)

Hardware parameters:
  Ti = 32 (input tile parallelism)
  To = 32 (output channels per group)
  OUT_GROUP = 2
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vten.kernel.tensor import Tensor
from vten.kernel.base import Kernel, register


# ── Constants ──────────────────────────────────────────────────────

Ti = 32  # Input tile parallelism
To = 32  # Output channels per group

# Kernel spatial size → total weight tile elements
WGT_TILE_SIZE = {1: 1, 2: 8, 3: 27}  # kD*kH*kW (padded to 3×3×3 for k=3)


# ── Helper functions ───────────────────────────────────────────────


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _compute_output_dims(
    in_d: int, in_h: int, in_w: int,
    ifm_stride: int, ofm_stride: int, kernel_size: int,
) -> tuple[int, int, int]:
    """Compute output spatial dimensions per npu_3d_analysis.md §8.3."""
    if ofm_stride == 2:
        # Transpose conv: upsample by 2
        return 2 * in_d, 2 * in_h, 2 * in_w
    elif ifm_stride == 2:
        # Downsample by 2
        return _ceil_div(in_d, 2), _ceil_div(in_h, 2), _ceil_div(in_w, 2)
    else:
        return in_d, in_h, in_w


# ── Kernel Class ───────────────────────────────────────────────────


class NPU3DKernel(Kernel):
    """NPU_3D 3D convolution accelerator verification kernel.

    Tensors
    -------
    ifm     : Input feature map   — (IN_CH, IN_DEPTH, IN_HEIGHT, IN_WIDTH) int8/uint8
    weight  : Convolution weights — (OUT_CH, IN_CH+CONCAT_CH, kD, kH, kW) int8
    bias    : Output bias         — (OUT_CH,) int32
    ofm     : Output feature map  — (OUT_CH, OUT_D, OUT_H, OUT_W) int8/uint8
    concat  : Skip-connection IFM — (CONCAT_CH, IN_DEPTH, IN_HEIGHT, IN_WIDTH) int8 (optional)
    """

    spec = "examples/conv3d/specs/npu_3d.yaml"

    # ── Tensor declarations ────────────────────────────────────────
    # Shapes use runtime parameters resolved at instantiate()

    ifm = Tensor(
        shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
        dtype=torch.int8,
        interface="ddr0",
    )

    weight = Tensor(
        shape=("${OUT_CH}", "${TOTAL_IN_CH}", "${KD}", "${KH}", "${KW}"),
        dtype=torch.int8,
        interface="hbm",
    )

    bias = Tensor(
        shape=("${OUT_CH}",),
        dtype=torch.int32,
        interface="ddr1",
    )

    ofm = Tensor(
        shape=("${OUT_CH}", "${OUT_DEPTH}", "${OUT_HEIGHT}", "${OUT_WIDTH}"),
        dtype=torch.int8,
        interface="ddr0",
    )

    concat = Tensor(
        shape=("${CONCAT_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
        dtype=torch.int8,
        interface="ddr0",
    )

    # ── Register handles ───────────────────────────────────────────

    ctrl_bias_loader = register("ctrl_bias_loader")
    ctrl_act_quant = register("ctrl_act_quant")
    ctrl_psum_buffer = register("ctrl_psum_buffer")
    ctrl_weight_loader = register("ctrl_weight_loader")
    ctrl_mac_atu = register("ctrl_mac_atu")
    ctrl_fmapio = register("ctrl_fmapio")

    # ── Input generation ───────────────────────────────────────────

    def generate_inputs(self, seed: int | None = None):
        """Generate random IFM, weight, bias (and optional concat) tensors.

        Channel dimensions are NOT padded here — padding is done during
        tensor serialization (Stage 3) per the HW layout requirements.
        """
        gen = torch.Generator().manual_seed(seed or 0)

        # IFM: signed or unsigned based on IFM_DTYPE
        is_signed = getattr(self, "IFM_DTYPE", 1)  # default signed
        if is_signed:
            self.ifm.data = torch.randint(
                -128, 127, self.ifm._resolved_shape,
                dtype=torch.int8, generator=gen,
            )
        else:
            self.ifm.data = torch.randint(
                0, 255, self.ifm._resolved_shape,
                dtype=torch.uint8, generator=gen,
            )

        # Weight: always int8
        self.weight.data = torch.randint(
            -128, 127, self.weight._resolved_shape,
            dtype=torch.int8, generator=gen,
        )

        # Bias: int32
        self.bias.data = torch.randint(
            -(2**15), 2**15, self.bias._resolved_shape,
            dtype=torch.int32, generator=gen,
        )

        # Concat (if enabled)
        is_concat = getattr(self, "IS_CONCAT", 0)
        if is_concat and self.concat._resolved_shape is not None:
            concat_ch = self.concat._resolved_shape[0]
            if concat_ch > 0:
                self.concat.data = torch.randint(
                    -128, 127, self.concat._resolved_shape,
                    dtype=torch.int8, generator=gen,
                )

    # ── Golden reference ───────────────────────────────────────────

    def forward(self) -> torch.Tensor:
        """PyTorch golden reference for NPU_3D conv3d.

        Implements the computation from npu_3d_analysis.md §10:
          1. Optional channel concat (skip connection)
          2. Conv3D or ConvTranspose3D
          3. Bias shift (right shift quantization)
          4. Optional ReLU + clip

        Returns: (OUT_CH, OUT_D, OUT_H, OUT_W) int8/uint8
        """
        is_concat = getattr(self, "IS_CONCAT", 0)
        ofm_stride = getattr(self, "OFM_STRIDE", 1)
        ifm_stride = getattr(self, "IFM_STRIDE", 1)
        bias_shift = getattr(self, "BIAS_SHIFT", 0)
        is_relu = getattr(self, "IS_RELU", 0)
        kernel_size = getattr(self, "KERNEL_SIZE", 3)

        # Spatial kernel size: 1→1, 2→3(padded), 3→3
        k_spatial = {1: 1, 2: 3, 3: 3}.get(kernel_size, 3)
        padding = 1 if k_spatial == 3 else 0

        # Input data
        ifm_float = self.ifm.data.float()
        if is_concat and self.concat.data is not None:
            concat_float = self.concat.data.float()
            input_data = torch.cat((ifm_float, concat_float), dim=0)
        else:
            input_data = ifm_float

        wgt_float = self.weight.data.float()
        bias_float = self.bias.data.float()

        # Convolution
        input_batch = input_data.unsqueeze(0)  # add batch dim

        if ofm_stride == 1:
            # Normal Conv3D
            result = F.conv3d(
                input_batch, wgt_float,
                bias=bias_float,
                stride=ifm_stride,
                padding=padding,
            ).to(torch.int32)
        else:
            # Transpose Conv3D (upsampling)
            result = F.conv_transpose3d(
                input_batch, wgt_float,
                bias=bias_float,
                stride=ofm_stride,
            ).to(torch.int32)

        # Quantization pipeline
        shifted = result >> bias_shift

        if is_relu:
            activated = F.relu(shifted)
            clipped = torch.clip(activated, 0, 255).to(torch.uint8)
        else:
            clipped = torch.clip(shifted, -128, 127).to(torch.int8)

        return clipped.squeeze(0)  # remove batch dim

    # ── Custom verification ────────────────────────────────────────

    def verify(self, hw_output: torch.Tensor, golden: torch.Tensor) -> bool:
        """NPU 3D verification with ±1 tolerance for rounding errors.

        Per npu_3d_analysis.md §10 (Output Comparison):
        - Exact match is ideal
        - ±1 tolerance is acceptable (rounding)
        - >1 diff is a significant error
        """
        diff = (hw_output.int() - golden.int()).abs()
        max_diff = diff.max().item()
        if max_diff > 1:
            mismatch_count = (diff > 1).sum().item()
            total = diff.numel()
            print(
                f"[NPU3D VERIFY FAIL] max_diff={max_diff}, "
                f"mismatches(>1)={mismatch_count}/{total}"
            )
            return False
        return True


# ── IFM Layout Transform ──────────────────────────────────────────
# These are standalone functions for use in tensor serialization.
# They implement the HW data layout from npu_3d_analysis.md §9.


def ifm_to_hw_layout(
    ifm: torch.Tensor,
    in_ch_pkt: int,
) -> torch.Tensor:
    """Transform IFM from (C, D, H, W) to HW layout (D, C_pkt, H, W, Ti).

    Per npu_3d_analysis.md §9.1:
      1. Pad channels to Ti boundary
      2. Reshape to (C_pkt, Ti, D, H, W)
      3. Permute to (D, C_pkt, H, W, Ti)
    """
    c, d, h, w = ifm.shape
    ich_pad = (Ti - c % Ti) % Ti

    if ich_pad > 0:
        ifm_padded = F.pad(ifm, (0, 0, 0, 0, 0, 0, 0, ich_pad))
    else:
        ifm_padded = ifm

    ifm_reshaped = ifm_padded.reshape(in_ch_pkt, Ti, d, h, w)
    ifm_permuted = ifm_reshaped.permute(2, 0, 3, 4, 1)  # (D, C_pkt, H, W, Ti)
    return ifm_permuted.contiguous()


def ofm_from_hw_layout(
    ofm_raw: torch.Tensor,
    out_ch: int,
    out_d: int, out_h: int, out_w: int,
) -> torch.Tensor:
    """Transform OFM from HW layout (D, C_pkt, H, W, To) back to (C, D, H, W).

    Per npu_3d_analysis.md §9.4:
      1. Reshape to (D, C_pkt, H, W, To)
      2. Permute to (C_pkt, To, D, H, W)
      3. Reshape to (C_pkt*To, D, H, W)
      4. Trim to actual out_ch
    """
    out_ch_pkt = _ceil_div(out_ch, To)
    ofm_reshaped = ofm_raw.reshape(out_d, out_ch_pkt, out_h, out_w, To)
    ofm_transposed = ofm_reshaped.permute(1, 4, 0, 2, 3)  # (C_pkt, To, D, H, W)
    ofm_flat = ofm_transposed.reshape(out_ch_pkt * To, out_d, out_h, out_w)
    return ofm_flat[:out_ch]


def weight_to_hw_layout(
    wgt: torch.Tensor,
    kernel_size: int,
    ofm_stride: int,
) -> torch.Tensor:
    """Transform weight from (O, I, kD, kH, kW) to HW tiled layout.

    Per npu_3d_analysis.md §9.2:
      1. Spatial flip (normal conv only)
      2. Transpose to (I, O, kD, kH, kW)
      3. Pad channels to Ti/To boundaries
      4. Pad spatial to 3×3×3
      5. Reshape to tiles (in_ch_pkt, Ti, out_ch_pkt, To, k_space)
    """
    out_ch, in_ch, kd, kh, kw = wgt.shape

    # Step 1: Spatial flip for normal conv
    if ofm_stride == 1:
        wgt = torch.flip(wgt, [2, 3, 4])

    # Step 2: Transpose
    wgt = wgt.permute(1, 0, 2, 3, 4)  # (I, O, kD, kH, kW)

    # Step 3: Pad channels
    ich_pad = (Ti - in_ch % Ti) % Ti
    och_pad = (To - out_ch % To) % To
    if ich_pad > 0 or och_pad > 0:
        wgt = F.pad(wgt, (0, 0, 0, 0, 0, 0, 0, och_pad, 0, ich_pad))

    in_ch_pkt = _ceil_div(in_ch, Ti)
    out_ch_pkt = _ceil_div(out_ch, To)

    # Step 4: Pad spatial to 3×3×3
    k_target = 3
    sd = k_target - kd
    sh = k_target - kh
    sw = k_target - kw
    if sd > 0 or sh > 0 or sw > 0:
        wgt = F.pad(wgt, (0, max(sw, 0), 0, max(sh, 0), 0, max(sd, 0)))

    k_space = k_target ** 3  # 27

    # Step 5: Reshape to tiles
    wgt_tiled = wgt.reshape(in_ch_pkt, Ti, out_ch_pkt, To, k_space)
    return wgt_tiled.contiguous()


def bias_to_hw_layout(bias: torch.Tensor, out_ch: int) -> torch.Tensor:
    """Transform bias from (O,) to (O_pkt, To) int32.

    Per npu_3d_analysis.md §9.3.
    """
    out_ch_pkt = _ceil_div(out_ch, To)
    och_pad = (To - out_ch % To) % To
    if och_pad > 0:
        bias = F.pad(bias, (0, och_pad))
    return bias.reshape(out_ch_pkt, To).contiguous()
