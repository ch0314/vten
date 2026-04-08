# NPU 3D Design & Host Code Analysis

> vTen implementer/tester 참고용 분석 문서
> 분석 대상:
> - RTL: `~/xhw/projects/npu/src/NPU_3D/design/`
> - Host: `~/xhw/projects/npu/host/NPU_3D_pyxrt/`

---

## 1. Design Overview

NPU_3D는 **32-parallel 3D convolution accelerator**로, **6개 독립 IP 커널**로 구성된다.

- **64 total MACs** (32 input tiles × 2 output groups)
- **6 IP Kernels**: fmapIO, weight_loader, mac_atu, psum_buffer, bias_loader, act_quant
- **각 IP별 독립 AXI4-Lite slave** (제어 레지스터)
- **AXI4 master**: DDR 2포트 (fmapIO + bias_loader) + HBM 32포트 (weight_loader)
- **IP 간 연결**: AXI4-Stream (internal)
- **Single clock domain**: `ap_clk` (active high), `ap_aresetn` (active low)
- **Pipeline**: IFM → MAC → Adder Tree → PSUM → Activation/Quantization → OFM

> **NOTE**: `NPU_3D_top.sv`는 legacy wrapper로, 단일 AXI4-Lite 포트를 사용한다.
> vTen 검증에서는 host code의 **6-IP 구조**를 따른다 (XRT 배포 구조 기준).

---

## 2. 6-IP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NPU 3D System                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ bias_loader   │    │ weight_loader │    │ fmapIO           │  │
│  │ [AXI4-Lite]   │    │ [AXI4-Lite]   │    │ [AXI4-Lite]      │  │
│  │ [AXI4 M: DDR] │    │ [AXI4 M: ×32] │    │ [AXI4 M: DDR]   │  │
│  └──────┬───────┘    └──────┬───────┘    └──┬──────────┬────┘  │
│      AXIS│bias           AXIS│wgt×64      AXIS│ifm    AXIS│ofm  │
│         ▼                   ▼               ▼          ▲       │
│  ┌──────────────────────────────────────┐   │          │       │
│  │              mac_atu                  │◄──┘          │       │
│  │              [AXI4-Lite]              │              │       │
│  └──────────────┬───────────────────────┘              │       │
│              AXIS│psum×6                                │       │
│                 ▼                                       │       │
│  ┌──────────────────────┐                              │       │
│  │    psum_buffer        │                              │       │
│  │    [AXI4-Lite]        │                              │       │
│  └──────────┬───────────┘                              │       │
│          AXIS│psum                                      │       │
│             ▼                                           │       │
│  ┌──────────────────────┐                              │       │
│  │    act_quant          │──────────────────────────────┘       │
│  │    [AXI4-Lite]        │                                     │
│  └──────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
```

### External Interfaces (BFM 필요)

| Type | Count | Purpose |
|------|-------|---------|
| AXI4-Lite slave | 6 | 각 IP별 제어 레지스터 |
| AXI4 master (DDR) | 2 | fmapIO (IFM/OFM) + bias_loader (bias) |
| AXI4 master (HBM) | 32 | weight_loader (32 banks) |
| **Total BFMs** | **40** | |

### Internal Interfaces (BFM 불필요, IP 간 직접 연결)

| From | To | Protocol | Width | Streams |
|------|----|----------|-------|---------|
| fmapIO | mac_atu | AXI-Stream | 256b + 32b(coo) | 2 |
| weight_loader | mac_atu | AXI-Stream | 256b | 64 (32×2) |
| mac_atu | psum_buffer | AXI-Stream | 256b | 6 |
| psum_buffer | act_quant | AXI-Stream | 64b | 1 |
| bias_loader | act_quant | AXI-Stream | 256b | 1 |
| act_quant | fmapIO | AXI-Stream | 256b | 1 |

---

## 3. Per-IP Module Specification

### 3.1 fmapIO (Feature Map I/O)

**Module**: `fmapIO_top`
**RTL**: `design/fmapIO/rtl/fmapIO_top.sv`

| Port Type | Details |
|-----------|---------|
| Clock/Reset | `clk`, `rstn` |
| AXI4-Lite (slave) | 16-bit addr, 32-bit data |
| AXI4 (master) | 64-bit addr, 256-bit data (DDR, read+write) |
| AXIS (master) | ifm data (256b), ifm coordinates (32b) |
| AXIS (slave) | ofm data from act_quant (256b) |

### 3.2 weight_loader

**Module**: `weight_loader_top`
**RTL**: `design/weight_loader/rtl/weight_loader_top.sv`

| Port Type | Details |
|-----------|---------|
| Clock/Reset | `ap_clk`, `ap_aresetn` |
| AXI4-Lite (slave) | 16-bit addr, 32-bit data, 71 registers |
| AXI4 (master) ×32 | 64-bit addr, 256-bit data (HBM banks 0-31) |
| AXIS (master) ×64 | 256b weight streams (32 groups × 2 per group) |

### 3.3 mac_atu (MAC + Adder Tree Unit)

**Module**: `mac_atu_top_wrapper`
**RTL**: `design/mac/rtl/mac_atu_top_wrapper.sv`

| Port Type | Details |
|-----------|---------|
| Clock/Reset | `ap_clk`, `ap_aresetn` |
| AXI4-Lite (slave) | 16-bit addr, 32-bit data |
| AXIS (slave) | ifm (256b) + coo (32b) + 64 weight streams (256b each) |
| AXIS (master) ×6 | psum outputs (256b each) |

### 3.4 psum_buffer

**Module**: `psum_buffer_top`
**RTL**: `design/psum_buffer/rtl/psum_buffer_top.sv`

| Port Type | Details |
|-----------|---------|
| Clock/Reset | `ap_clk`, `ap_aresetn` |
| AXI4-Lite (slave) | 16-bit addr, 32-bit data |
| AXIS (slave) ×6 + coo | 6 psum inputs (256b) + coordinates (32b) |
| AXIS (master) | accumulated output (64b) |

### 3.5 bias_loader

**Module**: `bias_loader_top`
**RTL**: `design/bias_loader/rtl/bias_loader_top.sv`

| Port Type | Details |
|-----------|---------|
| Clock/Reset | `clk`, `rstn` |
| AXI4-Lite (slave) | 16-bit addr, 32-bit data |
| AXI4 (master) | 64-bit addr, 256-bit data (DDR) |
| AXIS (master) | bias output (256b) |

### 3.6 act_quant (Activation + Quantization)

**Module**: `act_quant_top`
**RTL**: `design/activation/rtl/act_quant_top.sv`

| Port Type | Details |
|-----------|---------|
| Clock/Reset | `clk`, `rstn` |
| AXI4-Lite (slave) | 16-bit addr, 32-bit data |
| AXIS (slave) | psum input (256b) + bias input (256b) |
| AXIS (master) | quantized output (256b) |

---

## 4. Per-IP Register Maps (RTL 검증 완료)

> Host code (XRT 기준) 레지스터 오프셋. RTL의 `*_reg_map.sv` 파일과 교차 확인 완료.

### 4.1 fmapIO_kernel

| Offset | Field | R/W | Description |
|--------|-------|-----|-------------|
| 0x014 | IN_DEPTH | RW | Input depth |
| 0x018 | IN_HEIGHT | RW | Input height |
| 0x01C | IN_WIDTH | RW | Input width |
| 0x020 | IN_CH | RW | Input channels |
| 0x024 | OUT_CH | RW | Output channels |
| 0x028 | IFM_STRIDE | RW | Input stride |
| 0x02C | OFM_STRIDE | RW | Output stride |
| 0x030 | IS_CONCAT | RW | Concat enable |
| 0x034 | CONCAT_CH | RW | Concat channels |
| 0x038 | IFM_START_ADDR_LSB | RW | IFM addr lower 32b |
| 0x03C | IFM_START_ADDR_MSB | RW | IFM addr upper 32b |
| 0x040 | OFM_START_ADDR_LSB | RW | OFM addr lower 32b |
| 0x044 | OFM_START_ADDR_MSB | RW | OFM addr upper 32b |
| 0x048 | CONCAT_START_ADDR_LSB | RW | Concat addr lower 32b |
| 0x04C | CONCAT_START_ADDR_MSB | RW | Concat addr upper 32b |
| 0x050 | VSYNC | RW | Layer trigger (write 1) |
| 0x054 | LAYER_DONE | **RO** | Layer complete flag (poll until 1) |

### 4.2 weight_loader_kernel

| Offset | Field | R/W | Description |
|--------|-------|-----|-------------|
| 0x010 | IN_WIDTH | RW | Input width |
| 0x014 | IN_HEIGHT | RW | Input height |
| 0x018 | IN_DEPTH | RW | Input depth |
| 0x01C + 8×i | WGT_BASEADDR_[i]_LSB | RW | Weight bank i addr lower 32b (i=0..31) |
| 0x020 + 8×i | WGT_BASEADDR_[i]_MSB | RW | Weight bank i addr upper 32b |
| 0x11C | VSYNC | RW | Trigger (write 1) |
| 0x120 | KERNEL_SIZE | RW | Kernel size (1, 2, or 3) |
| 0x124 | IN_CH | RW | Input channels |
| 0x128 | OUT_CH | RW | Output channels |

> **Weight address 오프셋 패턴** (RTL `weight_reg_map.sv` 확인):
> Bank 0: LSB=0x01C, MSB=0x020 / Bank 1: LSB=0x024, MSB=0x028 / ... / Bank 31: LSB=0x114, MSB=0x118

### 4.3 mac_atu_kernel

| Offset | Field | R/W | Description |
|--------|-------|-----|-------------|
| 0x010 | IFM_IS_SIGNED | RW | Data signedness (0=unsigned, 1=signed) |
| 0x014 | IN_CH | RW | Input channels |
| 0x018 | OUT_CH | RW | Output channels |
| 0x01C | IS_CONCAT | RW | Concat flag |
| 0x020 | CONCAT_IS_SIGNED | RW | Concat data signedness |
| 0x024 | CONCAT_CH | RW | Concat channels |
| 0x028 | IN_WIDTH | RW | Input width |
| 0x02C | IN_HEIGHT | RW | Input height |

### 4.4 psum_buffer_kernel

| Offset | Field | R/W | Description |
|--------|-------|-----|-------------|
| 0x010 | IN_CH | RW | Input channels |
| 0x014 | OUT_CH | RW | Output channels |
| 0x018 | IN_WIDTH | RW | Input width |
| 0x01C | IN_HEIGHT | RW | Input height |
| 0x020 | IN_DEPTH | RW | Input depth |
| 0x024 | VSYNC | RW | Trigger (write 1) |
| 0x028 | KERNEL_SIZE | RW | Kernel size (1, 2, or 3) |
| 0x02C | IFM_STRIDE | RW | Input stride |
| 0x030 | OFM_STRIDE | RW | Output stride |

### 4.5 bias_loader_kernel

| Offset | Field | R/W | Description |
|--------|-------|-----|-------------|
| 0x010 | BIAS_START_ADDR_LSB | RW | Bias addr lower 32b |
| 0x014 | BIAS_START_ADDR_MSB | RW | Bias addr upper 32b |
| 0x018 | OUT_CH | RW | Output channels |
| 0x01C | VSYNC | RW | Trigger (write 1) |

### 4.6 act_quant_kernel

| Offset | Field | R/W | Description |
|--------|-------|-----|-------------|
| 0x014 | in_depth | RW | Input depth |
| 0x018 | in_height | RW | Input height |
| 0x01C | in_width | RW | Input width |
| 0x020 | out_ch | RW | Output channels |
| 0x024 | bias_shift | RW | Right-shift amount for quantization |
| 0x028 | is_relu | RW | ReLU enable (0/1) |
| 0x02C | ifm_stride | RW | Input stride |
| 0x030 | ofm_stride | RW | Output stride |
| 0x034 | VSYNC | RW | Trigger (write 1) |

---

## 5. Data Flow Architecture

```
External Memory (DDR + HBM)
    │
    ├─ [DDR] ──AXI4──► fmapIO ──AXIS──► mac_atu
    │                     ▲                  │
    │                     │              AXIS│psum×6
    │                     │                  ▼
    │                     │            psum_buffer
    │                     │                  │
    │                     │              AXIS│psum
    │                     │                  ▼
    ├─ [DDR] ──AXI4──► bias_loader ──AXIS──► act_quant
    │                                        │
    │                                    AXIS│ofm
    │                                        ▼
    │                   fmapIO ◄──AXIS──── act_quant
    │                     │
    │  ◄──AXI4──── fmapIO (OFM write-back to DDR)
    │
    └─ [HBM×32] ──AXI4──► weight_loader ──AXIS×64──► mac_atu
```

### Pipeline Stages

| Stage | Module | Input | Output | Latency |
|-------|--------|-------|--------|---------|
| 1. IFM Load | fmapIO.ifm_loader | DDR→URAM | 32×8-bit + 3D coords | ~4 cyc |
| 2. MAC | mac_atu | 32×8b IFM + weights | 32×2×16b products | 1 cyc |
| 3. Adder Tree | mac_atu.adder_tree | 32×16b products | 2×27×21b psums | 5 cyc |
| 4. PSUM Buffer | psum_buffer | 6×256b AXIS | 64b psum | ~3 cyc |
| 5. Activation | act_quant | 32b psum + 32b bias | 2×8b quantized | ~3 cyc |
| 6. OFM Write | fmapIO.ofm_writer | 16b act output | 256b DDR burst | ~2 cyc |

---

## 6. Key Hardware Parameters (base_pkg.sv)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Ti | 32 | Input parallelism (tiles per cycle) |
| To | 32 | Output channels per group |
| OUT_GROUP | 2 | Output groups (2 × To = 64 parallel) |
| K_SIZE | 3 | Kernel spatial size (3D) |
| DW | 8 | Data width (bits) |
| D_BITS / H_BITS / W_BITS | 8 | Coordinate widths |
| ICH_BIT / OCH_BIT | 9 | Channel count widths (max 511) |
| SHIFT_BITS | 5 | Shift amount width (0-31) |
| AXI_DATA_WIDTH | 256 | AXI4 data bus width |
| AXI_ADDR_WIDTH | 64 | AXI4 address width |
| AXI_MAX_BURST_LEN | 128 | Max burst beats |
| AXILITE_DATA_WIDTH | 32 | Control register data width |
| AXILITE_ADDR_WIDTH | 16 | Control register addr width |

---

## 7. Host Code Workflow

### 7.1 Execution Flow (Per Layer)

```
1. Write bias_loader registers → VSYNC=1
2. Write act_quant registers (no VSYNC trigger at write time)
3. Write psum_buffer registers → VSYNC=1
4. Write weight_loader registers → VSYNC=1
5. Write mac_atu registers (no VSYNC trigger at write time)
6. Write fmapIO registers → VSYNC=1
7. Poll fmapIO LAYER_DONE (offset 0x054) until == 1
```

### 7.2 Buffer Allocation

```
bias_bo      : Bank 33 — single buffer, all layer biases concatenated
fmap_bo[0-7] : Bank 32 — 8 buffers multiplexed for IFM/OFM/Concat
wgt_bo[0-31] : Banks 0-31 — 32 parallel weight banks
```

### 7.3 Buffer Size Formulas

```python
Ti, To = 32, 32
in_ch_pkt  = ceil(in_ch / Ti)
out_ch_pkt = ceil(out_ch / To)
concat_ch_pkt = ceil(concat_ch / Ti)

WGT_TILE_SIZE = {3: 32, 2: 8, 1: 1}[kernel_size]

# Output dimensions
if ofm_stride == 2:     out_d,h,w = 2 * in_d,h,w
elif ifm_stride == 2:   out_d,h,w = ceil(in_d,h,w / 2)
else:                   out_d,h,w = in_d,h,w

BIAS_SIZE = out_ch_pkt * To * 4  # int32
WGT_SIZE  = in_ch_pkt * out_ch_pkt * To * WGT_TILE_SIZE
IFM_SIZE  = in_d * (in_ch_pkt - concat_ch_pkt if is_concat else in_ch_pkt) * in_h * in_w * Ti
OFM_SIZE  = out_d * out_ch_pkt * To * out_h * out_w
```

---

## 8. Tensor Data Layout & Transformation

### 8.1 IFM (Input Feature Map)

```python
# Original: (in_ch, in_depth, in_height, in_width)
# Step 1: Pad channels to Ti boundary
ich_pad = (Ti - in_ch % Ti) % Ti
ifm_padded = pad(ifm, (0,0,0,0,0,0, 0,ich_pad))   # shape: (in_ch+pad, D, H, W)

# Step 2: Reshape to tiles
ifm_reshaped = reshape(ifm_padded, (in_ch_pkt, Ti, D, H, W))

# Step 3: Permute for HW layout
ifm_permuted = permute(ifm_reshaped, (2, 0, 3, 4, 1))
# → shape: (D, in_ch_pkt, H, W, Ti)

# Step 4: Final shape for host_top.py
ifm_final_shape = (D, in_ch_pkt, 1, H, W, 1, Ti)

# Step 5: Flatten to int8/uint8
ifm_flat = ifm_permuted.to(torch.int8).flatten().numpy()
```

**핵심 Permutation**: `(C_pkt, Ti, D, H, W) → (D, C_pkt, H, W, Ti)`

### 8.2 Weight

```python
# Original: (out_ch, in_ch, kD, kH, kW)
# Step 1: Spatial flip (normal conv only, not transpose)
if ofm_stride == 1:
    wgt = torch.flip(wgt, [2, 3, 4])

# Step 2: Transpose channel dims
wgt = wgt.permute(1, 0, 2, 3, 4)  # → (in_ch, out_ch, kD, kH, kW)

# Step 3: Pad channels
wgt_padded = pad(wgt, channel_padding)  # → (in_ch_pkt*Ti, out_ch_pkt*To, kD, kH, kW)

# Step 4: Pad spatial to 3×3×3
wgt_spatial_padded = pad(wgt_padded, spatial_padding)

# Step 5: Reshape to tiles
wgt_tiled = reshape(wgt, (in_ch_pkt, Ti, out_ch_pkt, To, kernel_space))
# kernel_space = kD * kH * kW (padded to 27 for k=3)

# Step 6: Flatten to int8
wgt_flat = wgt_tiled.to(torch.int8).flatten().numpy()
```

### 8.3 Bias

```python
# Original: (out_ch,)
# Pad to To boundary
bias_padded = pad(bias, (0, och_pad))  # → (out_ch_pkt * To,)
# Reshape
bias_reshaped = reshape(bias_padded, (out_ch_pkt, To))
# Flatten to int32
bias_flat = bias_reshaped.to(torch.int32).flatten().numpy()
```

### 8.4 OFM (Output Feature Map)

```python
# Hardware output layout: (out_d, out_ch_pkt, out_h, out_w, To)
# Need to transpose back to: (out_ch, out_d, out_h, out_w)
ofm_reshaped = reshape(ofm_raw, (out_d, out_ch_pkt, out_h, out_w, To))
ofm_transposed = permute(ofm_reshaped, (1, 4, 0, 2, 3))  # → (out_ch_pkt, To, D, H, W)
ofm_final = reshape(ofm_transposed, (out_ch_pkt * To, out_d, out_h, out_w))
# Trim to actual out_ch
ofm_trimmed = ofm_final[:out_ch]
```

---

## 9. Golden Reference (PyTorch)

```python
def create_ofm_ref(layer, wgt, bias, ifm, concat=None):
    if layer.is_concat:
        input_data = torch.cat((ifm, concat), dim=0)  # channel concat
    else:
        input_data = ifm

    if layer.ofm_stride == 1:
        # Normal Conv3D
        result = F.conv3d(input_data.unsqueeze(0), wgt, bias=bias,
                         stride=layer.ifm_stride,
                         padding=1 if layer.kernel_size == 3 else 0).to(torch.int32)
    else:
        # Transpose Conv3D
        result = F.conv_transpose3d(input_data.float(), wgt.float(),
                                    bias=bias.float(),
                                    stride=layer.ofm_stride).to(torch.int32)

    # Quantization pipeline
    shifted = result >> layer.bias_shift       # right shift
    if layer.is_relu:
        activated = F.relu(shifted)
        clipped = torch.clip(activated, 0, 255)    # uint8 range
    else:
        clipped = torch.clip(shifted, -128, 127)   # int8 range

    return clipped.squeeze(0)
```

### Output Comparison

- **Exact match**: `np.array_equal(ref, actual)`
- **±1 tolerance**: Rounding errors (acceptable)
- **>1 diff**: Significant error (investigate)

---

## 10. Conv3D Operation Details

### 10.1 Supported Configurations

| Parameter | Values | Description |
|-----------|--------|-------------|
| kernel_size | 1, 2, 3 | Maps to 1×1×1, 2×2×2, 3×3×3 (padded) |
| ifm_stride | 1, 2 | Downsampling stride |
| ofm_stride | 1, 2 | Upsampling (transpose conv) |
| is_relu | 0, 1 | ReLU activation |
| is_concat | 0, 1 | Skip connection enable |
| ifm_is_signed | 0, 1 | Signed input |
| bias_shift | 0-31 | Quantization shift |

### 10.2 Computation

```
output[och][d][h][w] = clip(relu(
    (Σ_{ich,kd,kh,kw} input[ich][d+kd][h+kh][w+kw] × weight[och][ich][kd][kh][kw] + bias[och])
    >> bias_shift
))
```

### 10.3 U-Net 3D Layer Example (28 layers)

```
Encoder:
  L0-1:  1→32ch  (conv3d 3×3×3, stride 1)
  L2:    32→64   (stride 2, downsample)
  L3-4:  64→128  (stride 2)
  L5-9:  128→320 (bottleneck)

Decoder:
  L12:   320→320 (ofm_stride=2, transpose conv, upsample)
  L13:   640→320 (concat with L9)
  L15:   320→256 (upsample)
  L16:   512→256 (concat with L7)
  ...
  L27:   32→3   (1×1×1 kernel, final output)
```

---

## 11. vTen Mapping

### 11.1 CompositeKernel 구조

NPU 3D는 vTen CompositeKernel로 모델링한다. 6개 sub-kernel이 각각 독립 IP에 대응.

```python
class NPU3DKernel(CompositeKernel):
    # Sub-kernels
    fmapIO = FmapIOKernel.bind(
        interface_map={"ctrl": External("ctrl_fmapio"),
                       "ddr":  External("ddr_fmap")})
    weight_loader = WeightLoaderKernel.bind(
        interface_map={"ctrl": External("ctrl_wgt"),
                       "hbm":  External("hbm_wgt")})  # split 32 ports
    mac_atu = MacAtuKernel.bind(
        interface_map={"ctrl": External("ctrl_mac")})
    psum_buffer = PsumBufferKernel.bind(
        interface_map={"ctrl": External("ctrl_psum")})
    bias_loader = BiasLoaderKernel.bind(
        interface_map={"ctrl": External("ctrl_bias"),
                       "ddr":  External("ddr_bias")})
    act_quant = ActQuantKernel.bind(
        interface_map={"ctrl": External("ctrl_act")})

    # Internal AXI-Stream connections (BFM 불필요)
    connections = [
        Connect(fmapIO.ifm_out, mac_atu.ifm_in),       # Internal
        Connect(weight_loader.wgt_out, mac_atu.wgt_in), # Internal
        Connect(mac_atu.psum_out, psum_buffer.psum_in), # Internal
        Connect(psum_buffer.out, act_quant.psum_in),    # Internal
        Connect(bias_loader.bias_out, act_quant.bias_in), # Internal
        Connect(act_quant.ofm_out, fmapIO.ofm_in),     # Internal
    ]
```

### 11.2 BFM Protocol Mapping

| NPU Interface | vTen BFM | Protocol | Role | Count |
|---------------|----------|----------|------|-------|
| 6× ctrl AXI4-Lite | vten_bfm_axilite | AXI4-Lite | MASTER | 6 |
| fmapIO DDR | vten_bfm_axi4 | AXI4 | SLAVE | 1 |
| bias_loader DDR | vten_bfm_axi4 | AXI4 | SLAVE | 1 |
| weight_loader HBM×32 | vten_bfm_axi4 | AXI4 | SLAVE | 32 |
| **Total** | | | | **40** |

> AXI4 BFM은 SLAVE (DUT가 MASTER). DUT의 AR/AW 요청에 응답하여 SHM 데이터를 제공/수집.
> AXI4-Lite BFM은 MASTER. 호스트가 레지스터를 write/read/poll.

### 11.3 Command Mapping

| Host Operation | vTen OpCode | Details |
|----------------|-------------|---------|
| IFM upload to DDR | LOAD | fmapIO DDR BFM에 데이터 적재 |
| Weight upload to HBM | LOAD ×32 | split 32 ports → 32 LOAD commands |
| Bias upload to DDR | LOAD | bias_loader DDR BFM에 데이터 적재 |
| Register write | WRITE_REG | Per-IP AXI4-Lite BFM |
| VSYNC trigger | WRITE_REG | Write 1 to VSYNC offset |
| Poll LAYER_DONE | POLL_REG | ctrl_fmapio, offset 0x054 |
| Read OFM from DDR | STORE | fmapIO DDR BFM에서 데이터 회수 |
| Compare golden | (Host-side) | run(verify=True)로 PyTorch ref vs HW output |

### 11.4 Scheduler 파라미터

NPU 3D의 40 BFMs를 수용하려면 Scheduler 파라미터 자동 상향 필요:

```toml
# vten.toml (자동 계산되지만 명시적 설정도 가능)
[backend.scheduler]
max_bfms = 40     # 6 AXI4-Lite + 2 DDR AXI4 + 32 HBM AXI4
max_ifaces = 42   # max_bfms + headroom
max_cmds = 256    # single layer 기준 충분
```

### 11.5 Data Signedness 주의사항

- `ifm_is_signed=0` → uint8, `ifm_is_signed=1` → int8
- `concat_is_signed` 별도 플래그
- Golden reference의 clip 범위가 달라짐: relu → [0,255], non-relu → [-128,127]

---

## 12. Debug Ports (검증용)

NPU_3D 모듈은 다음 debug 신호를 노출한다:

```systemverilog
output logic [255:0]    axi2ifm_dat;      // DDR→IFM data
output logic            axi2ifm_vld;
output logic [7:0]      ifm2mac_dat[32];  // IFM→MAC data (per tile)
output logic            ifm2mac_vld;
output logic [15:0]     act2ofm_dat;      // ACT→OFM data (2×8b)
output logic            act2ofm_vld;
output logic [255:0]    axi2wgt_dat[32];  // Weight data (per bank)
output logic            axi2wgt_vld[32];
output logic [255:0]    axi2bias_dat;     // Bias data
output logic            axi2bias_vld;
output logic [255:0]    ofm2axi_dat;      // OFM→DDR data
output logic            ofm2axi_vld;
output logic [63:0]     psum2act_dat;     // PSUM→ACT data
output logic            psum2act_vld;
```

### FSM States

**fmapIO**: `LAYER_IDLE → LAYER_PROCESSING → LAYER_PROCESS_DONE → LAYER_TRANSACTION_DONE → IDLE`

**OFM Writer**: `IDLE → PROCESS → LAYER_DONE → IDLE`

**Bias Loader**: `IDLE → PROCESS → IDLE`

---

## 13. 요약: 검증 시 핵심 체크포인트

1. **6-IP Register Configuration**: 6개 IP 커널의 레지스터를 정확한 순서/값으로 설정
2. **Tensor Layout**: `(D, C_pkt, H, W, Ti)` permutation 정확히 구현
3. **Channel Padding**: Ti/To=32 경계로 padding
4. **Weight Flip**: Normal conv는 spatial flip, transpose conv는 flip 없음
5. **32-Bank Weight Distribution**: 각 bank별 주소 계산
6. **Bias Shift**: Right-shift quantization 정확도
7. **LAYER_DONE Polling**: fmapIO ctrl, offset 0x054
8. **Output Comparison**: ±1 tolerance 허용
9. **Scheduler Scaling**: MAX_BFMS ≥ 40 (자동 계산)

---

## 14. Phase별 NPU 3D 테스트 매핑

> 각 Phase의 단위 테스트에서 NPU 3D의 실제 패턴을 참고하여 현실적인 테스트를 작성한다.
> `examples/conv3d/`는 Phase 5 (E2E Validation)에서 사용한다.

### Phase 1: Python Core (kernel/, dsl/, spec/)

**test_tensor.py** — NPU 3D 텐서 패턴
- IFM shape `(IN_CH, D, H, W)` — 파라미터 혼합: `("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}")`
- Weight shape `(OUT_CH, IN_CH, 3, 3, 3)` — 5D 텐서
- Bias shape `(OUT_CH,)` — 1D int32
- `fill_random`에서 int8/uint8 구분 (IFM_DTYPE에 따라)
- Ti=32 경계 채널 padding 검증: `in_ch=48` → resolved 후 numel 계산

**test_kernel.py** — NPU 3D Kernel 구조
- 5개 tensor descriptor (ifm, weight, bias, ofm, concat) + 6개 register handle 등록
- `__init_subclass__`로 tensor.name 자동 설정 확인
- `generate_inputs(seed)` — int8 IFM, int8 weight, int32 bias 생성
- `forward()` — conv3d golden reference (F.conv3d + bias_shift + relu/clip)
- verification — ±1 tolerance (기본 allclose 대신 커스텀)
- `bind()` — 6개 AXI4-Lite를 bank tuple `("ctrl", "dma")` 형태로 바인딩

**test_composite.py** — NPU 3D는 단일 Kernel이므로 직접 해당 없음
- 그러나 향후 multi-layer pipeline (여러 NPU_3D 인스턴스 체이닝)을 위해
  `Connect(layer1.ofm_proxy, layer2.ifm_proxy)` 패턴 테스트

**test_spec_parser.py** — NPU 3D kernel_spec.yaml 파싱
- 6개 AXI4-Lite 인터페이스 파싱 (각각 독립 register set)
- AXI4 DDR: `data_width=256`, `addr_width=64`, `memory_region=ddr`, `tensors=[ifm,ofm,concat]`
- AXI4 HBM: `split.mode=channel_interleave`, 32개 ports
- `auto_bind` 패턴들:
  - `{tensor: ifm, value: address, bits: "31:0"}` — 64-bit 주소 split
  - `{tensor: bias, value: size_bytes}` — 텐서 크기
  - `{param: "${OUT_CH}"}` — 파라미터 직접 바인딩
- `packing.bus_width` 검증: 256-bit AXI4에 8-bit × 32 = 256-bit (exact match)
- register offset 범위: bias_loader 0x010~0x01C, weight_loader 0x010~0x128

**test_dsl.py** — NPU 3D 실행 시퀀스
- Host workflow operation chain (§7.1):
  `LOAD(ifm) → LOAD(wgt) → LOAD(bias) → CONFIGURE → WRITE_REG(vsync) × 3 → POLL_REG(layer_done) → STORE(ofm)`
- 의존성: configure는 3개 load에 dep, fmapIO vsync는 bias/wgt vsync에 dep
- `add_commit_dependency`: pull의 commit이 poll_register에 의존
- 6개 IP kernel의 VSYNC trigger 순서: bias_loader → weight_loader → fmapIO

### Phase 2: Runtime Engine (runtime/)

**test_runtime_resolver.py** — NPU 3D 파라미터
- 런타임 파라미터: `IN_CH=64, OUT_CH=128, IN_DEPTH=8, ...`
- 산술 표현식: `"${IN_CH}//32"` → `in_ch_pkt`
- 고정 + 변수 혼합: `Ti=32` (고정) + `IN_CH="${IN_CH}"` (변수)
- 파라미터 우선순위: runtime > kernel_spec > project

**test_runtime_serializer.py** — NPU 3D 텐서 직렬화
- IFM layout transform: `(C, D, H, W)` → pad → `(D, C_pkt, H, W, Ti)` → flatten → bytes
- Weight transform: spatial flip → transpose → pad → tile → bytes
- Bias: `(O,)` → pad to To → `(O_pkt, To)` → int32 bytes
- OFM inverse: `(D, O_pkt, H, W, To)` → `(O, D, H, W)`
- int8 packing: `element_width=8, elements_per_beat=32` → 256-bit 단위
- int32 bias packing: `element_width=32, elements_per_beat=8` → 256-bit 단위

**test_runtime_address.py** — NPU 3D 메모리 할당
- DDR region: IFM, OFM, Concat은 같은 ddr_fmap 인터페이스 → 주소 겹치지 않게 할당
- DDR region: Bias는 ddr_bias 인터페이스 → 별도 할당
- HBM 32-bank: 각 bank별 weight 주소 계산
- 4096-byte alignment
- `set_address()` 수동 오버라이드 후 configure 반영

**test_runtime_ir.py** — NPU 3D IR lowering
- CONFIGURE → 6개 IP의 auto_bind register를 N × WRITE_REG로 확장
- PUSH_TENSOR → LOAD + PUSH (IFM, Weight, Bias 각각)
- Weight PUSH_TENSOR → split 32포트이므로 32× LOAD + PUSH commands
- WRITE_REG(vsync) → 1× WRITE_REG per IP
- POLL_REG(layer_done) → 1× POLL_REG (ctrl_fmapio 0x054)
- PULL_TENSOR → PULL + STORE (OFM)

**test_runtime_shm.py** — SHM 이미지
- Command slot: LOAD cmd에 `phys_addr`, `size`, `buffer_id` 정확히 인코딩
- WRITE_REG cmd: `reg_offset`, `reg_value` 인코딩
- POLL_REG cmd: `reg_offset=0x054`, `reg_mask`, `reg_expected` 인코딩
- Buffer descriptor: IFM/Weight/Bias/OFM 각각 direction, size
- Data region: 직렬화된 텐서 데이터 (IFM int8, Bias int32)
- 전체 SHM 크기 계산 검증

### Phase 3: SV + C Backend (vten_sv/)

- BFM 인스턴스: AXI4 slave ×34 (DDR 2 + HBM 32) + AXI4-Lite master ×6
- DDR BFM: 256-bit burst read/write, max 128 beats
- AXI4-Lite BFM: 16-bit addr, 32-bit data
- Scheduler: MAX_BFMS=40, MAX_IFACES=42
- DPI-C: SHM region mmap, command fetch, data copy

### Phase 4: Integration (codegen/, cli/, backend/)

- `vten build`: 6-IP RTL + vten_sv BFM → xsim elaboration
- `vten run`: SHM 이미지 전달 → xsim 실행 → 결과 수집
- Jinja2 template: tb_top에 DDR 2포트 + HBM 32포트 + AXI4-Lite 6포트 BFM 인스턴스화
- Scheduler 파라미터 자동 계산 (§3.4 of 06_codegen_and_cli.md)

### Phase 5: E2E Validation (examples/conv3d/)

- **`examples/conv3d/`** 디렉토리 사용
  - `specs/npu_3d.yaml` — kernel_spec
  - `kernels/npu_3d_kernel.py` — Kernel 클래스 + golden reference
  - `tests/test_conv3d.py` — 전체 파이프라인 테스트
- 테스트 케이스:
  - 3×3×3 normal conv (stride=1, relu, bias_shift=8)
  - 1×1×1 pointwise conv (final layer)
  - stride-2 downsample (encoder)
  - transpose conv upsample (decoder, ofm_stride=2)
  - skip connection (is_concat=1)
  - U-Net 28-layer 순차 실행
- 검증: ±1 tolerance, max_diff 통계
