# SV Interface Array 지원 및 포트 네이밍 규칙

> NPU_3D 실제 디자인 분석 기반으로, wrapper.sv.j2 템플릿이 production RTL을
> 완전히 대체하기 위해 필요한 SV interface array 지원과 Vitis 호환 포트 네이밍을
> 정리한다.

---

## 1. 배경

NPU_3D 커널들은 core 내부에서 **SV interface 배열**을 사용한다:

```systemverilog
// mac_atu.sv
axis.slave  __WGT2MAC_axis[Ti][OUT_GROUP]     // 2D: [32][2] = 64 streams
axis.master __ATU2PSUM_axis[ATU_AXIS_OUT]     // 1D: [6] = 6 streams

// weight_loader_intf_port.sv
aximm.master __wgt_aximm[Ti]                  // 1D: [32] = 32 AXI-MM ports
axis.master  __wgt_axis[Ti][OUT_GROUP]        // 2D: [32][2] = 64 streams

// psum_buffer_intf_port.sv
axis.slave   __psum_in_axis[6]                // 1D: [6] = 6 streams
```

현재 vten wrapper 템플릿은 인터페이스를 **개별 인스턴스**로만 생성한다:

```jinja2
vten_axis_if #(.DATA_W({{ dw }})) {{ iface.name }}_if();
```

이 방식으로는 64개 인터페이스를 개별 선언하고 core에 개별 연결해야 하는데,
core는 `__wgt_axis[32][2]` 같은 배열 포트를 기대하므로 연결이 불가능하다.

---

## 2. 설계 원칙: `name` 기반 단일 이름 규칙

### 핵심 원칙

**사용자는 `name`만 지정하면 나머지 이름이 자동으로 결정된다.**

| 결정되는 이름 | 규칙 | 예시 (`name: wgt`) |
|--------------|------|-------------------|
| core 포트 | `name` 그대로 | `.wgt(wgt_if)` |
| 외부 flat 포트 | `{prefix}_{name}_{signal}` | `s_axis_wgt_0_0_tdata` |
| 내부 SV interface | `{name}_if` | `wgt_if[32][2]` |

- **`core_port` 필드는 없다** — core 모듈의 포트 이름을 spec의 `name`과 일치시킨다.
- **`rtl_port`는 선택적 override** — 기본값과 다른 외부 포트 이름이 필요할 때만 지정한다.

### rtl_port 기본값 자동 생성

`rtl_port`가 생략되면 protocol + role + name으로 자동 생성:

```python
prefix = {
    ("axi4_stream", "master"): "m_axis_",
    ("axi4_stream", "slave"):  "s_axis_",
    ("axi4", "master"):        "m_axi_",
    ("axi4", "slave"):         "s_axi_",
    ("axi4_lite", "slave"):    "s_axilite_",
    ("axi4_lite", "master"):   "m_axilite_",
}
rtl_port = prefix[(protocol, role)] + name
```

### Core 리팩토링 규칙

기존 core의 포트 이름을 spec의 `name`과 일치시킨다:

```systemverilog
// Before (mac_atu.sv)
module mac_atu (
    axilite.slave __mac_atu_axilite_intf,
    axis.slave    __WGT2MAC_axis[Ti][OUT_GROUP],
    axis.master   __ATU2PSUM_axis[ATU_AXIS_OUT],
    ...
);

// After (mac_atu_core.sv)
module mac_atu_core (
    input logic [0:0] reg_ifm_is_signed,  // ctrl 제거 → flat wire
    ...
    axis.slave  wgt[Ti][OUT_GROUP],        // name: wgt
    axis.master psum[ATU_AXIS_OUT],        // name: psum
    ...
);
```

---

## 3. 변경 사항

### 3.1 Interface Array 그룹 지원

#### 문제

Core가 SV interface 배열을 포트로 사용하면, wrapper는:
1. 외부에는 **flat 포트를 개별 선언** (Vitis/BFM 호환)
2. 내부에는 **SV interface 배열을 선언**
3. flat 포트와 배열 요소를 **인덱스로 매핑**

이 패턴은 NPU_3D의 기존 수동 wrapper에서 이미 사용 중이다:

```systemverilog
// mac_atu_top_wrapper.sv — 현재 수동 구현
axis #(.DATA_WIDTH(256)) __wgt_axis[32][2] ();         // 배열 선언

`AXIS_ASSIGN_SLAVE_TO_FLAT(wgt_0_0, __wgt_axis[0][0])  // flat → 배열[0][0]
`AXIS_ASSIGN_SLAVE_TO_FLAT(wgt_0_1, __wgt_axis[0][1])  // flat → 배열[0][1]
`AXIS_ASSIGN_SLAVE_TO_FLAT(wgt_1_0, __wgt_axis[1][0])  // flat → 배열[1][0]
// ... 64개 반복

mac_atu mac_atu_inst (
    .__WGT2MAC_axis(__wgt_axis),   // 배열 통째로 연결
    ...
);
```

#### kernel_spec.yaml 표현 방식

```yaml
interfaces:
  # --- 1D 배열: psum_buffer의 axis slave [6] ---
  psum_in:
    protocol: axi4_stream
    role: slave
    data_width: 256
    array:
      dimensions: [6]
      flat_name_pattern: "in_{i}"
    # rtl_port 생략 → 자동: s_axis_in_0, s_axis_in_1, ...
    # core 포트: .psum_in(psum_in_if)

  # --- 2D 배열: mac_atu의 axis slave [32][2] ---
  wgt:
    protocol: axi4_stream
    role: slave
    data_width: 256
    array:
      dimensions: [32, 2]
      flat_name_pattern: "wgt_{i}_{j}"
    # rtl_port 생략 → 자동: s_axis_wgt_0_0, s_axis_wgt_0_1, ...
    # core 포트: .wgt(wgt_if)

  # --- 1D 배열: weight_loader의 aximm master [32] ---
  wgt_dma:
    protocol: axi4
    role: master
    data_width: 256
    addr_width: 64
    array:
      dimensions: [32]
      flat_name_pattern: "dma_{i}"
    # rtl_port 생략 → 자동: m_axi_dma_0, m_axi_dma_1, ...
    # core 포트: .wgt_dma(wgt_dma_if)

  # --- 단일 인터페이스 (기존 방식, 변경 없음) ---
  ifm:
    protocol: axi4_stream
    role: slave
    data_width: 256
    # rtl_port 생략 → 자동: s_axis_ifm
    # core 포트: .ifm(ifm_if)
```

#### 생성 코드 패턴

`array` 필드가 있으면 wrapper 템플릿이 다음을 생성:

```systemverilog
// ---- 1. Flat 포트 선언 (외부 Vitis/BFM 호환) ----
// array.flat_name_pattern = "wgt_{i}_{j}", prefix = s_axis_
input  logic [255:0] s_axis_wgt_0_0_tdata,
input  logic         s_axis_wgt_0_0_tvalid,
output logic         s_axis_wgt_0_0_tready,
input  logic         s_axis_wgt_0_0_tlast,
// ... 64개 반복 ...

// ---- 2. SV interface 배열 선언 (내부) ----
vten_axis_if #(.DATA_W(256)) wgt_if[32][2]();

// ---- 3. Flat ↔ 배열 요소 매핑 ----
assign wgt_if[0][0].tdata  = s_axis_wgt_0_0_tdata;
assign wgt_if[0][0].tvalid = s_axis_wgt_0_0_tvalid;
assign s_axis_wgt_0_0_tready = wgt_if[0][0].tready;
assign wgt_if[0][0].tlast  = s_axis_wgt_0_0_tlast;
// ... 64개 반복 ...

// ---- 4. Core 연결 (name 기반) ----
mac_atu_core u_core (
    .wgt(wgt_if),       // name="wgt" → .wgt(wgt_if)
    .psum(psum_if),     // name="psum" → .psum(psum_if)
    .ifm(ifm_if),       // 단일 인터페이스도 동일 규칙
    ...
);
```

#### flat 포트 이름 생성 규칙

배열 인터페이스의 각 요소에 대한 flat 포트 이름:

```
{prefix} + flat_name_pattern.format(i=..., j=...) + "_{signal}"
```

예시:
- `flat_name_pattern: "wgt_{i}_{j}"`, role=slave, protocol=axi4_stream
- `[0][0]` → `s_axis_wgt_0_0_tdata`, `s_axis_wgt_0_0_tvalid`, ...
- `[31][1]` → `s_axis_wgt_31_1_tdata`, `s_axis_wgt_31_1_tvalid`, ...

`flat_name_pattern`이 생략되면 `name`에 인덱스를 붙여 자동 생성:
- 1D: `{name}_{i}` → `psum_0`, `psum_1`, ...
- 2D: `{name}_{i}_{j}` → `wgt_0_0`, `wgt_0_1`, ...

#### 구현 범위

| 항목 | 세부 |
|------|------|
| **models.py** | `ArraySpec` dataclass 추가: `dimensions: list[int]`, `flat_name_pattern: str \| None` |
| **models.py** | `InterfaceSpec`에 `array: ArraySpec \| None` 필드 추가 |
| **models.py** | `InterfaceSpec.rtl_port` 기본값 로직 (protocol + role + name) |
| **wrapper.sv.j2** | `array` 필드 존재 시 flat 포트를 루프로 선언 |
| **wrapper.sv.j2** | SV interface 배열 인스턴스 생성 |
| **wrapper.sv.j2** | flat ↔ 배열 요소 assign 생성 |
| **wrapper.sv.j2** | core 연결 시 `name` 기반으로 연결 (`array` 유무 무관) |
| **kernel_spec.yaml parser** | `array` 필드 파싱 |

---

### 3.2 Vitis 호환 포트 네이밍

#### NPU_3D port.svh 네이밍 규칙

| 프로토콜 | Role | 접두사 | 예시 |
|----------|------|--------|------|
| AXI4-Stream | master | `m_axis_` | `m_axis_psum_0_tdata` |
| AXI4-Stream | slave | `s_axis_` | `s_axis_wgt_0_0_tdata` |
| AXI4 (full) | master | `m_axi_` | `m_axi_dma_0_awaddr` |
| AXI4 (full) | slave | `s_axi_` | `s_axi_dma_0_awaddr` |
| AXI4-Lite | slave | `s_axilite_` | `s_axilite_control_awaddr` |
| AXI4-Lite | master | `m_axilite_` | `m_axilite_control_awaddr` |

이 접두사는 Vitis v++ linker의 `stream_connect`, `sp` 설정에서 참조된다:

```ini
# NPU_3D_krnl.cfg
stream_connect=weight_loader_1.m_axis_wgt_0_0:mac_atu_1.s_axis_wgt_0_0
stream_connect=mac_atu_1.m_axis_psum_0:psum_buffer_1.s_axis_in_0
```

#### rtl_port override 예시

기본값이 맞지 않는 경우에만 `rtl_port`를 명시:

```yaml
interfaces:
  ctrl:
    protocol: axi4_lite
    role: slave
    rtl_port: s_axilite_control   # 기본값 "s_axilite_ctrl"과 다름
    generate_controller: true
```

#### Vivado X_INTERFACE pragmas (선택사항)

xclbin 빌드를 위해 `(* X_INTERFACE_INFO *)` pragma가 필요하다.
현재 NPU_3D는 이를 수동으로 작성하거나 pack_kernel.tcl에서 처리한다.

```systemverilog
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite_control:m_axis_psum_0:..., ASSOCIATED_RESET ap_aresetn" *)
input ap_clk,
```

vten wrapper에서 자동 생성하면 pack_kernel.tcl 의존성을 제거할 수 있으나,
pack_kernel.tcl 방식을 유지해도 무방하다.

---

## 4. NPU_3D 커널별 배열 인터페이스 요약

| 커널 | spec name | Protocol | Role | 배열 | flat 이름 패턴 |
|------|-----------|----------|------|------|----------------|
| **mac_atu** | wgt | axis | slave | [32][2] | `wgt_{i}_{j}` |
| **mac_atu** | psum | axis | master | [6] | `psum_{i}` |
| **weight_loader** | wgt_dma | aximm | master | [32] | `dma_{i}` |
| **weight_loader** | wgt | axis | master | [32][2] | `wgt_{i}_{j}` |
| **psum_buffer** | psum_in | axis | slave | [6] | `in_{i}` |
| **fmapIO** | — | — | — | 없음 | — |
| **act_quant** | — | — | — | 없음 | — |
| **bias_loader** | — | — | — | 없음 | — |

---

## 5. 설계 시 고려 사항

### 5.1 `flat_name_pattern` 인덱스 변수

- 1D: `{i}` (0부터 시작)
- 2D: `{i}`, `{j}` (각각 첫 번째, 두 번째 차원)
- 생략 시 기본값: 1D → `{name}_{i}`, 2D → `{name}_{i}_{j}`
- Python `str.format(i=..., j=...)` 또는 Jinja2 루프로 처리

### 5.2 Interface 배열 선언 시 simulator 호환성

```systemverilog
// Vivado xsim, Verilator 모두 지원하는 문법:
vten_axis_if #(.DATA_W(256)) wgt_if[32][2]();
```

Verilator는 interface 배열을 지원하지만, **generate 블록 내 interface 인스턴스**는
제한이 있을 수 있으므로 top-level 선언을 권장한다.

### 5.3 비배열 인터페이스와의 혼용

하나의 커널에 배열 인터페이스와 단일 인터페이스가 공존한다.
템플릿은 `array` 필드 유무로 분기하면 된다.

```yaml
interfaces:
  ctrl:
    protocol: axi4_lite
    role: slave
    # 단일 인터페이스 (기존 방식)
  ifm:
    protocol: axi4_stream
    role: slave
    # 단일 인터페이스 (기존 방식)
  wgt:
    protocol: axi4_stream
    role: slave
    array:
      dimensions: [32, 2]
      # 배열 인터페이스 (신규)
```

Core 연결 코드는 배열/단일 구분 없이 동일한 규칙:
```systemverilog
u_core (
    .ifm(ifm_if),           // 단일: vten_axis_if ifm_if()
    .wgt(wgt_if),           // 배열: vten_axis_if wgt_if[32][2]()
    .reg_xxx(reg_xxx),      // 레지스터: 기존 방식 동일
);
```

---

## 6. 우선순위

| 순위 | 항목 | 이유 |
|------|------|------|
| **P0** | Interface array (3.1) | mac_atu, weight_loader wrapper 대체에 필수 |
| **P1** | rtl_port 기본값 (3.2) | 편의성, v++ 설정 파일과의 일관성 |
| **P2** | X_INTERFACE pragma 자동 생성 | pack_kernel.tcl 의존성 제거 (선택) |

---

## 7. 참고: 기존 NPU_3D 코드 위치

| 파일 | 역할 |
|------|------|
| `design/mac/rtl/mac_atu_top_wrapper.sv` | 2D 배열 (64 axis) wrapper 예시 |
| `design/mac/rtl/mac_atu.sv` | Core에서 배열 포트 사용 예시 |
| `design/weight_loader/rtl/weight_loader_top.sv` | 32 aximm + 64 axis wrapper |
| `design/weight_loader/rtl/weight_loader_intf_port.sv` | 배열 인터페이스 선언 |
| `design/psum_buffer/rtl/psum_buffer_intf_port.sv` | 1D 배열 인터페이스 선언 |
| `lib/axi/port.svh` | flat 포트 매크로 (네이밍 규칙 원본) |
| `NPU_3D_krnl.cfg` | v++ link config (포트 이름 참조) |
