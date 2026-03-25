# vTen kernel_spec.yaml Complete Schema

**Version 0.4.2 — March 2026**
**참조 모델: `00_data_models.md` (KernelSpec, InterfaceSpec, PackingScheme 등)**
**Status: Phase 1 구현 전 완성 필요**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Top-Level Structure](#2-top-level-structure)
3. [parameters Section](#3-parameters-section)
4. [memory_regions Section](#4-memory_regions-section)
5. [interfaces Section](#5-interfaces-section)
6. [AXI4-Stream Interface](#6-axi4-stream-interface)
7. [AXI4 Memory-Mapped Interface](#7-axi4-memory-mapped-interface)
8. [AXI4-Lite Interface](#8-axi4-lite-interface)
9. [packing Sub-Schema](#9-packing-sub-schema)
10. [split Sub-Schema](#10-split-sub-schema)
11. [registers Sub-Schema](#11-registers-sub-schema)
12. [auto_bind Sub-Schema](#12-auto_bind-sub-schema)
13. [register_banks Sub-Schema](#13-register_banks-sub-schema)
14. [Parsing Implementation Guide](#14-parsing-implementation-guide)
15. [vten spec --detect](#15-vten-spec---detect)

---

## 1. Overview

`kernel_spec.yaml`은 하나의 RTL 모듈에 대한 인터페이스 사양을 정의한다. RTL 소스를 수정하지 않으며, 모든 의미론적 매핑(텐서↔포트, dtype, 패킹)을 이 파일에 집중한다.

파싱 결과는 `KernelSpec` dataclass (`00_data_models.md` §5.8)로 변환된다.

### 1.1 파일 위치

**v0.5.0부터 커널별 디렉토리 구조 사용:**

```
PROJECT_ROOT/kernels/<kernel_name>/kernel_spec.yaml
```

예시:
```
my_npu/
├── kernels/
│   ├── conv3d/
│   │   └── kernel_spec.yaml      # Conv3D 커널 인터페이스 사양
│   ├── dma_ifm/
│   │   └── kernel_spec.yaml
│   └── npu_top/
│       └── kernel_spec.yaml      # CompositeKernel 사양
```

`kernel_spec.yaml`은 반드시 `kernels/<name>/` 디렉토리에 위치해야 한다. `vten build`는 이 경로만 탐색한다:

```
kernels/<kernel_name>/kernel_spec.yaml
```

---

## 2. Top-Level Structure

```yaml
kernel: <string>           # REQUIRED. 커널 이름 (Python 클래스와 매칭)
rtl_top: <string>          # REQUIRED. RTL 탑 모듈 파일 경로

parameters:                # OPTIONAL. 커널 레벨 파라미터
  <key>: <value>

memory_regions:            # OPTIONAL. 메모리 영역 정의 (AXI4 인터페이스용)
  <region_name>: { ... }

clock:                     # OPTIONAL. 클럭 설정
  name: <string>           # 기본값: "clk"

reset:                     # OPTIONAL. 리셋 설정
  name: <string>           # 기본값: "rst_n"
  active_low: <bool>       # 기본값: true

interfaces:                # REQUIRED. 인터페이스 정의
  <interface_name>: { ... }
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `kernel` | string | ✓ | — | 커널 이름 |
| `rtl_top` | string | ✓ | — | RTL 탑 모듈 파일 경로 |
| `parameters` | dict | — | {} | 파라미터 키-값 |
| `memory_regions` | dict | — | {} | 메모리 영역 |
| `clock` | dict | — | `{name: "clk"}` | 클럭 신호 설정 |
| `reset` | dict | — | `{name: "rst_n", active_low: true}` | 리셋 신호 설정 |
| `interfaces` | dict | ✓ | — | 인터페이스 (최소 1개) |

> **예시 (NPU_3D):** `clock: {name: ap_clk}`, `reset: {name: ap_aresetn, active_low: true}`

---

## 3. parameters Section

```yaml
parameters:
  C: "${C}"           # 런타임에서 해결될 변수
  H: 32               # 고정 정수
  K: 128
  STRIDE: 1
```

값은 **정수** 또는 **`${name}` 표현식 문자열**.
표현식 내 산술 지원: `"${C}//${TILE_C}"`, `"(${D}-${KD})//${STRIDE}+1"`

---

## 4. memory_regions Section

```yaml
memory_regions:
  ddr:
    base: 0x0000_0000
    size: 0x1_0000_0000
    alignment: 4096
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `base` | int (hex OK) | ✓ | — | 시작 주소 |
| `size` | int (hex OK) | ✓ | — | 영역 크기 (바이트) |
| `alignment` | int | — | 4096 | 텐서 할당 정렬 (바이트) |

AXI4 인터페이스에서 `memory_region` 필드로 참조된다.

---

## 5. interfaces Section

각 인터페이스는 `protocol` 필드에 따라 필수 서브필드가 달라진다.

```yaml
interfaces:
  <name>:
    rtl_port: <string>       # REQUIRED. RTL 포트 접두사
    protocol: <protocol>     # REQUIRED. axi4_stream | axi4 | axi4_lite
    ... (프로토콜별 필드)
```

---

## 6. AXI4-Stream Interface

```yaml
ifm_stream:
  rtl_port: s_axis_ifm
  protocol: axi4_stream
  tensor: ifm                # REQUIRED. 바인딩된 텐서 이름
  packing:                   # REQUIRED. 패킹 사양
    element_width: 8
    elements_per_beat: 32
    bit_order: lsb_first
```

| 필드 | 타입 | 필수 | 기본값 |
|------|------|------|--------|
| `rtl_port` | string | ✓ | — |
| `protocol` | "axi4_stream" | ✓ | — |
| `tensor` | string | ✓ | — |
| `packing` | PackingSpec | ✓ | — |

`data_width`는 packing에서 유추: `element_width × elements_per_beat` (standard 모드).

---

## 7. AXI4 Memory-Mapped Interface

```yaml
data_port:
  rtl_port: m_axi_data
  protocol: axi4
  data_width: 256              # REQUIRED. 데이터 버스 폭 (비트)
  addr_width: 64               # OPTIONAL. 주소 버스 폭 (비트, 기본 64)
  memory_region: ddr           # REQUIRED. 참조 메모리 영역
  tensors: [ifm, weight, ofm]  # 복수 텐서 공유 시 리스트
  # 또는 tensor: ifm           # 단일 텐서
  packing:                     # REQUIRED
    element_width: 8
    elements_per_beat: 32
    alignment: packed
  split:                       # OPTIONAL. 멀티포트 분할
    mode: channel_interleave
    ports: [...]
    interleave: { unit: 4096 }
```

| 필드 | 타입 | 필수 | 기본값 |
|------|------|------|--------|
| `rtl_port` | string | ✓ | — |
| `protocol` | "axi4" | ✓ | — |
| `data_width` | int | ✓ | — |
| `addr_width` | int | — | 64 |
| `memory_region` | string | ✓ | — |
| `tensor` 또는 `tensors` | string 또는 list[string] | ✓ | — |
| `packing` | PackingSpec | ✓ | — |
| `split` | SplitSpec | — | None |

`tensor`와 `tensors`는 상호 배타적. 하나만 지정.

---

## 8. AXI4-Lite Interface

```yaml
ctrl:
  rtl_port: s_axilite_ctrl
  protocol: axi4_lite
  addr_width: 32               # OPTIONAL. 주소 버스 폭 (비트, 기본 32)
  registers:                   # REQUIRED (register_banks 없을 때)
    - name: start
      offset: 0x00
      fields: { go: "0:0" }
    - name: status
      offset: 0x04
      fields: { done: "0:0", busy: "1:1" }
    - name: ifm_base_lo
      offset: 0x10
      auto_bind: { tensor: ifm, value: address, bits: "31:0" }
  register_banks:              # OPTIONAL. 복수 서브커널 뱅크
    dma_ifm:    { base_offset: 0x000 }
    mac:        { base_offset: 0x200 }
```

| 필드 | 타입 | 필수 | 기본값 |
|------|------|------|--------|
| `rtl_port` | string | ✓ | — |
| `protocol` | "axi4_lite" | ✓ | — |
| `addr_width` | int | — | 32 |
| `registers` | list[RegisterSpec] | ✓ | — |
| `register_banks` | dict | — | None |

---

## 9. packing Sub-Schema

```yaml
# Standard mode
packing:
  element_width: 8           # REQUIRED. 원소 비트 폭
  elements_per_beat: 32      # REQUIRED. 비트당 원소 수
  bit_order: lsb_first       # OPTIONAL. lsb_first | msb_first
  alignment: packed           # OPTIONAL. packed | aligned
  byte_order: little          # OPTIONAL. little | big

# Custom mode
packing:
  mode: custom
  fields:
    - { name: data_a, bits: [0, 23] }
    - { name: data_b, bits: [24, 47] }
    - { name: valid_mask, bits: [48, 49] }
```

| 필드 | 타입 | 필수 | 기본값 |
|------|------|------|--------|
| `element_width` | int | ✓ (standard) | — |
| `elements_per_beat` | int | ✓ (standard) | — |
| `bit_order` | string | — | "lsb_first" |
| `alignment` | string | — | "packed" |
| `byte_order` | string | — | "little" |
| `mode` | string | — | "standard" |
| `fields` | list | ✓ (custom) | — |

### 9.1 Packing Width Constraint (v0.4.2)

`bus_width`는 PackingScheme에서 계산되는 값이다 (`00_data_models.md` §5.1).

**AXI4 인터페이스:**

```
packing.bus_width ≤ interface.data_width
```

- **같은 경우 (bus_width == data_width):** 완전 활용. 패딩 없음.
- **작은 경우 (bus_width < data_width):** 차이 비트는 zero-padding. LSB에 유효 데이터, MSB에 0 패딩. 경고 출력.
- **큰 경우 (bus_width > data_width):** `SpecValidationError` — 한 비트에 버스 폭보다 많은 데이터를 패킹할 수 없음.

**AXI4-Stream 인터페이스:**

`data_width`가 명시적으로 선언되지 않는다. 이 경우 `data_width = packing.bus_width`로 암묵 추론. RTL 포트의 `tdata` 폭과 일치해야 하며, 불일치는 `vten build` 시 RTL 포트 폭 검증 단계에서 감지.

**파서 검증 구현:**

```python
def _validate_packing(self, iface: InterfaceSpec):
    """Packing 유효성 검증."""
    if iface.packing is None:
        return

    # custom mode: 필드 간 비트 겹침 검사 (00_data_models.md §5.1)
    # 겹치면 ValidationError raise. allow_overlap 옵션 미지원.
    iface.packing.validate_custom_fields()

    bus_width = iface.packing.bus_width  # 계산된 패킹 폭

    if iface.protocol == Protocol.AXI4:
        if iface.data_width is None:
            raise SpecValidationError(
                f"AXI4 interface '{iface.name}' requires explicit data_width"
            )
        if bus_width > iface.data_width:
            raise SpecValidationError(
                f"Interface '{iface.name}': packing bus_width ({bus_width}) "
                f"exceeds data_width ({iface.data_width}). "
                f"Reduce elements_per_beat or element_width."
            )
        if bus_width < iface.data_width:
            import warnings
            warnings.warn(
                f"Interface '{iface.name}': packing bus_width ({bus_width}) "
                f"< data_width ({iface.data_width}). "
                f"Upper {iface.data_width - bus_width} bits will be zero-padded."
            )

    elif iface.protocol == Protocol.AXI4S:
        # AXI4-Stream: data_width = bus_width (암묵적)
        # RTL 포트 폭 검증은 vten build 시 수행
        pass
```

---

## 10. split Sub-Schema

```yaml
split:
  mode: channel_interleave     # REQUIRED. channel_interleave | block_split
  ports:                       # REQUIRED. 포트 리스트
    - { name: hbm_ch0, base_addr: 0x00000000 }
    - { name: hbm_ch1, base_addr: 0x00000000 }
  interleave:                  # channel_interleave 시 필수
    unit: 4096                 # 라운드 로빈 단위 (바이트)
```

---

## 11. registers Sub-Schema

```yaml
registers:
  - name: <string>             # REQUIRED. 레지스터 이름
    offset: <int>              # REQUIRED. 바이트 오프셋
    fields:                    # OPTIONAL. 비트 필드 매핑
      <field_name>: "<hi>:<lo>"
    auto_bind:                 # OPTIONAL. 자동 바인딩
      { ... }
```

---

## 12. auto_bind Sub-Schema

```yaml
# 텐서 주소
auto_bind: { tensor: ifm, value: address, bits: "31:0" }

# 텐서 크기
auto_bind: { tensor: ifm, value: size_bytes }
auto_bind: { tensor: ifm, value: size_beats }
auto_bind: { tensor: ifm, value: size_elements }

# 파라미터
auto_bind: { param: "${C}" }

# 표현식
auto_bind: { expr: "${N}*${K}*${D}*${H}*${W}" }
```

| 필드 | 타입 | 조건 |
|------|------|------|
| `tensor` | string | value가 address/size_* 일 때 |
| `value` | string | tensor 지정 시 필수. address\|size_bytes\|size_beats\|size_elements |
| `bits` | string | value=address일 때만. "31:0" 등 |
| `param` | string | tensor/value 없을 때 |
| `expr` | string | tensor/value/param 없을 때 |

---

## 13. register_banks Sub-Schema

CompositeKernel에서 복수 서브커널이 하나의 AXI-Lite 인터페이스를 공유할 때:

```yaml
register_banks:
  dma_ifm:    { base_offset: 0x000 }
  dma_weight: { base_offset: 0x100 }
  mac:        { base_offset: 0x200 }
  dma_ofm:    { base_offset: 0x300 }
  global:     { base_offset: 0x400 }
```

뱅크 간 주소 겹침 검증 필수 (V3: BankOverlapError).

---

## 14. Parsing Implementation Guide

### 14.1 파싱 흐름

```python
def load_kernel_spec(yaml_path: str) -> KernelSpec:
    raw = yaml.safe_load(open(yaml_path))
    validate_top_level(raw)

    parameters = raw.get('parameters', {})
    memory_regions = {
        name: parse_memory_region(name, spec)
        for name, spec in raw.get('memory_regions', {}).items()
    }
    interfaces = {
        name: parse_interface(name, spec)
        for name, spec in raw['interfaces'].items()
    }

    return KernelSpec(
        kernel_name=raw['kernel'],
        rtl_top=raw['rtl_top'],
        parameters=parameters,
        memory_regions=memory_regions,
        interfaces=interfaces,
    )
```

### 14.2 검증 규칙

1. `kernel`, `rtl_top`, `interfaces` 필수
2. 각 인터페이스에 `rtl_port`, `protocol` 필수
3. 프로토콜별 필수 필드 검증 (§6-8)
4. AXI4 인터페이스의 `memory_region`이 `memory_regions`에 존재하는지 확인
5. `tensor`/`tensors` 상호 배타성 검증
6. `auto_bind` 필드 조합 유효성 (tensor+value vs param vs expr)
7. `register_banks` 정의 시 주소 겹침 검증

---

## 15. vten spec --detect

### 15.1 동작

```bash
$ vten spec --detect rtl/conv3d_top.sv
```

1. RTL 파일 파싱 → 포트 이름, 방향(input/output/inout), 비트 폭 추출
2. 포트 이름 패턴으로 프로토콜 추론:
   - `s_axis_*`, `m_axis_*` → AXI4-Stream
   - `m_axi_*`, `s_axi_*` → AXI4
   - `s_axilite_*` → AXI4-Lite
3. 스켈레톤 YAML 생성 (TODO 마커 포함)

### 15.2 생성 예시

```yaml
# Auto-generated by vten spec --detect
# TODO: Fill in tensor names, packing, and auto_bind specifications

kernel: conv3d_top
rtl_top: rtl/conv3d_top.sv

parameters:
  # TODO: Define parameters

interfaces:
  s_axis_ifm:
    rtl_port: s_axis_ifm
    protocol: axi4_stream
    tensor: TODO          # TODO: tensor name
    packing:
      element_width: TODO
      elements_per_beat: TODO

  m_axi_data:
    rtl_port: m_axi_data
    protocol: axi4
    data_width: 256       # detected from port width
    memory_region: TODO
    tensors: [TODO]
    packing:
      element_width: TODO
      elements_per_beat: TODO

  s_axilite_ctrl:
    rtl_port: s_axilite_ctrl
    protocol: axi4_lite
    registers:
      # TODO: Define registers
      - name: TODO
        offset: 0x00
```

### 15.3 파서 범위

초기 지원: SystemVerilog module 포트 선언 (`input`, `output`, `inout`).
파라미터 추출, 인터페이스 타입 추출은 향후 확장.
