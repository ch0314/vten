# AXI-Lite Controller Codegen Enhancement

> NPU_3D 실제 디자인 분석 기반으로, 현재 vten의 axilite_ctrl / wrapper 자동 생성이
> production RTL을 완전히 대체하기 위해 필요한 변경 사항을 정리한다.

---

## 1. 배경

NPU_3D는 6개 독립 IP 커널(act_quant, bias_loader, fmapIO, mac_atu, psum_buffer,
weight_loader)로 구성된 3D Convolution Accelerator이다. 각 커널마다 수동 작성된
AXI-Lite controller와 wrapper가 있다.

분석 결과, 아래 **2가지 변경**만 추가하면 vten 자동 생성으로 기존 수동 코드를
완전히 대체할 수 있다.

---

## 2. 변경 사항

### 2.1 Vitis reserved region을 고려한 기본 register offset

#### 문제

Vitis/XRT 커널은 AXI-Lite 주소 공간의 0x00~0x13을 시스템 레지스터로 예약한다:

| Offset | 용도 |
|--------|------|
| 0x00 | CTRL (ap_start / ap_done / ap_idle / ap_ready) |
| 0x04 | GIE (Global Interrupt Enable) |
| 0x08 | IER (IP Interrupt Enable) |
| 0x0C | ISR (IP Interrupt Status) |
| 0x10 | 예약 |

NPU_3D의 6개 커널 모두 사용자 레지스터를 **0x14 (byte address 20)** 부터 배치한다.

현재 vten의 `axilite_ctrl.sv.j2`는 사용자가 `kernel_spec.yaml`에 명시한 offset을
그대로 사용하므로, 사용자가 0x14부터 offset을 지정하면 동작은 한다. 그러나:

- 새 커널 작성 시 offset을 0x00부터 시작하면 Vitis 예약 영역과 충돌
- XRT 빌드 파이프라인(package_ip.tcl, kernel.xml)에서도 이 규칙을 따라야 함

#### 변경

1. `RegisterSpec`에 명시적 offset이 없으면 **0x14부터 자동 할당**하는 옵션 추가
2. `InterfaceSpec`에 `user_register_base` 필드 추가 (기본값: `0x14`)

```python
# models.py - InterfaceSpec
@dataclass
class InterfaceSpec:
    ...
    user_register_base: int = 0x14  # Vitis reserved: 0x00-0x13
```

3. `kernel_spec.yaml`에서 offset을 생략할 수 있게 하고, parser가 자동 할당:

```yaml
# offset 명시: 그대로 사용
registers:
  - name: in_depth
    offset: 0x14
    fields: { value: "31:0" }

# offset 생략: user_register_base(0x14)부터 4-byte 단위로 자동 할당
registers:
  - name: in_depth       # → 0x14
    fields: { value: "31:0" }
  - name: in_height      # → 0x18
    fields: { value: "31:0" }
```

4. `package_ip.tcl.j2`와 `kernel.xml.j2`에서도 0x00~0x13은 Vitis 시스템 레지스터로
   자동 선언하고, 사용자 레지스터는 그 뒤에 배치

#### 영향 범위

| 파일 | 변경 내용 |
|------|----------|
| `vten/spec/models.py` | `InterfaceSpec.user_register_base` 필드 추가 |
| `vten/spec/parser.py` | offset 생략 시 자동 할당 로직 |
| `templates/axilite_ctrl.sv.j2` | 변경 없음 (offset은 이미 RegisterSpec에서 결정) |
| `templates/package_ip.tcl.j2` | Vitis reserved region 자동 선언 |
| `templates/kernel.xml.j2` | address range에 reserved region 반영 |

---

### 2.2 Hardware-sourced register (`source: hardware`)

#### 문제

NPU_3D fmapIO 커널에 다음 패턴이 존재한다:

```systemverilog
// fmapIO_ctrl.sv (line 121-127)
if (wready && wvalid) begin
    update_reg(awaddr_reg_idx, wdata, wmask, reg_map);
end else begin
    reg_map.vsync.data <= vsync_in;       // 외부 신호가 레지스터를 덮어씀
    reg_map.layer_done.data <= layer_done; // HW가 set하는 RO 레지스터
end
```

이것은 `pulse`와 다르다:
- **pulse**: write 후 다음 cycle에 자동으로 0 (host→HW 1-shot trigger)
- **hw_source**: write가 없는 cycle에 외부 신호 값이 레지스터에 반영됨 (HW→host status)

현재 vten은 `pulse`만 지원하고, HW 입력이 레지스터를 override하는 패턴은 미지원.

#### 변경

1. `RegisterSpec`에 `source` 필드 추가:

```python
# models.py - RegisterSpec
@dataclass
class RegisterSpec:
    name: str
    offset: int
    fields: dict[str, str] | None = None
    auto_bind: AutoBindSpec | None = None
    interface_name: str = ""
    access: str = "rw"
    pulse: bool = False
    reset_value: int = 0
    source: str = "software"  # NEW: "software" | "hardware"
```

| source 값 | 의미 | 포트 방향 | 동작 |
|-----------|------|----------|------|
| `"software"` (기본값) | host가 write, core가 read | controller → core (output) | 현재 동작과 동일 |
| `"hardware"` | core가 값을 공급, host가 read | core → controller (input) | write가 없는 cycle에 HW 입력으로 갱신 |

2. `kernel_spec.yaml` 예시:

```yaml
registers:
  - name: vsync
    offset: 0x50
    fields: { value: "0:0" }
    source: hardware    # write가 없을 때 core의 vsync_in으로 복원됨
  - name: layer_done
    offset: 0x54
    fields: { value: "0:0" }
    access: ro
    source: hardware    # core가 완료 시 set, host가 polling
```

3. `axilite_ctrl.sv.j2` 템플릿 변경:

**포트 선언** — `source: hardware`인 레지스터는 access와 무관하게 항상 input:

```jinja2
{# 현재 (line 32-38) #}
{% for reg in registers %}
{% if reg.access == 'ro' or reg.access == 'w1c' %}
    input  logic [{{ reg.width - 1 }}:0] reg_{{ reg.name }}{{ "," if not loop.last else "" }}
{% else %}
    output logic [{{ reg.width - 1 }}:0] reg_{{ reg.name }}{{ "," if not loop.last else "" }}
{% endif %}
{% endfor %}

{# 변경 후 #}
{% for reg in registers %}
{% if reg.source == 'hardware' %}
    input  logic [{{ reg.width - 1 }}:0] reg_{{ reg.name }}{{ "," if not loop.last else "" }}
{% elif reg.access == 'ro' or reg.access == 'w1c' %}
    input  logic [{{ reg.width - 1 }}:0] reg_{{ reg.name }}{{ "," if not loop.last else "" }}
{% else %}
    output logic [{{ reg.width - 1 }}:0] reg_{{ reg.name }}{{ "," if not loop.last else "" }}
{% endif %}
{% endfor %}
```

**Write path** — `source: hardware` + `access: rw`인 레지스터는 write 가능하되,
write가 없는 cycle에 HW 값으로 복원:

```jinja2
{# Write decode 블록 뒤에 추가 (line 92 부근) #}

{% for reg in registers if reg.source == 'hardware' and reg.access != 'ro' %}
            // HW-sourced: restore from core when not being written
            if (!(aw_done && w_done && !s_bvalid &&
                  aw_addr_latch == ADDR_W'('h{{ '%04X' | format(reg.offset) }})))
                reg_{{ reg.name }}_latch <= reg_{{ reg.name }};
{% endfor %}
```

**Read path** — `source: hardware`인 레지스터는 core 입력을 직접 읽음:

```jinja2
{# Read decode case문 안 (line 112-120) #}
{% for reg in registers %}
{% if reg.source == 'hardware' %}
                    ADDR_W'('h{{ '%04X' | format(reg.offset) }}): s_rdata <= {% if reg.width < data_width %}{{ "{" }}{{ data_width - reg.width }}'d0, reg_{{ reg.name }}{{ "}" }}{% else %}reg_{{ reg.name }}{% endif %};
{% elif reg.pulse %}
                    ...
{% endif %}
{% endfor %}
```

> **구현 노트**: `source: hardware` + `access: ro`는 단순히 input 포트를 read path에
> 연결하면 된다 (내부 latch 불필요). `source: hardware` + `access: rw`는 write 시
> 내부 latch에 저장하되, 다음 cycle에 HW 값으로 복원하는 로직이 필요하다.
> NPU_3D fmapIO의 vsync가 이 패턴이다.
>
> 다만 `source: hardware` + `access: rw` 조합은 실제로는 fmapIO의 vsync 한 곳에서만
> 사용되며, 대부분의 경우 `source: hardware`는 `access: ro`와 함께 쓰인다.
> 따라서 우선순위는:
> 1. `source: hardware` + `access: ro` — 반드시 지원 (layer_done 패턴)
> 2. `source: hardware` + `access: rw` — 선택적 지원 (vsync 패턴, pulse로 대체 가능)

4. `wrapper.sv.j2` 템플릿 변경:

core 인스턴스화 시 `source: hardware` 레지스터의 포트 방향이 반대임을 반영:

```jinja2
{# Core port 연결 (line 246-248) — 현재는 방향 구분 없이 연결 #}
{# source: hardware인 경우 core의 output → controller의 input #}
{# source: software인 경우 controller의 output → core의 input #}
{# wire 이름이 동일하므로 연결 코드 자체는 변경 불필요 #}
{# 단, core의 포트 방향 문서화에 반영 필요 #}
```

> wrapper의 wire 연결은 방향을 명시하지 않으므로 코드 변경은 없다.
> 단, core 작성 가이드에서 `source: hardware` 레지스터는 core의 **output**으로
> 선언해야 함을 명시해야 한다.

#### 영향 범위

| 파일 | 변경 내용 |
|------|----------|
| `vten/spec/models.py` | `RegisterSpec.source` 필드 추가 |
| `vten/spec/parser.py` | `source` 파싱, 유효성 검증 (`source: hardware` + `pulse: true` 금지 등) |
| `templates/axilite_ctrl.sv.j2` | 포트 방향 분기, read path에 HW input 반영 |
| `templates/wrapper.sv.j2` | 변경 없음 (wire 연결은 방향 무관) |
| `specs/03_kernel_spec_schema.md` | `source` 필드 문서화 |

---

## 3. 변경 불필요 확인 사항

분석 과정에서 검토했으나 변경이 **불필요**한 항목:

### 3.1 SV interface ↔ flat 포트 변환

현재 `wrapper.sv.j2`가 이미 flat ↔ SV interface 변환을 생성한다 (line 117-202).
사용자 core가 `vten_axis_if`, `vten_aximm_if` 등 SV interface를 사용하면,
wrapper가 외부 flat 포트와 내부 SV interface를 자동 연결한다.

NPU_3D의 수동 wrapper (`fmapIO_wrapper.sv`)가 하는 것과 동일:
```systemverilog
// 수동 (NPU_3D)
axilite #(...) __fmapIO_axilite ();
`AXILITE_ASSIGN_SLAVE_TO_FLAT(fmapIO_axilite, __fmapIO_axilite)

// 자동 (vten wrapper.sv.j2, line 130-143)
vten_axis_if #(.DATA_W(256)) ifm_loader_out_if();
assign m_axis_ifm_loader_out_tdata  = ifm_loader_out_if.tdata;
assign m_axis_ifm_loader_out_tvalid = ifm_loader_out_if.tvalid;
assign ifm_loader_out_if.tready     = m_axis_ifm_loader_out_tready;
```

**결론: 변경 불필요.**

### 3.2 64-bit 주소 조합 (LSB/MSB → 64-bit)

NPU_3D의 controller가 `{MSB, LSB}` 조합을 출력 포트로 제공하지만,
이는 core 내부에서 한 줄로 처리 가능:

```systemverilog
// core 안에서
assign ifm_start_addr = {reg_ifm_start_addr_msb, reg_ifm_start_addr_lsb};
```

`kernel_spec.yaml`에서 LSB/MSB를 별도 레지스터로 선언하면 충분:

```yaml
registers:
  - name: ifm_start_addr_lsb
    offset: 0x38
    fields: { value: "31:0" }
    auto_bind: { tensor: ifm, value: address, bits: "31:0" }
  - name: ifm_start_addr_msb
    offset: 0x3C
    fields: { value: "31:0" }
    auto_bind: { tensor: ifm, value: address, bits: "63:32" }
```

**결론: 변경 불필요.**

### 3.3 AXI-MM DMA 상태 머신

NPU_3D `weight_loader_aximm_ctrl.sv`는 5-state DMA 엔진(IDLE → AXI_INIT_READ →
AXI_READ_WAIT → BANK_DONE_WAIT → DONE)을 구현한다. 이것은 AXI-Lite 레지스터
controller가 아니라 **데이터 플레인 로직**이다.

이런 커스텀 상태 머신은 자동 생성 대상이 아니며, 사용자가 core 안에 구현한다.
`kernel_spec.yaml`에서 해당 AXI-MM 인터페이스는 `generate_controller: false`로
선언하면 된다.

**결론: 변경 불필요. 현재 설계가 올바르다.**

### 3.4 wstrb (Write Strobe) 처리

NPU_3D의 controller는 wstrb을 wmask로 변환하여 byte-level write를 지원한다:
```systemverilog
assign wmask = {{8{wstrb[3]}}, {8{wstrb[2]}}, {8{wstrb[1]}}, {8{wstrb[0]}}};
reg_map[idx] <= (wdata & wmask) | (reg_map[idx] & ~wmask);
```

현재 vten의 `axilite_ctrl.sv.j2`는 wstrb을 무시하고 전체 write한다.
이는 대부분의 경우 문제없지만(host는 보통 32-bit 전체를 write),
엄밀한 AXI spec 준수를 위해 향후 개선 가능. **현 단계에서는 불필요.**

---

## 4. NPU_3D 커널별 대체 가능성 요약

위 2가지 변경 적용 후:

| 커널 | 레지스터 수 | 특수 패턴 | 자동 생성 대체 |
|------|-----------|----------|--------------|
| mac_atu | 8 (RW) | 없음 | 즉시 가능 |
| psum_buffer | 9 (RW + pulse) | vsync pulse | 즉시 가능 |
| act_quant | 9 (RW + pulse) | vsync pulse | 즉시 가능 |
| bias_loader | 4 (RW + pulse) | 64-bit addr, vsync pulse | 즉시 가능 |
| weight_loader axilite | 75 (RW + pulse) | 32개 base addr, vsync pulse | 즉시 가능 |
| weight_loader aximm | N/A | DMA 상태 머신 | core 로직 (자동 생성 대상 아님) |
| fmapIO | 16 (RW + RO) | layer_done (HW RO), vsync (HW input) | `source: hardware` 필요 |

---

## 5. 구현 우선순위

1. **`source: hardware` + `access: ro`** — fmapIO의 `layer_done` 패턴. input 포트를
   read path에 연결하면 됨. 가장 단순하고 가장 필요.
2. **register offset 자동 할당 (0x14 base)** — 새 커널 작성 편의성 + XRT 호환성.
3. **`source: hardware` + `access: rw`** (선택) — fmapIO의 `vsync` 패턴. pulse로
   대체 가능하므로 우선순위 낮음.

---

## 6. 참고: NPU_3D 기존 코드 위치

| 커널 | Controller | Wrapper | Register Map |
|------|-----------|---------|-------------|
| act_quant | `design/activation/rtl/act_quant_ctrl.sv` | `act_quant_wrapper.sv` | `act_quant_reg.sv` |
| bias_loader | `design/bias_loader/rtl/bias_loader_ctrl.sv` | `bias_loader_wrapper.sv` | `bias_loader_reg.sv` |
| fmapIO | `design/fmapIO/rtl/fmapIO_ctrl.sv` | `fmapIO_wrapper.sv` | `fmapIO_reg.sv` |
| mac_atu | `design/mac/rtl/mac_atu_axilite_ctrl.sv` | `mac_atu_top_wrapper.sv` | (inline enum) |
| psum_buffer | `design/psum_buffer/rtl/psum_buffer_intf_port_axilite_ctrl.sv` | `psum_buffer_intf_port.sv` | `psum_pkg.sv` |
| weight_loader | `design/weight_loader/rtl/weight_loader_axilite_ctrl.sv` | `weight_loader_intf_port.sv` | `wgt_pkg.sv` |
