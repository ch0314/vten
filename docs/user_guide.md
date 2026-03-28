# vTen User Guide

**Version 0.5.0 — March 2026**

vTen은 DSA(Domain-Specific Accelerator) 검증을 위한 텐서 중심 프레임워크이다.
Python으로 검증 시나리오를 정의하고, 8-stage 컴파일 파이프라인으로 SHM 이미지를 생성한 후,
BFM이 DUT를 구동하여 RTL 수준 검증을 자동화한다.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [DUT 작성 가이드](#3-dut-작성-가이드)
4. [kernel_spec.yaml 작성](#4-kernel_specyaml-작성)
5. [Kernel 클래스 작성](#5-kernel-클래스-작성)
6. [TestScenario 작성](#6-testscenario-작성)
7. [CLI 워크플로우](#7-cli-워크플로우)
8. [Functional API](#8-functional-api)
9. [CompositeKernel (멀티 IP)](#9-compositekernel-멀티-ip)
10. [고급 기능](#10-고급-기능)
11. [트러블슈팅](#11-트러블슈팅)

---

## 1. Quick Start

```bash
# 프로젝트 초기화
cd my_project
vten init --kernel my_accel --backend xsim

# 빌드 (DPI-C, codegen, SV compile)
vten build --kernel my_accel

# 테스트 실행
vten run --kernel my_accel --test TestMyAccel

# 결과 확인
vten report
```

---

## 2. 프로젝트 구조

```
my_project/
├── vten.toml                # 프로젝트 설정
├── rtl/                     # 공유 RTL 소스
│   └── my_accel.sv
├── kernels/
│   └── my_accel/
│       ├── kernel_spec.yaml # 인터페이스 사양
│       ├── my_accel_kernel.py  # Kernel 클래스
│       ├── tests/
│       │   └── test_my_accel.py  # TestScenario
│       └── build/           # 빌드 산출물 (자동 생성)
└── results/                 # 테스트 결과 (자동 생성)
```

### vten.toml 기본 설정

```toml
[project]
name = "my_project"
version = "0.1.0"

[parameters]
N = 1024                     # 전역 파라미터 (커널에서 ${N}으로 참조)

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
timeout_ms = 10000

[rtl]
sources = ["rtl/**/*.sv"]
```

---

## 3. DUT 작성 가이드

### 3.1 기본 규칙

vTen은 DUT의 외부 인터페이스를 BFM으로 구동한다. DUT 자체는 표준 RTL로 작성하되:

- **클럭/리셋**: `clk` (posedge), `rst_n` (active-low) 권장
- **AXI4-Stream**: `tdata`, `tvalid`, `tready`, `tlast` 4-wire 프로토콜
- **AXI4**: 5채널 (AW/W/B/AR/R), 표준 AXI4 프로토콜
- **AXI4-Lite**: 5채널, 32-bit data, 제어 레지스터용

### 3.2 최소 Streaming DUT 예시

```systemverilog
// rtl/passthrough.sv — 가장 단순한 형태
module passthrough #(parameter DATA_W = 256)(
    input  logic             clk,
    input  logic             rst_n,
    // AXI4-Stream slave (input)
    input  logic [DATA_W-1:0] s_axis_tdata,
    input  logic              s_axis_tvalid,
    output logic              s_axis_tready,
    input  logic              s_axis_tlast,
    // AXI4-Stream master (output)
    output logic [DATA_W-1:0] m_axis_tdata,
    output logic              m_axis_tvalid,
    input  logic              m_axis_tready,
    output logic              m_axis_tlast
);
    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready = m_axis_tready;
    assign m_axis_tlast  = s_axis_tlast;
endmodule
```

### 3.3 AXI-Lite + AXI4 DUT (generate_controller 활용)

`generate_controller: true` 설정 시 AXI-Lite 보일러플레이트(Write FSM, Read FSM)가 자동 생성된다.
사용자는 **core 모듈**만 작성하면 된다:

```systemverilog
// rtl/vector_alu_core.sv — register는 flat wire로 수신
module vector_alu_core #(parameter DATA_W = 256, ADDR_W = 64)(
    input  logic clk, rst_n,
    // Registers (axilite_ctrl이 생성)
    input  logic [31:0] reg_src_a_addr_lo, reg_src_a_addr_hi,
    input  logic [31:0] reg_src_b_addr_lo, reg_src_b_addr_hi,
    input  logic [31:0] reg_dst_addr_lo, reg_dst_addr_hi,
    input  logic [31:0] reg_length,
    input  logic [31:0] reg_op_mode,
    input  logic        reg_start,         // pulse: 1 cycle high
    output logic        reg_done,          // ro: core → host
    // AXI4 Master (SV interface)
    vten_aximm_if.master m_axi
);
    // ... DMA + ALU 로직 ...
endmodule
```

빌드 시 자동 생성되는 파일:
- `vector_alu_axilite_ctrl.sv` — AW/W/B/AR/R FSM
- `vector_alu_wrapper.sv` — ctrl + core 연결, 외부 flat 포트

### 3.4 SV Interface 사용 (선택)

`vten_sv/`에서 제공하는 SV interface를 core에서 사용하면 포트가 깔끔해진다:

```systemverilog
module my_core (
    input  logic clk, rst_n,
    vten_axis_if.slave  s_axis,    // AXI4-Stream
    vten_aximm_if.master m_axi     // AXI4 Memory-Mapped
);
```

---

## 4. kernel_spec.yaml 작성

`kernel_spec.yaml`은 DUT의 인터페이스를 선언적으로 기술한다. RTL을 수정하지 않으며, 텐서↔포트 매핑을 이 파일에 집중한다.

### 4.1 최소 예시 (Streaming)

```yaml
kernel: passthrough
rtl_top: rtl/passthrough.sv

interfaces:
  input_stream:
    rtl_port: s_axis
    protocol: axi4_stream
    tensor: data_in
    packing:
      element_width: 8
      elements_per_beat: 32
      bit_order: lsb_first

  output_stream:
    rtl_port: m_axis
    protocol: axi4_stream
    tensor: data_out
    packing:
      element_width: 8
      elements_per_beat: 32
      bit_order: lsb_first
```

### 4.2 AXI4-Lite + AXI4 예시

```yaml
kernel: vector_alu
rtl_top: rtl/vector_alu_core.sv

memory_regions:
  ddr:
    base: 0x10000000
    size: 0x10000000
    alignment: 4096

interfaces:
  ctrl:
    protocol: axi4_lite
    generate_controller: true    # AXI-Lite FSM 자동 생성
    registers:
      - name: src_a_addr_lo
        offset: 0x00
        auto_bind: { tensor: operand_a, value: address, bits: "31:0" }
      - name: src_a_addr_hi
        offset: 0x04
        auto_bind: { tensor: operand_a, value: address, bits: "63:32" }
      - name: length
        offset: 0x18
        auto_bind: { tensor: operand_a, value: size_beats }
      - name: op_mode
        offset: 0x1C
      - name: ctrl
        offset: 0x20
        pulse: true
        fields: { start: "0:0" }
      - name: status
        offset: 0x24
        access: ro
        fields: { done: "0:0" }

  mem_port:
    protocol: axi4
    data_width: 256
    addr_width: 64
    memory_region: ddr
    tensors: [operand_a, operand_b, result]
    packing:
      element_width: 8
      elements_per_beat: 32
```

### 4.3 Interface Array 예시

```yaml
interfaces:
  din:
    protocol: axi4_stream
    role: slave
    data_width: 256
    array:
      dimensions: [4]           # 4개 포트: s_axis_din_0 ~ s_axis_din_3
    tensor: data_in
    packing:
      element_width: 8
      elements_per_beat: 32
```

### 4.4 rtl_port 기본값

`rtl_port`를 생략하면 protocol + role + name으로 자동 생성:

| Protocol | Role | 접두사 | 예시 (name=wgt) |
|----------|------|--------|-----------------|
| axi4_stream | slave | `s_axis_` | `s_axis_wgt` |
| axi4_stream | master | `m_axis_` | `m_axis_wgt` |
| axi4 | master | `m_axi_` | `m_axi_wgt` |
| axi4_lite | slave | `s_axilite_` | `s_axilite_wgt` |

### 4.5 Register Offset 자동 할당

offset을 생략하면 `user_register_base` (기본 0x14)부터 4-byte 단위로 자동 할당:

```yaml
registers:
  - name: in_depth       # → 0x14
  - name: in_height      # → 0x18
  - name: in_width       # → 0x1C
```

### 4.6 auto_bind 종류

| 값 | 설명 | 예시 |
|----|------|------|
| `address` | 텐서의 물리 주소 (bits로 LSB/MSB 분리) | `{ tensor: ifm, value: address, bits: "31:0" }` |
| `size_beats` | 직렬화 크기 / beat 크기 | `{ tensor: ifm, value: size_beats }` |
| `size_bytes` | 직렬화된 바이트 크기 | `{ tensor: ifm, value: size_bytes }` |
| `size_elements` | 텐서 요소 개수 | `{ tensor: ifm, value: size_elements }` |
| `param` | 파라미터 참조 | `{ param: "${C}" }` |
| `expr` | 산술 표현식 | `{ expr: "${N}*${K}" }` |

---

## 5. Kernel 클래스 작성

Python `Kernel` 서브클래스로 텐서 선언, 입력 생성, golden reference를 정의한다.

### 5.1 기본 구조

```python
import torch
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor

class PassthroughKernel(Kernel):
    spec = "kernels/passthrough/kernel_spec.yaml"

    data_in = Tensor(shape=("${N}",), dtype=torch.int8, interface="input_stream")
    data_out = Tensor(shape=("${N}",), dtype=torch.int8, interface="output_stream")

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self):
        """Golden reference: 기대 출력을 torch로 계산."""
        return self.data_in.data.clone()
```

### 5.2 주요 요소

| 요소 | 설명 |
|------|------|
| `spec` | kernel_spec.yaml 경로 (PROJECT_ROOT 기준) |
| `Tensor(shape, dtype, interface)` | 텐서 디스크립터. `shape`에 `"${N}"`처럼 파라미터 사용 가능 |
| `generate_inputs()` | 랜덤 입력 생성 (seed로 재현 가능) |
| `forward()` | Golden reference 계산 (torch 기반) |

### 5.3 지원 dtype

`torch.int8`, `torch.int16`, `torch.int32`, `torch.float32` 등.

---

## 6. TestScenario 작성

`TestScenario`는 검증 시나리오를 정의하는 클래스이다. `vten run`으로 실행된다.

### 6.1 기본 Streaming 테스트

```python
from vten.cli.run import TestScenario

class TestPassthrough(TestScenario):
    kernel = "passthrough"

    def run(self, ctx, cfg):
        from passthrough_kernel import PassthroughKernel

        k = ctx.instantiate(PassthroughKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)       # Host → SHM
        h_push = ctx.push_tensor(k.data_in, dep=h_load)  # SHM → BFM → DUT
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)  # DUT → BFM → SHM
        ctx.verify(h_pull, k.forward())            # Golden 비교
```

### 6.2 AXI4 + AXI-Lite (Register Control) 테스트

```python
class TestVectorAluAdd(TestScenario):
    kernel = "vector_alu"

    def run(self, ctx, cfg):
        from vector_alu_kernel import VectorAluKernel

        k = ctx.instantiate(VectorAluKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        # 1. Load tensors
        h_a = ctx.load_tensor(k.operand_a)
        h_b = ctx.load_tensor(k.operand_b)

        # 2. Configure (auto_bind: 주소 + 크기 자동 바인딩)
        h_cfg = ctx.configure(k, dep=[h_a, h_b])

        # 3. Manual register: op_mode
        h_op = ctx.write_register(k.ctrl, {"op_mode": 0}, dep=h_cfg)

        # 4. Push data via AXI4 BFM
        h_push_a = ctx.push_tensor(k.operand_a, dep=h_cfg)
        h_push_b = ctx.push_tensor(k.operand_b, dep=h_cfg)

        # 5. Start (pulse register)
        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_op)

        # 6. Poll done
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)

        # 7. Pull result
        h_pull = ctx.pull_tensor(k.result, dep=h_cfg)
        h_pull.add_commit_dependency(h_poll)

        # 8. Verify
        ctx.verify(h_pull, k.forward(op="add"))
```

### 6.3 DSL 연산 요약

| 연산 | 설명 | BFM 동작 |
|------|------|----------|
| `ctx.load_tensor(t)` | Host → SHM 버퍼 | LOAD 명령 |
| `ctx.push_tensor(t)` | SHM → DUT (BFM 구동) | PUSH 명령 |
| `ctx.pull_tensor(t)` | DUT → SHM (BFM 수집) | PULL 명령 |
| `ctx.store_tensor(t)` | SHM → Host (결과 읽기) | STORE 명령 |
| `ctx.configure(k)` | auto_bind 레지스터 일괄 설정 | WRITE_REG × N |
| `ctx.write_register(reg, fields)` | 수동 레지스터 쓰기 | WRITE_REG |
| `ctx.read_register(reg, field)` | 레지스터 읽기 | READ_REG |
| `ctx.poll_register(reg, field)` | 값이 될 때까지 폴링 | POLL_REG |
| `ctx.send_tensor(t)` | LOAD + PUSH 단축 | LOAD + PUSH |
| `ctx.recv_tensor(t)` | PULL + STORE 단축 | PULL + STORE |
| `ctx.verify(handle, golden)` | Golden 비교 등록 | Host-side |

### 6.4 의존성 제어

- `dep=handle` 또는 `dep=[h1, h2]`: 선행 명령 완료 후 실행 (issue dependency)
- `handle.add_commit_dependency(other)`: 선행 명령 커밋(=데이터 확정) 후 실행

### 6.5 파라미터 Sweep

`configs` 리스트로 여러 설정을 한 번에 테스트:

```python
class TestScaleAdd(TestScenario):
    kernel = "scale_add"
    configs = [
        {"name": "default"},
        {"name": "identity", "scale_factor": 1, "offset_value": 0},
        {"name": "big_scale", "scale_factor": 5, "offset_value": 3},
        {"name": "small_n", "N": 32},
    ]
    def run(self, ctx, cfg):
        # cfg에서 파라미터 꺼내서 사용
        N = cfg.get("N", 1024)
        ...
```

### 6.6 Probe 검증 (Early Mismatch Detection)

BFM이 시뮬레이터 내부에서 매 beat마다 golden 데이터와 비교해 첫 불일치 즉시 abort한다. 타임아웃까지 기다릴 필요 없이 조기에 실패를 감지할 수 있다.

**Output Probe** — `pull_tensor`에 `probe=True` 추가:

```python
h_pull = ctx.pull_tensor(k.data_out, dep=h_load, probe=True)
ctx.verify(h_pull, k.forward())  # golden도 SHM에 전달되어 BFM이 비교
```

첫 beat mismatch 발생 시 에러 코드 8로 조기 abort된다.

**Internal Probe** — CompositeKernel 내부 wire에 probe 삽입:

```python
class ScaleAddIProbe(CompositeKernel):
    connections = [
        Connect(scale.data_out, offset.data_in, Internal(probe=True)),
    ]
```

TestScenario에서 internal probe golden 데이터 설정:

```python
scale_golden = k.forward_scale_only(scale_factor=2)
ctx.set_internal_probe_golden("scale", "data_out", scale_golden)
```

**Probe Mismatch 출력 예:**

```
[PROBE MISMATCH] cmd=0 cycle=46 beat=0 expected=0x00000000_00000000 actual=0x042ECEC6_80B866CC
```

결과 디렉토리에 `mismatches.json`이 생성되며 beat/cycle/expected/actual 상세 정보를 포함한다.

---

### 6.7 GUI 디버깅 (Waveform Inspection)

```bash
# GUI 모드: xsim GUI에서 interactive 디버깅
vten run --kernel my_accel --test TestMyAccel --gui

# Waveform 덤프 (배치 모드)
vten run --kernel my_accel --test TestMyAccel --waveform

# 실패 시에만 waveform 저장
vten run --kernel my_accel --test TestMyAccel --waveform-on-fail
```

**GUI + Probe 워크플로우:**

1. `vten run --kernel my_accel --test TestMyAccel --gui` 실행
2. xsim GUI가 열리면 Tcl 콘솔에서 `run all` 입력
3. Probe mismatch 발생 시 `$stop`으로 시뮬레이션 일시정지
4. waveform에서 mismatch 시점의 DUT 내부 신호 확인
5. `restart` → `run all`로 재실행 가능 (Python이 restart를 자동 감지)
6. xsim 닫으면 Python이 결과 리포트 출력

**결과 디렉토리 구성:**

```
results/<kernel>/<test>/
├── summary.json          # probe_mismatch 섹션 포함
├── stats.json
├── mismatches.json       # probe mismatch 상세 (있을 경우)
└── waveform.wdb          # 파형 (--waveform 사용 시)
```

---

## 7. CLI 워크플로우

### 7.1 전체 흐름

```bash
# 1. 프로젝트 초기화
vten init --kernel my_accel --backend xsim

# 2. 빌드 (전 단계)
vten build --kernel my_accel

# 3. 단계별 빌드 (선택)
vten build --kernel my_accel --stage codegen      # codegen만
vten build --kernel my_accel --upto compile       # compile까지

# 4. 테스트 실행
vten run --kernel my_accel --test TestMyAccel
vten run --kernel my_accel --test TestMyAccel --waveform  # 파형 저장
vten run --kernel my_accel --test TestMyAccel --gui       # Vivado GUI

# 5. 결과 확인
vten report                        # 터미널 출력
vten report --format json          # JSON
vten report --format html          # HTML 리포트
```

### 7.2 빌드 스테이지 (xsim)

| Stage | 설명 |
|-------|------|
| `project_setup` | Vivado 프로젝트 생성 |
| `dpi_c` | DPI-C 공유 라이브러리 빌드 |
| `codegen` | Jinja2 → tb_top.sv, wrapper, controller |
| `compile_order` | Vivado compile order 추출 |
| `compile` | xvlog + xelab |

### 7.3 Config Override

```bash
vten run --kernel my_accel --test TestMyAccel --config N=2048
```

### 7.4 Backend 선택

```bash
vten build --kernel my_accel --backend verilator
vten run --kernel my_accel --test TestMyAccel --backend verilator
```

지원 백엔드: `xsim` (기본), `verilator`, `xrt` (FPGA 하드웨어)

### 7.5 디버깅 옵션

```bash
# 상세 로그 + 에러 시 full traceback
vten -v build --kernel my_accel

# 조용한 모드 (경고 이상만 출력)
vten -q run --kernel my_accel

# 디버그 로그를 파일에 저장
vten --log-file debug.log run --kernel my_accel

# 시뮬레이터 verbose 출력 ($display 등)
vten run --kernel my_accel -v

# 파형 저장 (항상 / 실패 시에만)
vten run --kernel my_accel --waveform
vten run --kernel my_accel --waveform-on-fail

# xsim GUI 모드 (대화형 파형 검사)
vten run --kernel my_accel --gui
```

### 7.6 Exit Code

| Code | 의미 | 예시 |
|------|------|------|
| 0 | 성공 | 빌드 완료, 테스트 실행 완료 |
| 1 | 사용자 에러 | vten.toml 없음, 빌드 실패, 잘못된 인자 |
| 2 | 내부 에러 | vTen 버그 (`-v`로 traceback 확인) |
| 130 | Ctrl-C | 사용자가 실행 중단 |

에러 메시지는 traceback 없이 깔끔하게 표시된다. 상세 정보가 필요하면 `-v` 사용:

```bash
$ vten build
ERROR cli       | vten.toml not found in .

$ vten -v build    # traceback 포함
```

---

## 8. Functional API

빌드 없이 Python에서 직접 커널을 실행하는 고수준 API.

### 8.1 run_kernel (One-shot)

```python
from vten import run_kernel
from my_kernel import MyKernel

outputs = run_kernel(
    MyKernel,
    {"data_in": input_tensor},
    backend=my_backend,
    params={"N": 1024},
    configure=True,      # auto_bind 레지스터 자동 설정
)
result = outputs["data_out"]
```

### 8.2 KernelExecutor (Reusable)

반복 호출 시 이전 출력을 입력으로 재사용하면 자동으로 alias(LOAD 스킵):

```python
from vten import KernelExecutor

npu = KernelExecutor(MyKernel, backend=b, params={"N": 1024})

# 첫 호출: 정상 LOAD + PUSH
y = npu(data_in=x)["data_out"]

# 두 번째 호출: y가 이전 출력이므로 LOAD 스킵 (alias)
z = npu(data_in=y)["data_out"]
```

---

## 9. CompositeKernel (멀티 IP)

### 9.1 구조

```python
from vten.kernel.composite import CompositeKernel, Connect, Internal
from vten.kernel.register import register

class ScaleAddKernel(CompositeKernel):
    # Sub-kernel 바인딩
    scale = ScaleKernel.bind(interface_map={
        "ctrl": ("scale_ctrl", "scale"),     # 독립 AXI-Lite
        "input_stream": "input_stream",      # 외부 노출
        "output_stream": Internal(),         # 내부 와이어
    })
    offset = OffsetKernel.bind(interface_map={
        "ctrl": ("offset_ctrl", "offset"),
        "input_stream": Internal(),
        "output_stream": "output_stream",
    })

    # Register 핸들
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")

    # 텐서 노출
    data_in = scale.data_in.expose("input_stream")
    data_out = offset.data_out.expose("output_stream")

    # RTL 내부 연결
    connections = [Connect(scale.data_out, offset.data_in)]
```

### 9.2 interface_map 규칙

| 매핑 형태 | 의미 |
|-----------|------|
| `"top_name"` | 외부로 노출 (BFM 연결) |
| `("top_name", "bank")` | 외부 노출 + register bank 배치 |
| `Internal()` | RTL 내부 와이어 (BFM 없음) |
| `Internal(probe=True)` | 내부 와이어 + BFM probe |

### 9.3 CompositeKernel 빌드

```bash
# 서브커널 먼저 빌드
vten build --kernel scale
vten build --kernel offset

# 컴포지트 빌드 (wrapper-of-wrappers 자동 생성)
vten build --kernel scale_add
```

---

## 10. 고급 기능

### 10.1 다양한 Packing

```yaml
# int16, 7개씩
packing:
  element_width: 16
  elements_per_beat: 7    # → 112-bit bus

# float32, 4개씩
packing:
  element_width: 32
  elements_per_beat: 4    # → 128-bit bus

# Custom field mapping
packing:
  mode: custom
  fields:
    - { name: data_a, bits: [0, 23] }
    - { name: data_b, bits: [24, 47] }
```

### 10.2 Multi-Backend

```toml
# vten.toml
[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"

[backend.verilator]
threads = 4

[backend.xrt]
xclbin_path = "build/xrt/my_kernel_hw_emu.xclbin"
target = "hw_emu"
platform = "xilinx_u280_gen3x16_xdma_1_202211_1"
```

### 10.3 XRT Hardware Emulation

```bash
# XRT 빌드 아티팩트 생성
vten build --kernel my_accel --backend xrt

# hw_emu 실행
vten run --kernel my_accel --test TestMyAccel --backend xrt
```

XRT 백엔드는 CommandInterpreter를 통해 IR Command를 XRT API로 직접 변환:
- `LOAD` → `BO.write(data)`
- `PUSH` → `BO.sync(TO_DEVICE)`
- `PULL` → `BO.sync(FROM_DEVICE)`
- `STORE` → `BO.read(size)`
- `WRITE_REG` → `ip.write_register(offset, value)`

### 10.4 Report 활용

```bash
# 터미널: 명령별 레이턴시, 활용률
vten report

# JSON: CI 파이프라인 통합용
vten report --format json

# HTML: 공유용 리포트
vten report --format html
```

결과 디렉토리 (`results/<kernel>/<test>/`):
- `summary.json`: 테스트 결과, 검증 통과/실패, max_diff
- `stats.json`: 명령별 성능 통계 (cycle 수, stall, 활용률)

---

## 11. 트러블슈팅

### 에러 메시지 읽기

vTen의 에러는 traceback 없이 깔끔한 메시지로 표시된다:

```
ERROR cli       | vten.toml not found in .                      ← 설정 에러
ERROR cli       | xsim not found: xsim                          ← 도구 누락
ERROR runner    | probe mismatch (config 1/1)                   ← 검증 실패
ERROR cli       | internal error: unexpected key 'foo'          ← 내부 에러
```

| 접두사 | 출처 | 설명 |
|--------|------|------|
| `cli` | CLI 진입점 | 설정 에러, 인자 에러 |
| `runner` | 테스트 실행기 | 검증 실패, probe mismatch |
| `backend` | 시뮬레이션 백엔드 | 타임아웃, BFM 에러 |
| `build.xsim` | 빌드 파이프라인 | xvlog/xelab 컴파일 에러 |

상세 traceback이 필요하면 `-v` 플래그 사용:

```bash
vten -v build --kernel my_accel
```

### 빌드 실패

```bash
# 개별 스테이지 실행으로 원인 파악
vten build --kernel my_accel --stage codegen   # codegen만
vten build --kernel my_accel --stage compile   # compile만
vten build --kernel my_accel --force           # 캐시 무시 재빌드
```

### 시뮬레이션 타임아웃

- `vten.toml`에서 `timeout_ms` 증가
- `--waveform` 옵션으로 파형 확인
- AXI4-Lite poll 타임아웃: `poll_timeout` 기본값 100000 사이클
- 타임아웃 시 stuck command 정보가 자동 출력됨 (cmd_id, stall cycles, 레지스터 상태)

### Probe Mismatch (ProbeMismatchError)

- `mismatches.json`에서 beat/cycle 정보 확인
- `--gui` 모드로 mismatch 시점의 waveform 확인
- `$stop` 후 xsim waveform 창에서 관련 신호 추적
- `restart`로 반복 디버깅 가능

### 시뮬레이터를 찾을 수 없음

```
ERROR cli       | xsim not found: /tools/Xilinx/Vivado/2023.2/bin/xsim
                  Check that Vivado is installed and vivado_path is set in vten.toml [backend.xsim]
```

- Vivado 설치 경로를 `vten.toml`의 `[backend.xsim].vivado_path`에 설정
- Verilator: `vten build`로 먼저 빌드 (`obj_dir/Vtb_top` 생성 필요)

### 테스트를 찾을 수 없음

```
ERROR cli       | Not found: no test scenario matching 'TestFoo'
```

- `kernels/<kernel>/tests/test_*.py` 파일에 `TestScenario` 서브클래스가 있는지 확인
- 테스트 파일에 문법 에러가 있으면 경고가 표시됨:
  ```
   WARN runner    | failed to load test_foo.py: invalid syntax (test_foo.py, line 12)
  ```

### 검증 실패 (VerificationError)

- `max_diff` 값 확인 — 부동소수점은 `torch.allclose`의 `atol`/`rtol` 조정 필요
- `probe=True`로 beat-level에서 첫 불일치 위치 확인
- Probe mismatch는 element 단위로 expected/actual 값을 표시:
  ```
  ERROR runner    | probe mismatch (config 1/1)
                     tensor: data_out (float32)
                     cmd_id: 2
                     first mismatch: beat 5 (elements [640..646])
                       [640]: expected=1.234567, actual=1.234560
                     total mismatches logged: 142
  ```
- `broken_passthrough` 예제 참고 (의도적 실패 패턴)

### 일반적인 실수

| 증상 | 원인 | 해결 |
|------|------|------|
| `vten.toml not found` | 잘못된 디렉토리에서 실행 | `--project <경로>` 지정 또는 프로젝트 디렉토리로 이동 |
| `ShapeMismatchError` | Tensor shape와 실제 데이터 크기 불일치 | `shape=("${N}",)` 확인 |
| `SpecValidationError` | kernel_spec.yaml 필드 오류 | packing bus_width ≤ data_width 확인 |
| BFM 데이터 불일치 | Packing bit_order 불일치 | RTL과 `lsb_first`/`msb_first` 맞춤 |
| Poll 타임아웃 | DUT done 신호 미발생 | RTL의 done 레지스터 로직 확인 |
| `ProbeMismatchError` | Probe가 golden과 RTL 출력 불일치 감지 | `--gui`로 mismatch 시점 waveform 확인 |
| `internal error` | vTen 내부 버그 | `-v`로 traceback 확인 후 보고 |
