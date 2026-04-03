# vTen User API & Workflow

**Version 0.6.0 — March 2026**

---

## Table of Contents

1. [API Level Overview](#1-api-level-overview)
2. [Level 0: Kernel Definition](#2-level-0-kernel-definition)
3. [Level 1: DSL (ExecutionContext)](#3-level-1-dsl-executioncontext)
4. [Level 2: Functional API](#4-level-2-functional-api)
5. [Workflow: 단일 커널](#5-workflow-단일-커널)
6. [Workflow: Multi-Layer Pipeline](#6-workflow-multi-layer-pipeline)
7. [CLI Workflow](#7-cli-workflow)
8. [API Quick Reference](#8-api-quick-reference)
9. [Config & Parameter System](#9-config--parameter-system)
10. [Migration Guide](#10-migration-guide)

---

## 1. API Level Overview

vTen은 추상화 수준이 다른 3개의 사용자 API 레벨을 제공한다.

```
Level 2  ┌──────────────────────────────────┐  가장 간결
Functional │  y = npu(ifm=x)["ofm"]          │  DSL 자동 생성
API       └────────────┬─────────────────────┘
                       │
Level 1   ┌────────────┴─────────────────────┐  세밀한 제어
DSL       │  h = ctx.push_tensor(k.ifm)       │  의존성/순서 직접 지정
          │  ctx.verify(h, golden)             │
          └────────────┬─────────────────────┘
                       │
Level 0   ┌────────────┴─────────────────────┐  항상 필요
Kernel    │  class MyKernel(Kernel):           │  텐서, 인터페이스,
Definition│      ifm = Tensor(...)             │  golden 모델 정의
          └──────────────────────────────────┘
```

| Level | 언제 사용 | 추상화 | 제어 수준 |
|-------|----------|--------|----------|
| **0 — Kernel** | 항상 | RTL 커널의 Python 모델 정의 | — |
| **1 — DSL** | 복잡한 의존성, probe, poll, barrier 필요 시 | 개별 op 단위 기록 | 높음 |
| **2 — Functional** | PyTorch 파이프라인 통합, 반복 호출 | send→configure→recv 자동 | 낮음 |

---

## 2. Level 0: Kernel Definition

모든 워크플로우의 시작점. RTL 모듈의 Python 모델을 정의한다.

### 2.1 kernel_spec.yaml

RTL 인터페이스를 선언적으로 기술한다. `PROJECT_ROOT/kernels/<name>/kernel_spec.yaml`에 위치.

```yaml
kernel: passthrough
rtl_top: rtl/passthrough.sv

parameters:
  N: "${N}"                          # 런타임에 해결되는 파라미터

interfaces:
  input_stream:
    rtl_port: s_axis                  # RTL 포트 접두사
    protocol: axi4_stream
    tensor: data_in                   # Python Tensor 이름과 매칭
    packing:
      element_width: 8
      elements_per_beat: 32

  output_stream:
    rtl_port: m_axis
    protocol: axi4_stream
    tensor: data_out
    packing:
      element_width: 8
      elements_per_beat: 32
```

### 2.2 Kernel 클래스

```python
from vten import Kernel, Tensor  # 통합 import

class PassthroughKernel(Kernel):
    spec = "kernels/passthrough/kernel_spec.yaml"

    # 텐서 선언: interface는 kernel_spec.yaml의 interface 이름과 매칭
    data_in  = Tensor(shape=("${N}",), dtype=torch.int8, interface="input_stream")
    data_out = Tensor(shape=("${N}",), dtype=torch.int8, interface="output_stream")

    def generate_inputs(self, seed=None):
        """입력 텐서 데이터 생성."""
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, data_in) -> dict[str, torch.Tensor]:
        """Golden reference 계산. 반환값은 {output_tensor_name: data} dict."""
        return {"data_out": data_in.clone()}
```

### 2.3 CompositeKernel

여러 서브 커널을 하나의 RTL top에 **공간적(spatial)**으로 합성한다.

```python
from vten import CompositeKernel

class NPUTopKernel(CompositeKernel):
    spec = "kernels/npu_top/kernel_spec.yaml"

    # 서브 커널 인스턴스 선언
    fmapio = FmapIOKernel()
    wgt    = WeightLoaderKernel()
    mac    = MACKernel()

    # 내부 RTL 연결 (>> 연산자로 선언)
    connections = [
        fmapio.ifm_out >> mac.ifm_in,
        wgt.wgt_out    >> mac.wgt_in,
        mac.psum_out   >> fmapio.ofm_in,
    ]
```

---

## 3. Level 1: DSL (ExecutionContext)

개별 DSL 연산을 직접 기록한다. 의존성, 순서, probe, poll 등 세밀한 제어가 필요할 때 사용.

### 3.1 기본 흐름

```python
from vten.runtime.context import ExecutionContext

ctx = ExecutionContext(backend=backend, project_params={"N": 1024})
k = ctx.instantiate(PassthroughKernel, N=1024)
k.generate_inputs(seed=42)

# L1: Host ↔ SHM
h_load = ctx.load_tensor(k.data_in)

# L2: SHM ↔ DUT
h_push = ctx.push_tensor(k.data_in, dep=h_load)
h_pull = ctx.pull_tensor(k.data_out, dep=h_push)

# 검증
ctx.verify(h_pull, k.forward()["data_out"])

result = ctx.run()   # → BatchResult
print(result.status)  # "DONE"
```

### 3.2 DSL Operations

**L1: Host ↔ SHM Memory**

| 메서드 | 설명 | IR 명령 |
|--------|------|---------|
| `ctx.load_tensor(t)` | Host → SHM (H2D 텐서 직렬화/적재) | LOAD |
| `ctx.store_tensor(t)` | SHM → Host (D2H 텐서 읽기/역직렬화) | STORE |

**L2: SHM ↔ Accelerator (BFM)**

| 메서드 | 설명 | IR 명령 |
|--------|------|---------|
| `ctx.push_tensor(t)` | SHM → DUT (BFM이 DUT에 데이터 전송) | PUSH |
| `ctx.pull_tensor(t)` | DUT → SHM (BFM이 DUT에서 데이터 수신) | PULL |

**L3: Control**

| 메서드 | 설명 | IR 명령 |
|--------|------|---------|
| `ctx.write_register(reg, fields)` | 레지스터 쓰기 | WRITE_REG |
| `ctx.read_register(reg, field)` | 레지스터 읽기 | READ_REG |
| `ctx.poll_register(reg, field)` | 레지스터 폴링 (조건 대기) | POLL_REG |
| `ctx.configure(kernel)` | auto_bind 레지스터 일괄 쓰기 | N × WRITE_REG |
| `ctx.barrier()` | 전체 동기화 펜스 | BARRIER |

**Shorthands** (L1+L2 결합)

| 메서드 | 확장 | 용도 |
|--------|------|------|
| `ctx.send_tensor(t)` | load + push | H2D 텐서 전체 전송 |
| `ctx.recv_tensor(t)` | pull + store | D2H 텐서 전체 수신 |
| `ctx.recv_tensor(t, chunks=N)` | N × (pull + store) | D2H 텐서를 N 청크로 분할 수신 |

**검증 & 버퍼 재사용**

| 메서드 | 설명 |
|--------|------|
| `ctx.verify(handle, golden)` | HW 출력 vs golden 비교 |
| `ctx.alias(src, dst)` | 버퍼 재사용 (LOAD/STORE 스킵) |
| `handle.add_commit_dependency(other)` | commit 의존성 추가 |
| `ctx.set_internal_probe_golden(sub, tensor, golden)` | 내부 probe golden 등록 |

### 3.3 의존성 그래프

의존성은 `dep=` 파라미터로 지정한다.

```python
l1 = ctx.load_tensor(k.ifm)         # dep 없음 → 즉시 실행 가능
l2 = ctx.load_tensor(k.weight)      # dep 없음 → l1과 병렬

cfg = ctx.configure(k, dep=[l1, l2]) # l1, l2 완료 후 configure
push = ctx.push_tensor(k.ifm, dep=cfg)
pull = ctx.pull_tensor(k.ofm, dep=push)

poll = ctx.poll_register(k.ctrl, "done", dep=cfg)
pull.add_commit_dependency(poll)      # poll 완료 전까지 pull commit 보류

store = ctx.store_tensor(k.ofm, dep=pull)
ctx.verify(store, k.forward()["ofm"])
```

### 3.4 Chunked Receive (`chunks=`)

DUT가 depth slice 사이에 tready rising edge를 요구하는 경우 등, 전송을 시간적으로 분할해야 할 때 사용한다. 선언적 Tensor 하나로 per-chunk 분할 수신이 가능하다.

```python
# chunks=N (int): N등분
handles = ctx.recv_tensor(k.partial_sum, chunks=in_depth, dep=h_cfg)
# → list[OperationHandle], 각 chunk별 handle

# chunks=[n0, n1, ...] (list[int]): element count 직접 지정
handles = ctx.recv_tensor(k.output, chunks=[100, 200, 100], dep=h_cfg)
```

**동작 원리:**
- 직렬화된 바이트 스트림을 N등분하여 분할 수신 (C-contiguous = **axis 0 기준**)
- 각 chunk는 별도의 PULL 커맨드 그룹으로 lowering됨
- BFM이 커맨드 사이에 tready를 deassert/reassert → edge-triggered DUT에 필요한 rising edge 제공
- Array interface와 결합 가능: 각 chunk가 array element별로 분할됨

> **제약:** 현재 axis 0 기준 분할만 지원. 임의 축 분할(`axis=` 파라미터)은 추후 지원 예정.

**Per-chunk 검증:**
```python
handles = ctx.recv_tensor(k.psum, chunks=4, dep=h_cfg)
for d in range(4):
    ctx.verify(handles[d], golden_per_depth[d])
```

**반환 타입:**
- `chunks=None` (기본값): `OperationHandle` — 기존 동작과 동일
- `chunks=N` 또는 `chunks=[...]`: `list[OperationHandle]`

### 3.5 BatchResult

```python
result = ctx.run()

result.status            # "DONE" | "ERROR"
result.total_cycles      # 총 사이클 수
result.per_command_stats  # 명령별 통계 (CmdStats 리스트)
result.output_tensors    # {"tensor_name": torch.Tensor} — D2H 텐서 자동 역직렬화
result.error             # 에러 정보 (있을 경우)
```

### 3.6 Probe 검증

Probe는 DUT 데이터를 golden과 실시간 비교하는 검증 메커니즘이다.
출력 텐서(output probe)와 내부 배선(internal probe) 모두 지원한다.

#### 선언적 Probe API (권장)

TestScenario에 `probes` 리스트를 선언하면, 커널 정의 변경 없이 probe가 자동 적용된다.
golden 데이터는 `forward()`의 golden chain에서 자동 추출된다.

```python
class TestScaleAddProbe(TestScenario):
    kernel = "scale_add"
    probes = ["scale.data_out", "data_out"]  # 선언만 하면 끝
```

| Probe 형식 | 예시 | 의미 |
|---|---|---|
| 단순 이름 | `"data_out"` | 해당 텐서의 PULL/RECV op에 `probe=True` 자동 적용 |
| 점(.) 구분 | `"scale.data_out"` | 서브커널 내부 텐서. golden chain pool에서 golden 자동 추출, `set_internal_probe_golden` 자동 호출 |

**동작 흐름:**

1. `TestScenario.run()` → `ctx._register_declarative_probes(probes)`
2. `ki.run(ctx)` 실행 — ops 기록 + `forward()` 호출 (golden chain pool 저장)
3. `ctx.run()` 시작 시:
   - `_apply_declarative_probes()`: 출력 probe → `op.probe = True` 사후 설정
   - `_resolve_internal_probe_golden()`: 내부 probe → `_golden_pool`에서 golden 추출
   - Engine `_ensure_probe_mappings()`: `INTERNAL` → `INTERNAL_PROBE` 동적 업그레이드

**내부 probe 요구사항:** golden chain이 설정되어 있어야 한다 (각 서브커널의 `forward()` 반환값으로 자동 구성).

#### 수동 Probe API

기존 수동 API도 그대로 사용 가능하다. `run()` override 시에는 수동 API를 사용한다.

**출력 Probe:**

```python
h_pull = ctx.pull_tensor(k.data_out, dep=h_push, probe=True)
ctx.verify(h_pull, k.forward()["data_out"])  # golden이 SHM에도 전달됨
```

**내부 Probe:**

```python
# 수동 golden 등록
ctx.set_internal_probe_golden("scale", "data_out", scale_golden)
```

Mismatch 발생 시 `ProbeMismatchError`가 raise되며, beat/cycle/expected/actual 정보를 포함한다.

#### GUI 모드: Waveform 디버깅

```bash
vten run --kernel my_accel --test TestMyAccel --gui       # interactive
vten run --kernel my_accel --test TestMyAccel --waveform  # waveform 덤프
```

GUI 모드에서 probe mismatch 발생 시 `$stop`으로 시뮬레이션이 일시정지된다.
xsim waveform 창에서 mismatch 시점의 신호를 확인하고, restart로 반복 디버깅이 가능하다.
Python은 restart 감지 후 CMD_READY를 재전송하여 세션을 유지한다.

---

## 4. Level 2: Functional API

DSL 연산을 자동 생성하는 고수준 API. PyTorch 파이프라인에 vten 연산을 삽입할 때 사용.

### 4.1 run_kernel — 1회성 호출

```python
from vten import run_kernel

x = torch.randn(1024, dtype=torch.int8)
outputs = run_kernel(
    PassthroughKernel,
    {"data_in": x},              # H2D 텐서: 이름 → torch.Tensor
    backend=xsim_backend,
    params={"N": 1024},
    configure=True,              # auto_bind 레지스터 일괄 설정
)
y = outputs["data_out"]          # D2H 텐서 자동 역직렬화
```

**자동 생성되는 DSL:**
1. `inputs`에 있는 텐서 → `ctx.send_tensor()` (= load + push)
2. `configure=True` → `ctx.configure()`
3. `inputs`에 없는 텐서 → `ctx.recv_tensor()` (= pull + store)

### 4.2 KernelExecutor — 반복 호출 + 자동 alias

연속 호출 시 이전 output을 다음 input으로 넘기면 **자동으로 alias를 적용**하여 버퍼를 재사용한다.

```python
from vten import KernelExecutor

npu = KernelExecutor(NPU3DKernel, backend=backend, configure=True)

x = input_tensor
for i, layer_params in enumerate(layers):
    x = npu(
        ifm=x,
        wgt=weights[i],
        bias=biases[i],
        _params=layer_params,
    )["ofm"]
# x가 최종 출력 — alias가 자동으로 cross-batch 버퍼 재사용 처리
```

**Auto-Alias 동작:**
- `KernelExecutor`는 이전 호출의 output 텐서 `id()`를 추적
- 다음 호출의 input에 이전 output 텐서가 그대로 전달되면:
  - `ctx.alias(prev_output_tensor, current_input_tensor)` 자동 호출
  - **LOAD 스킵**: alias target은 SHM에 이미 데이터가 있으므로 LOAD 불필요
  - **STORE 스킵**: alias source는 다음 batch에서 재사용되므로 STORE 불필요
  - SHM Data Region은 batch 간 보존

### 4.3 Level 1 vs Level 2 비교

동일한 작업을 두 레벨로 구현한 예시:

**Level 1 (DSL):**
```python
ctx = ExecutionContext(backend=backend, project_params={"N": 1024})
k = ctx.instantiate(PassthroughKernel, N=1024)
k.data_in.data = x

h_send = ctx.send_tensor(k.data_in)
h_recv = ctx.recv_tensor(k.data_out, dep=h_send)

result = ctx.run()
y = result.output_tensors["data_out"]
```

**Level 2 (Functional):**
```python
y = run_kernel(PassthroughKernel, {"data_in": x},
               backend=backend, params={"N": 1024})["data_out"]
```

---

## 5. Workflow: 단일 커널

### 5.1 프로젝트 초기화

```bash
$ vten init my_project --kernel passthrough
```

```
my_project/
├── vten.toml
├── rtl/
│   └── passthrough.sv
└── kernels/
    └── passthrough/
        ├── kernel_spec.yaml
        ├── passthrough_kernel.py
        └── tests/
            └── test_passthrough.py
```

### 5.2 Kernel 정의

1. RTL 작성 (`rtl/passthrough.sv`)
2. `kernel_spec.yaml` 작성 — 인터페이스, 프로토콜, 패킹 선언
3. `passthrough_kernel.py` 작성 — Tensor 선언 + `forward()` golden 모델

### 5.3 테스트 작성

**TestScenario 방식** (CLI `vten run` 사용 시):

```python
from vten import TestScenario

class TestPassthrough(TestScenario):
    kernel = "passthrough"

    def run(self, ctx, cfg):
        k = ctx.instantiate(PassthroughKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_push)
        ctx.verify(h_pull, k.forward()["data_out"])
```

**Functional API 방식** (pytest에서 직접 사용 시):

```python
def test_passthrough(xsim_backend):
    x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
    result = run_kernel(
        PassthroughKernel,
        {"data_in": x},
        backend=xsim_backend,
        params={"N": 1024},
    )
    assert torch.equal(result["data_out"], x)
```

### 5.4 빌드 & 실행

```bash
# 빌드 (5-stage pipeline)
$ vten build --kernel passthrough

# 실행
$ vten run --kernel passthrough --test test_passthrough

# 모든 테스트 실행 (--test 생략)
$ vten run --kernel passthrough

# 파형 덤프 포함 실행
$ vten run --kernel passthrough --test test_passthrough --waveform

# 실패 시에만 파형 저장
$ vten run --kernel passthrough --test test_passthrough --waveform-on-fail

# GUI 모드
$ vten run --kernel passthrough --test test_passthrough --gui
```

### 5.5 빌드 파이프라인 단계

```
Stage 1: project_setup  →  Vivado 프로젝트 생성 (project_setup.tcl)
Stage 2: dpi_c          →  SHM 브릿지 공유 라이브러리 빌드
Stage 3: codegen         →  tb_top.sv, BFM 인스턴스, 와이어 선언 생성
Stage 4: compile_order   →  resolve_order.tcl 실행 → filelist 생성
Stage 5: compile         →  xvlog + xelab → xsim.dir 생성
```

---

## 6. Workflow: Multi-Layer Pipeline

PyTorch 파이프라인의 일부를 vten으로 오프로드하는 패턴.

### 6.1 개요

```
Step 1 (torch)    Step 2 (vten)       Step 3 (torch)
───────────────   ─────────────────   ───────────────
preprocess(x)  →  npu(ifm=x)["ofm"]  →  postprocess(y)
  CPU/GPU            RTL sim/FPGA         CPU/GPU
```

### 6.2 단일 레이어 오프로드

```python
import torch
from vten import run_kernel
from my_kernels import Conv3DKernel

# Step 1: PyTorch 전처리
x = preprocess(raw_input)

# Step 2: RTL 시뮬레이션으로 연산
y = run_kernel(
    Conv3DKernel,
    {"ifm": x, "weight": w, "bias": b},
    backend=backend,
    params={"IN_CH": 64, "OUT_CH": 128, ...},
    configure=True,
)["ofm"]

# Step 3: PyTorch 후처리
output = postprocess(y)
```

### 6.3 Multi-Layer (KernelExecutor)

```python
from vten import KernelExecutor

npu = KernelExecutor(NPU3DKernel, backend=backend, configure=True)

# 6-layer inference pipeline
x = input_tensor
for i in range(6):
    x = npu(
        ifm=x,               # layer 1+에서 auto-alias 자동 적용
        wgt=weights[i],
        bias=biases[i],
        _params=layer_configs[i],
    )["ofm"]

final_output = x  # 최종 결과 torch.Tensor
```

**내부 동작 (layer 2부터):**

```
Layer 1: send(ifm) → send(wgt) → configure → recv(ofm)
                                                  │
         ┌──── id(output["ofm"]) 저장 ────────────┘
         │
Layer 2: alias(prev.ofm→cur.ifm) → send(wgt) → configure → recv(ofm)
         └── LOAD 스킵, SHM 버퍼 재사용 ──┘
```

### 6.4 Golden Matching 패턴

RTL 결과를 PyTorch golden과 비교:

```python
# PyTorch golden
golden = torch.nn.functional.conv3d(x, w, bias=b, ...)

# RTL 결과
hw_out = run_kernel(Conv3DKernel, {"ifm": x, "weight": w, "bias": b},
                    backend=backend, params=params, configure=True)["ofm"]

# 비교
assert torch.allclose(hw_out.float(), golden.float(), atol=1e-3)
```

---

## 7. CLI Workflow

### 7.1 명령어 요약

| 명령 | 용도 |
|------|------|
| `vten init <dir>` | 프로젝트 스켈레톤 생성 |
| `vten init <dir> --kernel <name>` | 기존 프로젝트에 커널 추가 |
| `vten build` | 전체 빌드 |
| `vten build --kernel <name>` | 특정 커널만 빌드 |
| `vten build --upto codegen` | 특정 단계까지만 빌드 |
| `vten build --skip-compile` | 코드 생성만 (컴파일 생략) |
| `vten run --kernel <name> --test <test>` | 특정 테스트 실행 |
| `vten run --kernel <name>` | 모든 테스트 실행 (--test 생략) |
| `vten run ... --waveform` | 파형 덤프 포함 |
| `vten run ... --gui` | xsim GUI 모드 |
| `vten run ... --waveform-on-fail` | 실패 시에만 파형 저장 |
| `vten report` | 결과 리포트 |

### 7.2 전형적인 개발 사이클

```bash
# 1. 프로젝트 생성
$ vten init my_npu --kernel conv3d

# 2. RTL + kernel_spec.yaml + kernel.py 작성
$ vim rtl/conv3d.sv
$ vim kernels/conv3d/kernel_spec.yaml
$ vim kernels/conv3d/conv3d_kernel.py

# 3. 테스트 작성
$ vim kernels/conv3d/tests/test_conv3d.py

# 4. 빌드 (코드 생성 → 컴파일)
$ vten build --kernel conv3d

# 5. 실행
$ vten run --kernel conv3d --test test_conv3d

# 5-b. 모든 테스트 실행 (--test 생략)
$ vten run --kernel conv3d

# 6. 디버그 (파형 확인)
$ vten run --kernel conv3d --test test_conv3d --waveform --gui

# 7. 결과 확인
$ vten report
```

---

## 8. API Quick Reference

### Import

```python
# 모든 사용자 API는 from vten import ... 한 줄로 접근
from vten import Kernel, Tensor, Direction, register     # 커널 정의
from vten import CompositeKernel                         # 컴포지트
from vten import TestScenario                            # 테스트
from vten import run_kernel, KernelExecutor              # 고급 API
```

### Kernel Definition

```python
from vten import Kernel, Tensor, register

class MyKernel(Kernel):
    spec = "kernels/my_kernel/kernel_spec.yaml"
    t_in  = Tensor(shape=(...), dtype=torch.int8, interface="iface_name")
    t_out = Tensor(shape=(...), dtype=torch.int8, interface="iface_name")
    ctrl  = register("ctrl")                       # 레지스터 인터페이스

    def generate_inputs(self, seed=None): ...       # 입력 생성 (auto-chain 가능)
    def forward(self, **inputs) -> dict[str, torch.Tensor]: ...  # Golden 계산
```

### ExecutionContext (Level 1)

```python
ctx = ExecutionContext(backend=b, project_params={...})
k = ctx.instantiate(MyKernel, **params)

# 연산
h = ctx.load_tensor(t) / ctx.store_tensor(t)       # Host ↔ SHM
h = ctx.push_tensor(t) / ctx.pull_tensor(t)        # SHM ↔ DUT
h = ctx.send_tensor(t) / ctx.recv_tensor(t)        # Shorthand (load+push / pull+store)
h = ctx.write_register(reg, {f: v}) / ctx.read_register(reg, f) / ctx.poll_register(reg, f)
h = ctx.configure(k) / ctx.barrier()

# 의존성 & 검증
ctx.verify(h, golden)
ctx.alias(src, dst)
h.add_commit_dependency(other_h)
ctx.set_internal_probe_golden(sub, tensor, golden)  # 내부 probe golden 등록

result = ctx.run()  # → BatchResult
```

### Functional API (Level 2)

```python
from vten import run_kernel, KernelExecutor

# 1회성
outputs = run_kernel(MyKernel, {"t_in": x}, backend=b, params={...}, configure=True)

# 반복 (auto-alias, multi-batch session 자동 관리)
npu = KernelExecutor(MyKernel, backend=b, params={...}, configure=True)
y = npu(t_in=x, _params={...})["t_out"]
```

### Multi-Config (Level 1 — 단일 배치, 복수 config)

```python
ctx = ExecutionContext(backend=b, project_params={"N": 1024})

for cfg in [{"scale_factor": 1}, {"scale_factor": 2}, {"scale_factor": 3}]:
    ki = ctx.instantiate(MyKernel, **cfg)
    ki.generate_inputs(seed=42)
    h_load = ctx.load_tensor(ki.data_in)
    h_push = ctx.push_tensor(ki.data_in, dep=h_load)
    h_pull = ctx.pull_tensor(ki.data_out, dep=h_push)
    ctx.verify(h_pull, ki.forward())
    ctx.config_boundary()           # BARRIER 삽입 + config_group 증가

result = ctx.run()                  # 1회 xsim 실행으로 3개 config 검증
```

### Multi-Batch Session (Level 2 — KernelExecutor)

```python
with KernelExecutor(MyKernel, backend=b, params={...}) as npu:
    # 배치 1
    y1 = npu(t_in=x1)["t_out"]     # open_session → submit → wait
    assert torch.equal(y1, expected1)

    # 배치 2 (동일 xsim 프로세스 재사용)
    y2 = npu(t_in=x2)["t_out"]     # submit_batch → wait_batch
    assert torch.equal(y2, expected2)
                                    # close_session (context manager exit)
```

---

## 9. Config & Parameter System

커널 개발 시 파라미터를 **어디에, 어떻게** 정의하고, 실행 시점에 어떻게 결합되는지 규정한다.

### 9.1 문제 정의

복잡한 가속기 검증에서 반복되는 문제:

1. **HW 상수 하드코딩** — `Ti=32, To=32, AXI_DW=256` 등이 모든 커널 파일에 복사됨
2. **compute_derived_params 수동 체이닝** — 6개 sub-kernel의 derived shape를 직접 import/호출/merge
3. **build-time vs runtime 구분 없음** — synthesis-time constant와 per-invocation config가 같은 `parameters:` 사전에 혼재
4. **register 매핑 분산** — config register 값 결정이 커널/spec/테스트/binder 4곳에 걸쳐 있음
5. **테스트 boilerplate** — 전체 테스트의 90%가 shape 계산/파라미터 전달 코드

### 9.2 설계 목표

- HW 상수를 한 곳(`kernel_spec.yaml` `build_params:`)에 정의하고 모든 커널이 참조
- Config register 값을 `runtime_params:` + register 매핑으로 선언적 정의
- Composite가 sub-kernel derived params를 자동 체이닝
- TestScenario에서 override할 파라미터만 기술, 나머지는 default cascade
- `generate_inputs()`는 `self.*`로만 파라미터 접근 (인자 전달 제거)

### 9.3 Config 계층 구조 — 4-Tier Merge

파라미터는 4개 계층에서 merge된다. 나중 계층이 높은 우선순위.

```
Tier 1 (lowest)  project_params       ← vten.toml [parameters]
Tier 2            build_params         ← vten.toml [build_params] + kernel_spec build_params:
Tier 3            kernel_spec_params   ← kernel_spec parameters: + runtime_params defaults
Tier 4 (highest)  test_override        ← ctx.instantiate() kwargs / TestScenario cfg
```

**Merge 규칙:**

```python
namespace = {}
namespace.update(project_params)          # Tier 1
namespace.update(build_params)            # Tier 2 (project [build_params] < spec build_params)
namespace.update(kernel_spec_parameters)  # Tier 3a
# Tier 3b: runtime_params defaults — setdefault (같은 키가 이미 있으면 유지)
for k, v in runtime_params_defaults.items():
    namespace.setdefault(k, v)
namespace.update(test_override)           # Tier 4
```

> **주의:** `runtime_params` defaults와 `parameters:`는 **동일 계층** (Tier 3, kernel_spec level).
> `runtime_params`의 default는 `parameters:`를 **보충** (같은 키가 parameters에 이미 정의되어 있으면 그대로 유지).

**build_params override 정책:** test_override(Tier 4)가 build_param 키를 덮으면 **warning** 출력 후 허용.
synthesis-time constant를 런타임에 변경하는 것은 위험하지만, 테스트 유연성을 위해 차단하지는 않는다.

### 9.4 kernel_spec.yaml 확장

#### 9.4.1 build_params

RTL 파라미터/synthesis-time 상수. Verilog `parameter`, VHDL `generic`에 해당.

```yaml
kernel: act_quant
rtl_top: act_quant_core

build_params:
  Ti: 32
  To: 32
  OUT_GROUP: 2
  PSUM_BITS: 32
  BIAS_BITS: 32
  QUANT_BITS: 8
  SHIFT_BITS: 4
  MAX_SHIFT: 16

parameters: {}

runtime_params:
  # ... (§9.4.2 참조)

interfaces:
  # ... (기존과 동일)
```

**규칙:**
- `build_params`는 flat `dict[str, int | str]`
- 모든 sub-kernel이 동일한 `build_params` namespace를 공유 (Tier 2에서 merge)
- 프로젝트 레벨 `vten.toml [build_params]` < kernel_spec `build_params:` (spec이 우선)
- 커널 Python 코드에서 `self.Ti`, `self.To` 등으로 접근 가능

**vten.toml에서 build_params:**

```toml
[build_params]
Ti = 32
To = 32
AXI_DW = 256

[parameters]
# 기존과 동일 — runtime 기본값
```

#### 9.4.2 runtime_params

Per-invocation config. 각 테스트 실행마다 바뀔 수 있는 파라미터.

**두 가지 형식:**

```yaml
runtime_params:
  # 형식 1: scalar — default 값만 지정 (기존 parameters:와 동일 동작)
  in_depth: 4
  in_height: 4
  in_width: 4

  # 형식 2: dict — default + register 매핑
  in_ch:
    default: 32
    register: ctrl.in_ch        # → auto WRITE_REG(ctrl, in_ch, <resolved_value>)
  out_ch:
    default: 32
    register: ctrl.out_ch
  bias_shift:
    default: 8
    register: ctrl.bias_shift
  is_relu:
    default: 1
    register: ctrl.is_relu
  ifm_stride:
    default: 1
    register: ctrl.ifm_stride
  ofm_stride:
    default: 1
    register: ctrl.ofm_stride
```

**데이터 모델:**

runtime_params 항목은 scalar 값 또는 `{default, register}` dict로 기술한다.

- scalar 값은 backward compat (`parameters:`에 넣은 것과 동일)
- dict 값은 `default` + `register` 필드를 가지는 매핑으로 파싱
- `register: ctrl.field_name` → `ctx.configure()` 시 해당 register에 자동 WRITE_REG 생성

**register 매핑 동작:**
1. `ctx.configure(kernel)` 호출 시 기존 role="config" auto-match **이후에** runtime_params register 매핑을 추가 처리
2. 동일 register를 role="config"과 runtime_params 둘 다 잡는 경우, runtime_params가 **우선** (나중에 append → 덮어씀)
3. register 필드 이름은 `kernel_spec.yaml` interfaces 섹션의 register 정의와 정확히 일치해야 함

#### 9.4.3 완전한 kernel_spec.yaml 예시

```yaml
kernel: act_quant
rtl_top: act_quant_core

clock:
  name: ap_clk

reset:
  name: ap_aresetn
  active_low: true

build_params:
  Ti: 32
  To: 32
  OUT_GROUP: 2
  PSUM_BITS: 32

parameters: {}

runtime_params:
  in_ch:       { default: 32, register: ctrl.out_ch }
  out_ch:      { default: 32, register: ctrl.out_ch }
  in_depth:    { default: 4,  register: ctrl.in_depth }
  in_height:   { default: 4,  register: ctrl.in_height }
  in_width:    { default: 4,  register: ctrl.in_width }
  bias_shift:  { default: 8,  register: ctrl.bias_shift }
  is_relu:     { default: 1,  register: ctrl.is_relu }
  ifm_stride:  { default: 1,  register: ctrl.ifm_stride }
  ofm_stride:  { default: 1,  register: ctrl.ofm_stride }

interfaces:
  ctrl:
    protocol: axi4_lite
    role: slave
    data_width: 32
    rtl_port: s_axilite_control
    generate_controller: true
    registers:
      - { name: vsync,      width: 1, pulse: true }
      - { name: in_depth,   width: 8 }
      - { name: in_height,  width: 8 }
      - { name: in_width,   width: 8 }
      - { name: out_ch,     width: 9 }
      - { name: bias_shift, width: 5 }
      - { name: is_relu,    width: 1 }
      - { name: ifm_stride, width: 2 }
      - { name: ofm_stride, width: 2 }

  psum_in:
    protocol: axi4_stream
    role: slave
    data_width: 64
    packing:
      element_width: 32
      elements_per_beat: 2

  bias_in:
    protocol: axi4_stream
    role: slave
    data_width: 64
    packing:
      element_width: 32
      elements_per_beat: 2

  quant_out:
    protocol: axi4_stream
    role: master
    data_width: 16
    packing:
      element_width: 8
      elements_per_beat: 2
```

### 9.5 Kernel 클래스 규약

#### 9.5.1 compute_derived_params

텐서 shape에 필요한 derived 파라미터를 계산하는 인스턴스 메서드. `self.*`로 파라미터에 접근한다.

```python
class ActQuantKernel(Kernel):
    spec = "kernels/act_quant/kernel_spec.yaml"

    def compute_derived_params(self) -> dict:
        """build_params + runtime_params에서 텐서 shape용 값 계산."""
        och_groups = (self.out_ch + self.To - 1) // self.To

        if self.ifm_stride == 2:
            eff_depth = self.in_depth // 2
            eff_height = self.in_height // 2
            eff_width = self.in_width // 2
        elif self.ofm_stride == 2:
            eff_depth = self.in_depth * 2
            eff_height = self.in_height * 2
            eff_width = self.in_width * 2
        else:
            eff_depth = self.in_depth
            eff_height = self.in_height
            eff_width = self.in_width

        total_beats = eff_depth * och_groups * eff_height * eff_width * (self.To // self.OUT_GROUP)
        total_psum_elems = total_beats * self.OUT_GROUP

        return {
            "total_psum_elems": total_psum_elems,
            "total_bias_elems": self.out_ch,
            "total_output_elems": total_psum_elems,
            "eff_depth": eff_depth,
            "eff_height": eff_height,
            "eff_width": eff_width,
        }
```

**핵심:** resolver가 build_params + runtime_params를 `self.*` attribute로 주입하므로, `self.To` 등으로 HW 상수에 접근한다. 커널 코드에 상수를 하드코딩할 필요 없음.

#### 9.5.2 generate_inputs — self.* 규약

`generate_inputs()`는 명시적 파라미터 인자를 받지 않는다. 모든 파라미터는 `self.*`를 통해 접근.

```python
# ✅ 올바른 패턴
def generate_inputs(self, seed=None):
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # 파라미터는 self.* 로 접근 (resolver가 주입)
    out_ch = self.out_ch
    total_psum = self.total_psum_elems   # compute_derived_params 결과도 self.*

    psum_data = torch.randint(-50000, 50000, (total_psum,),
                              dtype=torch.int32, generator=rng)
    bias_data = torch.randint(-10000, 10000, (out_ch,),
                              dtype=torch.int32, generator=rng)

    self.psum_in.data = psum_data
    self.bias_in.data = bias_data

# ❌ 잘못된 패턴 — 파라미터를 인자로 받지 않음
def generate_inputs(self, seed=None, in_ch=32, out_ch=32, ...):
    ...
```

**이유:** `ctx.instantiate()` 시 resolver가 파라미터를 resolve하고 커널 인스턴스의 attribute로 주입한다. `generate_inputs()`에 같은 파라미터를 다시 전달하면 불일치 위험이 생긴다.

#### 9.5.2.1 Auto-chain generate_inputs (Composite Registry)

CompositeKernel의 서브 커널이 자체 `generate_inputs()`를 정의하지 않은 경우,
framework가 자동으로 upstream 체인을 실행하여 입력을 생성한다.

```python
# PsumBufferKernel은 NpuPipelineKernel의 서브커널
# generate_inputs()를 정의하지 않아도 standalone 검증 가능

psum = ctx.instantiate(PsumBufferKernel, **params)
psum.generate_inputs(seed=42)
# → _lookup_composite()로 NpuPipelineKernel 발견
#   (실패 시 _discover_composite()로 sibling 디렉토리 자동 스캔)
# → reverse-BFS: upstream = {wl, fmap, mac} (target 제외, 사이클 회피)
# → upstream-only topo sort: [wl, fmap, mac]
# → wl.generate_inputs() + wl.forward() 실행
# → fmap.generate_inputs() + fmap.forward() 실행
# → mac.forward(ifmap=pool, weight=pool) 실행
#   → mac.forward()는 HW와 동일한 21-bit packed stream bytes 반환
# → psum.psum_in.data = pool[mac.partial_sum]  (physical format)
```

**동작 방식:**
1. `__init_subclass__`에서 `_composite_registry`에 sub-kernel → composite 매핑 자동 등록
2. `Kernel.generate_inputs()`가 자체 구현 없으면:
   - `_lookup_composite()`로 registry 조회 (class identity tolerance — re-import 허용)
   - 실패 시 `_discover_composite()`로 sibling 커널 디렉토리 자동 스캔
3. `_generate_inputs_for()`에서:
   - `_same_kernel_class()`로 target의 sub-ref name 매칭
   - reverse-BFS로 **upstream 의존성만** 수집 (target 제외 → 사이클 회피)
   - target의 resolved params (`_resolver.namespace`)를 upstream에 전달
   - upstream 순회: `generate_inputs()` → pool에서 connected inputs 설정 → `forward()`
4. target 커널의 connected input 텐서에 `data` + `logical_data` 설정

**데이터 흐름 (composite forward()와 동일):**
- Exposed inputs: `logical_data` → `layout_{name}()` → physical
- Connected inputs: pool에서 직접 전달 (forward() 출력 = HW physical format)
- forward() 출력 → pool로 전파 → downstream connected inputs로 전달

**사용자 작업:** source 커널(incoming connection 없는 커널)만 `generate_inputs()` 구현.
나머지 커널은 `forward()`만 구현하면 standalone 검증에서도 자동으로 입력을 얻는다.

**NPU 파이프라인 예시:**
```python
# source 커널: generate_inputs() 구현
class WeightLoaderKernel(Kernel):
    def generate_inputs(self, seed=None): ...  # ✅ 외부 입력 직접 생성

# downstream 커널: forward()만 구현, generate_inputs() 불필요
class PsumBufferKernel(Kernel):
    def forward(self, **inputs): ...           # ✅ auto-chain이 입력 전파

class ActQuantKernel(Kernel):
    def forward(self, **inputs): ...           # ✅ auto-chain이 입력 전파
```

#### 9.5.3 forward — HW Behavioral Model

`forward(self, **inputs)` 는 **HW의 동작을 그대로 서술하는 behavioral model**이다.
입력과 출력 모두 HW I/O와 동일한 physical format을 사용한다.

```python
def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """인자명 = input tensor 이름, 반환값 = {output tensor 이름: data}
    입출력은 HW와 동일한 physical format."""
```

**핵심 규칙:**
1. **인자 이름** = input tensor의 `name` (Kernel class에서 선언한 attribute 이름)
2. **반환값** = `{output_tensor_name: data}` dict — HW 출력과 동일한 format
3. **HW physical format** — forward()는 HW의 I/O를 그대로 재현
4. **params**는 `self.*`로 접근 (resolved params가 instance attrs로 노출)

**예시 — act_quant (단순 연산):**
```python
def forward(self, **inputs) -> dict[str, torch.Tensor]:
    psum_flat = inputs.get("psum_in", self.psum_in.logical_data)
    bias_data = inputs.get("bias_in", self.bias_in.logical_data)
    # ... ReLU → shift → clip ...
    return {"quant_out": (clipped & 0xFF).flatten().to(torch.uint8)}
```

**예시 — mac_atu (serialization 포함):**
```python
def forward(self, **inputs) -> dict[str, torch.Tensor]:
    """HW behavioral model: physical IFM/weight streams → packed psum streams."""
    mac_result = self._compute_einsum(**inputs)  # MAC 연산
    packed = self._serialize_psum(mac_result)     # 21-bit packed stream bytes
    return {"partial_sum": packed}                # HW 출력 format 그대로
```

**예시 — psum_buffer (deserialization + accumulation):**
```python
def forward(self, **inputs) -> dict[str, torch.Tensor]:
    """HW behavioral model: packed psum streams → accumulated output."""
    mac_result = self._deserialize_psum(inputs["psum_in"])  # 21-bit unpack
    # ... accumulation, stride handling ...
    return {"psum_out": result.flatten().to(torch.int32)}
```

**auto-chain과의 관계:** forward()가 HW physical format을 반환하므로,
upstream.forward() 출력이 connection을 통해 downstream에 전달될 때
별도의 layout/unlayout 변환 없이 직접 사용된다.

### 9.6 CompositeKernel — 자동 체이닝

#### 9.6.1 compute_derived_params 자동 체이닝

`CompositeKernel`은 모든 sub-kernel의 `compute_derived_params()`를 자동 호출하고 결과를 merge한다.
v2에서 `compute_derived_params`는 인스턴스 메서드이므로, composite는 각 sub-kernel 인스턴스의 메서드를 순차 호출한다.

```python
class CompositeKernel(Kernel):
    def compute_derived_params(self) -> dict:
        """Auto-chain: 모든 sub-kernel의 compute_derived_params()를 호출하고 merge."""
        derived = {}
        for attr_name in self._sub_kernel_names:
            sub = getattr(self, attr_name)
            sub_derived = sub.compute_derived_params()
            derived.update(sub_derived)
        return derived
```

**효과:**
- `ctx.instantiate(NpuPipelineKernel, in_ch=128, out_ch=128)` 만으로 동작
- 6개 sub-kernel의 compute_derived_params 결과가 composite namespace에 자동 merge
- sub-kernel 텐서의 `${...}` 참조가 모두 resolve됨
- `_run_pipeline()` 같은 수동 shape merge helper **불필요**

#### 9.6.2 Composite에서의 파라미터 흐름

```
ctx.instantiate(NpuPipelineKernel, in_ch=128, out_ch=128, bias_shift=10)
     │
     ▼ Tier 4: test_override = {in_ch: 128, out_ch: 128, bias_shift: 10}
     │
     ▼ ParameterResolver 4-tier merge
     │  → namespace = {Ti: 32, To: 32, ..., in_ch: 128, out_ch: 128, bias_shift: 10, ...}
     │
     ▼ composite.compute_derived_params()
     │  → calls each sub-kernel instance's compute_derived_params()
     │  → merges {total_psum_elems, ifm_total_bytes, ddr_wgt_bytes, ...}
     │
     ▼ Sub-kernel resolver에 composite namespace 전파
     │  → 각 sub-kernel의 텐서 shape ${...} 참조 resolve
     │
     ▼ configure() 시:
        → role="config" auto-match + runtime_params register 매핑
        → 각 sub-kernel의 ctrl register에 resolved 값 기록
```

### 9.7 TestScenario — Override-Only Config

TestScenario에서는 default와 **다른** 파라미터만 override하면 된다.

#### 9.7.1 기본 테스트 (default config)

```python
class TestActQuant(TestScenario):
    """Default config (all runtime_params defaults)."""
    kernel = "act_quant"
    # run() 미정의 → 기본 run() 사용:
    #   1. kernel auto-discover
    #   2. ctx.instantiate(kernel_cls, **cfg)
    #   3. k.generate_inputs(seed=42)
    #   4. k.run(ctx)
```

이것만으로 `act_quant`의 default config (in_ch=32, out_ch=32, ...) 전체 검증이 완료된다.

#### 9.7.2 Override 테스트

```python
class TestActQuantLargeChannel(TestScenario):
    """Large channel count."""
    kernel = "act_quant"
    cfg = {"in_ch": 128, "out_ch": 256}

class TestActQuantStride2(TestScenario):
    """IFM stride=2 downsampling."""
    kernel = "act_quant"
    cfg = {"ifm_stride": 2, "in_depth": 8, "in_height": 8, "in_width": 8}

class TestActQuantNoRelu(TestScenario):
    """Signed output (no ReLU)."""
    kernel = "act_quant"
    cfg = {"is_relu": 0}
```

**Before vs After 비교** (NPU pipeline 테스트):

```python
# ── BEFORE: 391줄 중 실제 로직 10줄, 나머지 boilerplate ──

class TestNpuPipeline(TestScenario):
    kernel = "npu_pipeline"

    def run(self, ctx, cfg):
        from weight_loader.weight_loader_kernel import compute_shapes as wgt_cs
        from fmapIO.fmapIO_kernel import compute_shapes as fmap_cs
        from act_quant.act_quant_kernel import compute_shapes as aq_cs
        from psum_buffer.psum_buffer_kernel import compute_shapes as psb_cs
        from mac_atu.mac_atu_kernel import compute_shapes as mac_cs
        from bias_loader.bias_loader_kernel import compute_shapes as bl_cs

        p = self._run_pipeline(cfg)   # 40줄: 6개 compute_shapes 호출/merge
        k = ctx.instantiate(NpuPipelineKernel, **p)
        k.generate_inputs(
            seed=42, in_ch=p["in_ch"], out_ch=p["out_ch"],
            in_width=p["in_width"], in_height=p["in_height"],
            in_depth=p["in_depth"], kernel_size=p["kernel_size"],
            ifm_stride=p["ifm_stride"], ofm_stride=p["ofm_stride"],
            bias_shift=p["bias_shift"], is_relu=p["is_relu"],
        )
        k.run(ctx)

# ── AFTER: override만 ──

class TestNpuPipeline(TestScenario):
    kernel = "npu_pipeline"
    # default run() 사용 — 자동으로 모든 sub-kernel shape 계산

class TestNpuPipelineLarge(TestScenario):
    kernel = "npu_pipeline"
    cfg = {"in_ch": 128, "out_ch": 128, "in_depth": 8}

class TestNpuPipelineStride2(TestScenario):
    kernel = "npu_pipeline"
    cfg = {"ifm_stride": 2}
```

### 9.8 ParameterResolver 내부 동작

```python
class ParameterResolver:
    def __init__(
        self,
        project_params: dict,
        kernel_params: dict,
        runtime_params: dict,
        build_params: dict | None = None,
        default_params: dict | None = None,
    ) -> None:
        self._build_param_keys: set[str] = set()
        self.namespace = {}

        # Tier 1
        self.namespace.update(project_params)

        # Tier 2
        if build_params:
            self.namespace.update(build_params)
            self._build_param_keys = set(build_params.keys())

        # Tier 3a
        self.namespace.update(kernel_params)

        # Tier 3b: runtime_params defaults (setdefault — 보충만)
        if default_params:
            for k, v in default_params.items():
                self.namespace.setdefault(k, v)

        # Tier 4: test override (build_param 키 override 시 warning)
        if self._build_param_keys:
            for k in runtime_params:
                if k in self._build_param_keys:
                    logger.warning(
                        "runtime param '%s' overrides build_param "
                        "(synthesis-time constant)", k
                    )
        self.namespace.update(runtime_params)

        self._resolve_namespace()
```

**Backward compatibility:**
- `build_params=None`, `default_params=None` 기본값
- 기존 3인자 호출 `ParameterResolver(proj, kern, rt)` 그대로 작동
- 새 필드가 없는 kernel_spec은 빈 dict → 기존 동작

### 9.9 Binder — runtime_params Register 매핑

`ctx.configure(kernel)` 호출 시, 기존 role="config" auto-match에 **추가하여** runtime_params의 `register:` 필드를 처리한다.

```python
def resolve_runtime_param_registers(view: FlattenedKernelView) -> list[RegisterBindingEntry]:
    """runtime_params.register 필드 기반 명시적 register 매핑."""
    result = []
    for sub_name, sub_ki in view.sub_kernels.items():
        spec = sub_ki.spec
        for param_name, param_spec in spec.runtime_params.items():
            if not isinstance(param_spec, dict) or not param_spec.get("register"):
                continue
            # "ctrl.field_name" → interface="ctrl", field="field_name"
            iface_name, field_name = param_spec["register"].split(".", 1)
            value = sub_ki._resolver.namespace.get(param_name)
            if value is None:
                continue
            # register offset lookup → absolute offset → RegisterBindingEntry
            ...
    return result
```

**Engine 통합:**

```python
# _resolve_auto_binds() 내부
runtime_param_bindings = resolve_runtime_param_registers(view)
view._register_bindings = (
    auto_bindings + config_bindings + composite_config_bindings
    + runtime_param_bindings   # ← 마지막에 추가 → 같은 register가 있으면 덮어씀
)
```

**우선순위:** role="config" auto-match < runtime_params register 매핑 (리스트 뒤쪽이 우선)

### 9.10 Flattener — build_params 전달

`KernelInstance.initialize()`에서 build_params를 merge하여 ParameterResolver에 전달한다.

```python
def initialize(self, project_params: dict) -> None:
    # project [build_params] < kernel_spec build_params
    project_build = project_params.get("build_params", {})
    spec_build = self.spec.build_params
    merged_build = {**project_build, **spec_build} if (project_build or spec_build) else None

    self._resolver = ParameterResolver(
        project_params,
        self.spec.parameters,
        self.runtime_params,
        build_params=merged_build,
        default_params=self.spec.default_params,
    )
```

### 9.11 기존 기능과의 공존

| 기존 기능 | 상태 | 비고 |
|-----------|------|------|
| `parameters:` flat dict | **유지** | Tier 3a에서 그대로 작동 |
| role="config" auto-match | **유지** | runtime_params register 매핑과 공존 |
| `compute_derived_params()` instance method | **유지** | composite에 auto-chain 추가 |
| `TestScenario.run()` override | **유지** | default run()과 선택적 사용 |
| `ctx.instantiate(K, **kwargs)` | **유지** | kwargs가 Tier 4 test_override |

**삭제 없음.** 모든 신규 기능은 additive only. 기존 코드는 변경 없이 작동한다.

---

## 10. Migration Guide

기존 NPU_3D 커널 코드를 새 config system으로 이전하는 가이드.

### 10.1 kernel_spec.yaml

**Before:**

```yaml
kernel: act_quant
rtl_top: act_quant_core

parameters: {}

interfaces:
  ctrl:
    # ... registers 정의 ...
```

**After:**

```yaml
kernel: act_quant
rtl_top: act_quant_core

build_params:
  Ti: 32
  To: 32
  OUT_GROUP: 2

parameters: {}

runtime_params:
  in_ch:       { default: 32, register: ctrl.out_ch }
  out_ch:      { default: 32, register: ctrl.out_ch }
  in_depth:    { default: 4,  register: ctrl.in_depth }
  in_height:   { default: 4,  register: ctrl.in_height }
  in_width:    { default: 4,  register: ctrl.in_width }
  bias_shift:  { default: 8,  register: ctrl.bias_shift }
  is_relu:     { default: 1,  register: ctrl.is_relu }
  ifm_stride:  { default: 1,  register: ctrl.ifm_stride }
  ofm_stride:  { default: 1,  register: ctrl.ofm_stride }

interfaces:
  # ... (기존과 동일)
```

### 10.2 Kernel Python 코드

**Before:**

```python
# 모든 커널 파일에 하드코딩
Ti = 32
To = 32
OUT_GROUP = 2
PSUM_BITS = 32

def compute_shapes(in_ch=32, out_ch=32, ...):
    och_groups = (out_ch + To - 1) // To
    ...

class ActQuantKernel(Kernel):
    @staticmethod
    def compute_derived_params(params: dict) -> dict:
        return compute_shapes(**params)

    def generate_inputs(self, seed=None):
        out_ch = self.out_ch
        ...
```

**After:**

```python
# 하드코딩 제거 — build_params에서 자동 주입

class ActQuantKernel(Kernel):
    spec = "kernels/act_quant/kernel_spec.yaml"

    def compute_derived_params(self) -> dict:
        och_groups = (self.out_ch + self.To - 1) // self.To  # self.*로 접근
        ...
        return {"total_psum_elems": ..., "eff_depth": ..., ...}

    def generate_inputs(self, seed=None):
        out_ch = self.out_ch              # resolver가 주입
        total_psum = self.total_psum_elems  # compute_derived_params 결과
        ...
```

### 10.3 Composite 테스트

**Before (391줄):**

```python
class TestNpuPipeline(TestScenario):
    kernel = "npu_pipeline"

    def _run_pipeline(self, cfg):
        """6개 sub-kernel compute_shapes 수동 호출 + merge."""
        from weight_loader.weight_loader_kernel import compute_shapes as wgt_cs
        from fmapIO.fmapIO_kernel import compute_shapes as fmap_cs
        # ... 4개 더 import ...

        params = {**DEFAULT_PARAMS, **cfg}
        wgt = wgt_cs(**params)
        fmap = fmap_cs(**params)
        # ... 4개 더 호출 ...
        params.update(wgt)
        params.update(fmap)
        # ... merge ...
        return params

    def run(self, ctx, cfg):
        p = self._run_pipeline(cfg)
        k = ctx.instantiate(NpuPipelineKernel, **p)
        k.generate_inputs(
            seed=42,
            in_ch=p["in_ch"], out_ch=p["out_ch"],
            in_width=p["in_width"], in_height=p["in_height"],
            in_depth=p["in_depth"], kernel_size=p["kernel_size"],
            ifm_stride=p["ifm_stride"], ofm_stride=p["ofm_stride"],
            bias_shift=p["bias_shift"], is_relu=p["is_relu"],
        )
        k.run(ctx)
```

**After (3줄):**

```python
class TestNpuPipeline(TestScenario):
    kernel = "npu_pipeline"
    # default run() 자동 사용:
    #   ctx.instantiate() → compute_derived_params auto-chain
    #   → generate_inputs(seed=42) → k.run(ctx)

class TestNpuPipelineLarge(TestScenario):
    kernel = "npu_pipeline"
    cfg = {"in_ch": 128, "out_ch": 128}

class TestNpuPipelineK1(TestScenario):
    kernel = "npu_pipeline"
    cfg = {"kernel_size": 1}
```

### 10.4 vten.toml

**Before:**

```toml
[parameters]
Ti = 32
To = 32
```

**After:**

```toml
[build_params]
Ti = 32
To = 32
AXI_DW = 256

[parameters]
# runtime defaults — 프로젝트 공통
```

### 10.5 마이그레이션 순서

1. `kernel_spec.yaml`에 `build_params:` + `runtime_params:` 추가
2. `vten.toml`에 `[build_params]` 섹션 추가
3. 커널 코드에서 하드코딩된 상수 제거, `self.Ti` 패턴으로 전환
4. `generate_inputs()`에서 명시적 파라미터 인자 제거, `self.*` 사용
5. 테스트에서 `_run_pipeline()` helper 제거, override-only `cfg` 사용
6. 기존 테스트 통과 확인 (`pytest`)
7. (선택) `parameters:`에 남은 값을 `runtime_params:`로 이전
