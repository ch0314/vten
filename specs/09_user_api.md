# vTen User API & Workflow

**Version 0.5.0 — March 2026**

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
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor

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

    def forward(self):
        """Golden reference 계산."""
        return self.data_in.data.clone()
```

### 2.3 CompositeKernel

여러 서브 커널을 하나의 RTL top에 **공간적(spatial)**으로 합성한다.

```python
from vten.kernel.composite import CompositeKernel, Connect, Internal

class NPUTopKernel(CompositeKernel):
    spec = "kernels/npu_top/kernel_spec.yaml"

    # 서브 커널 바인딩
    fmapio = FmapIOKernel.bind({"ctrl": "ctrl/fmapio", "ddr": "ddr"})
    wgt    = WeightLoaderKernel.bind({"ctrl": "ctrl/weight_loader", "hbm": "hbm"})
    mac    = MACKernel.bind({"ctrl": "ctrl/mac"})

    # 내부 RTL 연결 (BFM 불필요)
    connections = [
        Connect(fmapio.ifm_out, mac.ifm_in, Internal()),
        Connect(wgt.wgt_out,    mac.wgt_in,  Internal()),
        Connect(mac.psum_out,   fmapio.ofm_in, Internal(probe=True)),
    ]

    # 외부 노출 텐서
    ifm    = fmapio.ifm.expose("ddr")
    weight = wgt.weight.expose("hbm")
    ofm    = fmapio.ofm.expose("ddr")
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
ctx.verify(h_pull, k.forward())

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
ctx.verify(store, k.forward())
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
from vten.cli.run import TestScenario

class TestPassthrough(TestScenario):
    kernel = "passthrough"

    def run(self, ctx, cfg):
        k = ctx.instantiate(PassthroughKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_push)
        ctx.verify(h_pull, k.forward())
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

### Kernel Definition

```python
class MyKernel(Kernel):
    spec = "kernels/my_kernel/kernel_spec.yaml"
    t_in  = Tensor(shape=(...), dtype=torch.int8, interface="iface_name")
    t_out = Tensor(shape=(...), dtype=torch.int8, interface="iface_name")
    ctrl  = register("ctrl")                       # 레지스터 인터페이스

    def generate_inputs(self, seed=None): ...       # 입력 생성
    def forward(self): ...                          # Golden 계산
    def verify(self, hw_output, golden): ...        # 커스텀 비교 (선택)
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
