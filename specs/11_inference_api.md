# vTen Inference API

**Version 0.7.0 — April 2026**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [Tensor Device Extension](#3-tensor-device-extension)
4. [InferenceSession](#4-inferencesession)
5. [InferenceModule (nn.Module)](#5-inferencemodule-nnmodule)
6. [Backend: Persistent Mode](#6-backend-persistent-mode)
7. [CommandInterpreter: BO Reuse](#7-commandinterpreter-bo-reuse)
8. [ExecutionContext: Inference Mode](#8-executioncontext-inference-mode)
9. [Examples](#9-examples)
10. [API Cleanup](#10-api-cleanup)
11. [Implementation Notes](#11-implementation-notes)

---

## 1. Overview

vTen의 inference API는 verification에서 검증된 `Kernel.run(ctx)` 프로토콜을
그대로 사용하여 실제 FPGA 추론을 수행한다.

```
Layer 3: nn.Module                conv2(conv1(x)).cpu()
                                     │
Layer 2: Eager Execution          session.run(K, inputs={...}, **params)
            │                        │  → dict[str, Tensor(on_device)]
            │                        │
Layer 1: vTen Core                ExecutionContext(mode="inference")
            │                        │  Kernel.run(ctx) — 기존 DSL
            │                        │  compile → XrtBackend(persistent)
            ▼                        ▼
         FPGA Device              CommandInterpreter (BO pool)
```

**핵심 결정:**
- `session.run()` = atomic primitive (단일 커널 실행)
- 결과 = `Tensor(on_device=True)` — device에 상주, `.cpu()`로 host 전송
- Python이 orchestrator — skip connection, concat, multi-branch 모두 Python 코드
- 별도 `DeviceTensor` 클래스 없음 — 기존 `Tensor` 확장

---

## 2. Design Principles

### 2.1 Kernel-Granular Eager Execution

PyTorch eager mode + CUDA device tensor 패턴을 차용한다.
각 `session.run()`이 하나의 커널을 즉시 실행하고, 결과가 device에 남아있다가
다음 커널의 input으로 전달된다.

```python
r1 = session.run(NpuKernel, inputs={"ifm_mem": x, ...}, **layer1)
r2 = session.run(NpuKernel, inputs={"ifm_mem": r1["ofm_mem"], ...}, **layer2)
y = r2["ofm_mem"].cpu()  # 최종만 host로
```

이 모델이 vTen에 자연스러운 이유:

1. **기존 철학과 일치**: `Kernel.run(ctx)`가 atomic execution unit.
   inference에서도 동일한 granularity 유지.

2. **임의 topology 지원**: skip connection, residual, concat — 전부 Python 변수
   할당으로 표현. graph compiler나 alias 일반화 메커니즘 불필요.

3. **디버깅 용이**: 아무 중간 지점에서나 `.cpu()` 호출하여 결과 확인 가능.

### 2.2 Verification과 Inference의 관계

같은 `Kernel.run(ctx)`를 사용하되, `ExecutionContext`의 mode로 동작이 달라진다:

| | Verification | Inference |
|---|---|---|
| `ctx.run(verify=True)` | golden 비교 실행 | **no-op** (verify 무시) |
| `ctx.push_tensor()` (bound) | LOAD+PUSH | **skip** (BO on device) |
| `ctx.pull_tensor()` | PULL + STORE | **PULL only** (device에 남김) |
| `ctx.configure()` | WRITE_REG | WRITE_REG (동일) |
| `ctx.write_register()` | WRITE_REG | WRITE_REG (동일) |
| output | `result.output_tensors` (torch.Tensor) | `dict[str, Tensor(on_device)]` |

### 2.3 No Separate DeviceTensor

기존 `Tensor` 클래스가 이미 선언(shape, dtype)과 런타임 데이터(data)를 함께 관리한다.
device 상태는 이 런타임 데이터의 자연스러운 확장이다.

```python
# 현재
tensor.data = torch.randn(...)     # host 데이터

# 확장
tensor.on_device  # True면 _bo에 FPGA 데이터
tensor.cpu()      # device → host 전송
```

CUDA의 `tensor.is_cuda` / `.cpu()` 패턴과 동일한 사용자 경험.

---

## 3. Tensor Device Extension

### 3.1 추가 필드

```python
class Tensor:
    # 기존 필드 (변경 없음)
    shape: tuple[str | int, ...]
    dtype: torch.dtype
    interface: str
    direction: Direction | None
    name: str
    data: torch.Tensor | None
    _resolved_shape: tuple[int, ...] | None
    _address: int | None

    # NEW: device 상태
    _bo: Any = None                         # xrt.bo (FPGA device memory)
    _bo_size: int = 0                       # serialized byte count
    _deserialize_fn: Callable | None = None # bytes → torch.Tensor (unlayout 포함)
```

### 3.2 API

```python
@property
def on_device(self) -> bool:
    """데이터가 FPGA device에 있는가."""
    return self._bo is not None

def cpu(self) -> torch.Tensor:
    """Device → host 전송. unlayout 자동 적용.

    on_device=False면 host data를 그대로 반환.
    둘 다 없으면 RuntimeError.
    """
    if self._bo is None:
        if self.data is not None:
            return self.data
        raise RuntimeError("no data on host or device")
    self._bo.sync(FROM_DEVICE)
    raw = bytes(self._bo.read(self._bo_size))
    if self._deserialize_fn:
        return self._deserialize_fn(raw)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)

def numpy(self) -> np.ndarray:
    """Convenience: .cpu().numpy()."""
    return self.cpu().numpy()

def _bind_bo(self, bo: Any, size: int, deserialize_fn: Callable | None = None) -> None:
    """InferenceSession 내부 호출. BO를 바인딩한다."""
    self._bo = bo
    self._bo_size = size
    self._deserialize_fn = deserialize_fn
```

### 3.3 라이프사이클

| 상태 | `data` | `_bo` | `on_device` | 생성 시점 |
|------|--------|-------|-------------|----------|
| 선언 | None | None | False | class body |
| host 데이터 | torch.Tensor | None | False | generate_inputs() / 사용자 할당 |
| device 데이터 | None | xrt.bo | True | session.run() output / session.upload() |
| host 전송 후 | torch.Tensor | xrt.bo | True | .cpu() 호출 후 |

`session.cleanup()` 호출 시 BO가 해제되어 `_bo`가 invalid 상태가 된다.
이는 CUDA device가 없을 때 GPU tensor 접근이 실패하는 것과 동일한 패턴.

---

## 4. InferenceSession

### 4.1 생성

```python
from vten import InferenceSession

backend = XrtBackend(project_config, persistent=True)
session = InferenceSession(backend, base_params={"Ti": 32, "To": 32})
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `backend` | `XrtBackend` | persistent=True 권장 |
| `base_params` | `dict` | 모든 run()에 공통 적용되는 파라미터 |

생성 시 `backend._persistent = True`가 자동 설정된다.

### 4.2 upload() — 정적 데이터 업로드

```python
w_dev = session.upload(
    data=weight_tensor,          # torch.Tensor (logical)
    tensor_name="wgt_mem",       # kernel의 텐서 이름
    kernel_class=NpuKernel,      # layout 메서드 참조용
    params={"in_ch": 64, ...},   # shape resolution용
) -> Tensor
```

**동작:**
1. kernel_class를 임시 인스턴스화하여 layout 메서드 및 packing 정보 확보
2. `layout_wgt_mem(data)` 호출 (있으면) → physical data
3. serialize → bytes
4. `xrt.bo` 생성, write, sync TO device
5. `Tensor(on_device=True)` 반환

**용도:** weight, bias 등 batch 간 변하지 않는 데이터를 1회 업로드.

### 4.3 run() — 단일 커널 실행 (핵심 primitive)

```python
result = session.run(
    kernel_class=NpuKernel,
    inputs={
        "ifm_mem": x,              # torch.Tensor 또는 Tensor(on_device)
        "wgt_mem": w_dev,           # upload()으로 올린 Tensor
        "bias_mem": b_dev,
    },
    # **params: 커널 파라미터 (base_params와 merge)
    in_ch=64, out_ch=128, in_depth=16, in_height=32, in_width=32,
    kernel_size=3, ifm_stride=1, ofm_stride=1,
) -> dict[str, Tensor]
```

**동작:**

```
1. params = {**base_params, **params}
2. ctx = ExecutionContext(backend, project_params=params, mode="inference")
3. ki = ctx.instantiate(kernel_class, **params)
4. 입력 바인딩:
   - torch.Tensor → ki.get_tensor(name).logical_data = data (layout 자동)
   - Tensor(on_device) → ctx.bind_device_buffer(tensor, data._bo) (skip 등록)
5. ki.run(ctx)  ← 기존 DSL sequence 그대로 실행
   - ctx.push_tensor() → bound면 NoOpHandle, 아니면 LOAD+PUSH op 기록
   - ctx.configure() → WRITE_REG ops 기록
   - ctx.pull_tensor() → PULL op만 기록 (STORE 안 함)
6. result = ctx.run()  → compile (Stages 0-6) + execute (XrtBackend)
7. output을 Tensor(on_device=True)로 wrap하여 반환
```

**반환값:** `dict[str, Tensor]` — 각 D2H 텐서가 `on_device=True` 상태.

### 4.4 run_pipeline() — Sequential Chain Convenience

```python
result = session.run_pipeline(
    kernel_class=NpuKernel,
    layers=[layer1_params, layer2_params, layer3_params],
    inputs={"ifm_mem": input_tensor},
    per_layer_inputs=[
        {"wgt_mem": w1_dev, "bias_mem": b1_dev},
        {"wgt_mem": w2_dev, "bias_mem": b2_dev},
        {"wgt_mem": w3_dev, "bias_mem": b3_dev},
    ],
    chain={"ofm_mem": "ifm_mem"},   # default
) -> dict[str, Tensor]
```

**내부 구현 — run() 반복:**

```python
def run_pipeline(self, kernel_class, layers, inputs, per_layer_inputs, chain=None):
    chain = chain or {"ofm_mem": "ifm_mem"}
    current = dict(inputs)
    result = None
    for i, params in enumerate(layers):
        merged = {**current, **per_layer_inputs[i]}
        result = self.run(kernel_class, inputs=merged, **params)
        current = {dst: result[src] for src, dst in chain.items() if src in result}
    return result
```

`run_pipeline`은 단순 sequential topology에 대한 sugar.
skip connection, concat 등 비선형 topology는 `run()`을 직접 사용.

### 4.5 cleanup()

```python
session.cleanup()
```

`backend.cleanup()` 호출. 모든 BO 해제. 이후 upload된 Tensor의 `.cpu()` 호출 시 에러.

---

## 5. InferenceModule (nn.Module)

### 5.1 정의

```python
from vten import InferenceModule

class NPUConv3D(InferenceModule):
    kernel_cls = NpuPipelineKernel
    input_name = "ifm_mem"       # forward(x)의 x가 매핑되는 텐서
    output_name = "ofm_mem"      # forward() 반환 텐서
```

### 5.2 생성

```python
conv = NPUConv3D(
    session,                     # InferenceSession
    weight=weight_tensor,        # torch.Tensor → 자동 upload
    bias=bias_tensor,
    # **params: 커널 파라미터
    in_ch=64, out_ch=128, kernel_size=3,
)
```

weight, bias는 생성 시 `session.upload()`으로 device에 올려 `Tensor(on_device)`로 보관.

### 5.3 forward()

```python
def forward(self, x: torch.Tensor | Tensor) -> Tensor:
```

**반환: `Tensor(on_device=True)`.** host 전송은 사용자가 `.cpu()` 호출 시에만 발생.

```python
conv1 = NPUConv3D(session, weight=w1, bias=b1, **L1)
conv2 = NPUConv3D(session, weight=w2, bias=b2, **L2)

y_device = conv2(conv1(x))   # 중간에 host 전송 없음
y_host = y_device.cpu()       # 최종 결과만 host로
```

### 5.4 nn.Sequential 호환

```python
model = nn.Sequential(conv1, conv2, conv3)
y = model(x).cpu()   # 각 layer 사이에 host 전송 없음
```

layer 사이에 CPU 연산을 끼우려면 `.cpu()` + 일반 연산 + 다음 layer input:

```python
y1_device = conv1(x)
y1_host = y1_device.cpu()       # device → host
y1_relu = torch.relu(y1_host)   # CPU 연산
y2_device = conv2(y1_relu)      # host → device (자동 layout + upload)
```

---

## 6. Backend: Persistent Mode

### 6.1 XrtBackend 변경

```python
class XrtBackend(Backend):
    def __init__(self, project_config: dict, persistent: bool = False):
        ...
        self._persistent = persistent
        self._interpreter: CommandInterpreter | None = None
```

| 모드 | interpreter 수명 | BO 수명 | 용도 |
|------|-----------------|---------|------|
| `persistent=False` (기본) | execute() 단위 | execute() 단위 | verification |
| `persistent=True` | backend 수명 | backend 수명 | inference |

### 6.2 execute() 변경

```python
def execute(self, compiled: CompiledResult) -> BackendResult:
    if self._device is None:
        self._init_device()

    ip_map = self._build_ip_map(compiled)
    mem_bank_map = self._build_mem_bank_map(compiled)
    addr_bindings = self._build_addr_bindings(compiled)

    if self._persistent and self._interpreter is not None:
        interpreter = self._interpreter
        interpreter.update_maps(ip_map, mem_bank_map, addr_bindings)
    else:
        interpreter = CommandInterpreter(
            device=self._device, kernel=self._default_ip,
            xrt_module=self._xrt, poll_timeout_ms=self._poll_timeout_ms,
            ip_map=ip_map, mem_bank_map=mem_bank_map,
            addr_bindings=addr_bindings,
        )
        self._interpreter = interpreter

    # Prebound BO 주입 (DeviceTensor용)
    for buffer_id, bo in compiled.prebound_buffers.items():
        interpreter._buffers[buffer_id] = bo
        interpreter._prebound.add(buffer_id)

    interpreter.execute(compiled.commands, compiled.tensor_data)
    ...
```

### 6.3 CompiledResult 확장

```python
@dataclass
class CompiledResult:
    ...
    prebound_buffers: dict[int, Any] = field(default_factory=dict)
    # buffer_id → xrt.bo for Tensor(on_device) inputs
```

---

## 7. CommandInterpreter: BO Reuse

### 7.1 _exec_load — BO 재사용 + skip

```python
def _exec_load(self, cmd, tensor_data):
    data = tensor_data.get(cmd.buffer_id, b"")
    if not data:
        if cmd.buffer_id in self._buffers:
            return  # BO already on device → skip
        raise BackendError(...)

    # persistent mode: 기존 BO 크기 충분하면 재사용
    existing = self._buffers.get(cmd.buffer_id)
    if existing is not None and existing.size() >= len(data):
        existing.write(data)  # overwrite, realloc 안 함
    else:
        bo = self._xrt.bo(self._device, len(data), ...)
        bo.write(data)
        self._buffers[cmd.buffer_id] = bo
```

### 7.2 _exec_push — prebound skip

```python
def _exec_push(self, cmd):
    bo = self._buffers.get(cmd.buffer_id)
    if bo is None:
        raise BackendError(...)
    if cmd.buffer_id in self._prebound:
        return  # 이미 device에 synced → skip
    bo.sync(TO_DEVICE)
```

### 7.3 update_maps()

persistent mode에서 interpreter 재사용 시 매 execute마다 map을 갱신한다.

```python
def update_maps(self, ip_map, mem_bank_map, addr_bindings):
    self._ip_map = ip_map
    self._mem_bank_map = mem_bank_map
    self._addr_bindings = addr_bindings
    self._prebound.clear()  # 새 실행 시작 시 prebound 초기화
    self._output_buffers.clear()
    self._completed.clear()
```

---

## 8. ExecutionContext: Inference Mode

### 8.1 mode 파라미터

```python
ctx = ExecutionContext(
    backend=backend,
    project_params=params,
    mode="inference",           # NEW: "verification" (기본) | "inference"
)
```

### 8.2 동작 변경

**run(verify=True) — inference mode에서 no-op:**

```python
def run(self, verify: bool = False):
    # ... compile + execute ...
    if verify and self._mode != "inference":
        self._run_auto_verification(compiled, result)
    # inference mode에서는 verify=True여도 무시
```

**bind_device_buffer() — NEW:**

```python
def bind_device_buffer(self, tensor: Tensor, bo: Any) -> None:
    """Tensor(on_device)의 BO를 바인딩.
    이후 push_tensor() 호출 시 자동 skip."""
    self._bound_bos[tensor.name] = bo
```

**push_tensor() — bound면 skip:**

```python
def push_tensor(self, tensor, dep=None, probe=False):
    if self._mode == "inference" and tensor.name in self._bound_bos:
        return NoOpHandle()  # BO 이미 device에 → 아무것도 안 함
    # 기존 로직: LOAD + PUSH op 기록
```

**pull_tensor() — inference mode에서 PULL only:**

```python
def pull_tensor(self, tensor, dep=None, probe=False, chunks=None):
    if self._mode == "inference":
        # PULL만, STORE 안 함 (데이터는 device에 남김)
        return self._record(OpKind.PULL_TENSOR, tensor=tensor, dep=dep,
                            probe=probe, _skip_store=True)
    # 기존 로직: PULL + STORE op 기록
```

### 8.3 NoOpHandle

```python
class NoOpHandle:
    """bound된 push_tensor()가 반환하는 placeholder.
    dep= 체인에서 무시된다."""
    @property
    def op_index(self) -> int:
        return -1
```

---

## 9. Examples

### 9.1 단일 커널 실행

```python
from vten import InferenceSession
from vten.backend.xrt import XrtBackend

backend = XrtBackend(config, persistent=True)
session = InferenceSession(backend)

result = session.run(
    PassthroughKernel,
    inputs={"data_in": torch.randn(1024, dtype=torch.int8)},
    N=1024,
)
y = result["data_out"].cpu()
```

### 9.2 Multi-layer Sequential (host_top.py 대체)

```python
session = InferenceSession(backend, base_params={"Ti": 32, "To": 32})

# Weight/bias 1회 upload
w_devs = [session.upload(w, "wgt_mem", NpuKernel, p) for w, p in zip(weights, layers)]
b_devs = [session.upload(b, "bias_mem", NpuKernel, p) for b, p in zip(biases, layers)]

# 추론 루프 — weight sync 0회
for image in dataloader:
    result = session.run_pipeline(
        NpuKernel,
        layers=layer_params,
        inputs={"ifm_mem": image},
        per_layer_inputs=[
            {"wgt_mem": wd, "bias_mem": bd}
            for wd, bd in zip(w_devs, b_devs)
        ],
    )
    y = result["ofm_mem"].cpu()
```

### 9.3 3D U-Net (skip connection + concat)

```python
# Encoder
e1 = session.run(NpuKernel, inputs={"ifm_mem": x, "wgt_mem": w1, "bias_mem": b1}, **enc1)
e2 = session.run(NpuKernel, inputs={"ifm_mem": e1["ofm_mem"], "wgt_mem": w2, "bias_mem": b2}, **enc2)
e3 = session.run(NpuKernel, inputs={"ifm_mem": e2["ofm_mem"], "wgt_mem": w3, "bias_mem": b3}, **enc3)

# Bottleneck
bn = session.run(NpuKernel, inputs={"ifm_mem": e3["ofm_mem"], "wgt_mem": wb, "bias_mem": bb}, **bn_p)

# Decoder — skip connection은 Python 변수로 자연스럽게 표현
d3 = session.run(NpuKernel, inputs={
    "ifm_mem": bn["ofm_mem"],
    "concat_mem": e3["ofm_mem"],    # skip connection!
    "wgt_mem": wd3, "bias_mem": bd3,
}, is_concat=1, **dec3)

d2 = session.run(NpuKernel, inputs={
    "ifm_mem": d3["ofm_mem"],
    "concat_mem": e2["ofm_mem"],    # skip connection
    "wgt_mem": wd2, "bias_mem": bd2,
}, is_concat=1, **dec2)

d1 = session.run(NpuKernel, inputs={
    "ifm_mem": d2["ofm_mem"],
    "concat_mem": e1["ofm_mem"],
    "wgt_mem": wd1, "bias_mem": bd1,
}, is_concat=1, **dec1)

y = d1["ofm_mem"].cpu()
```

### 9.4 nn.Module

```python
from vten import InferenceModule

class NPUConv3D(InferenceModule):
    kernel_cls = NpuPipelineKernel
    input_name = "ifm_mem"
    output_name = "ofm_mem"

conv1 = NPUConv3D(session, weight=w1, bias=b1, **L1)
conv2 = NPUConv3D(session, weight=w2, bias=b2, **L2)

y = conv2(conv1(x)).cpu()
```

### 9.5 디버깅 — 중간 결과 확인

```python
r1 = session.run(NpuKernel, inputs={...}, **L1)

# 아무 중간 지점에서 host로 가져와 확인
debug = r1["ofm_mem"].cpu()
print(f"Layer 1 output: shape={debug.shape}, range=[{debug.min()}, {debug.max()}]")

r2 = session.run(NpuKernel, inputs={"ifm_mem": r1["ofm_mem"], ...}, **L2)
```

---

## 10. API Cleanup

### 10.1 삭제

| API | 파일 | 이유 |
|-----|------|------|
| `run_kernel()` | `vten/functional.py` | NPU_3D에서 미사용. `session.run()`이 상위 호환 |
| `KernelExecutor` | `vten/functional.py` | NPU_3D에서 미사용. 자체 DSL 생성은 복잡한 커널에 부적합 |
| `functional.py` 전체 | `vten/functional.py` | 위 두 API만 포함. 파일 삭제 |
| `BatchResult` | `vten/runtime/context.py` | `ExecutionResult`로 통일 |
| `make_tests()` | `vten/testing.py` | `TestScenario.configs`로 대체됨 |
| `config_table()` | `vten/testing.py` | 불필요한 wrapper |

### 10.2 09_user_api.md 수정

§4 "Level 2: Functional API" 섹션을 본 스펙(11_inference_api.md)으로 교체.

API 레벨 구조 변경:

```
Before:
  Level 0: Kernel Definition
  Level 1: DSL (ExecutionContext)
  Level 2: Functional API (run_kernel, KernelExecutor)

After:
  Level 0: Kernel Definition        ← 변경 없음
  Level 1: DSL (ExecutionContext)    ← 변경 없음 (+ mode="inference")
  Level 2: Inference (InferenceSession)  ← NEW
  Level 3: nn.Module (InferenceModule)   ← NEW
```

### 10.3 유지하는 API

| API | 이유 |
|-----|------|
| `TestScenario` | verification 워크플로우의 핵심 |
| `ctx.poll_register()` | XRT inference에 필요 (LAYER_DONE 폴링) |
| `ctx.read_register()` | 디버깅용 |
| `ctx.alias()` | 같은 ctx 내 multi-kernel에 여전히 유용 |
| STORE IR command | pull_tensor 내부에서 자동 생성 |

---

## 11. Implementation Notes

### 11.1 host_top.py 대응표

| host_top.py (~1100줄) | vTen inference | 메커니즘 |
|---|---|---|
| `KernelContext` (C++ device+IPs+BOs) | `XrtBackend(persistent=True)` | Python-only |
| `init_wgt_bias()` + 32-bank copy | `session.upload()` | Tensor(on_device) |
| `transform_ifm/wgt()` | `Kernel.layout_*()` | Kernel v2 메서드 |
| `inverse_transform_ofm()` | `Tensor.cpu()` → `unlayout_*()` | 자동 |
| `write_*_register_map()` ×6 | `kernel_spec.yaml` auto_bind | 선언적 |
| `poll LAYER_DONE` | `ctx.poll_register()` in `run()` | DSL |
| `LayerConfig` BO routing | Python 변수 (eager) | `r1["ofm"] → r2 input` |
| `fmap_bo[8]` max-size alloc | persistent BO pool | `_exec_load` 재사용 |
| concat handling | `inputs={"concat_mem": e["ofm"]}` | Python |

### 11.2 Kernel v2 layout 의존

`session.upload()`과 `session.run()`에서 `logical_data` 할당 →
Stage 3에서 `layout_*()` 자동 호출이 필요하다.

Kernel v2 layout이 아직 미구현인 경우, `tensor.data` 직접 할당으로 우회 가능:
사용자가 미리 layout을 적용한 physical 데이터를 넘기면 된다.

### 11.3 compile 캐싱 (향후 최적화)

같은 `(kernel_class, frozen_params)`로 반복 run() 시 CompiledResult를 캐싱하면
compile 오버헤드를 제거할 수 있다. 초기 구현에서는 매번 compile.

### 11.4 파일 변경 요약

| 파일 | 변경 | 줄수 |
|------|------|------|
| `vten/kernel/tensor.py` | MODIFY | +20 |
| `vten/runtime/context.py` | MODIFY | +25 |
| `vten/runtime/engine.py` | MODIFY | +2 |
| `vten/runtime/interpreter.py` | MODIFY | +10 |
| `vten/backend/xrt.py` | MODIFY | +15 |
| `vten/inference.py` | **NEW** | ~200 |
| `vten/__init__.py` | MODIFY | export 정리 |
| `vten/functional.py` | **DELETE** | -220 |
| `vten/testing.py` | MODIFY | -50 |
