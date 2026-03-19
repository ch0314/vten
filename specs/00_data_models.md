# vTen Shared Data Models

**Version 0.5.0 — March 2026**
**Role: 모든 스펙 파일의 공유 타입 정의 (Single Source of Truth)**

---

## Table of Contents

1. [Core Enumerations](#1-core-enumerations)
2. [Terminology Hierarchy](#2-terminology-hierarchy)
3. [Tensor Class](#3-tensor-class)
4. [Register Handle](#4-register-handle)
5. [Kernel Base Classes](#5-kernel-base-classes)
6. [KernelSpec Model (YAML Parse Result)](#6-kernelspec-model)
7. [FlattenedKernelView & Related Structures](#7-flattenedkernelview--related-structures)
8. [Operation & OperationHandle (Record Phase)](#8-operation--operationhandle)
9. [Execution IR: Command](#9-execution-ir-command)
10. [Binding Table](#10-binding-table)
11. [SHM Constants & Structures](#11-shm-constants--structures)
12. [Error Taxonomy](#12-error-taxonomy)
13. [Compiled Result & Execution Types](#13-compiled-result--execution-types)
14. [TestScenario & Test Discovery](#14-testscenario--test-discovery)

---

## 1. Core Enumerations

### 1.1 Protocol

```python
from enum import Enum

class Protocol(Enum):
    AXI4S = "axi4_stream"    # AXI4-Stream
    AXI4  = "axi4"           # AXI4 Memory-Mapped
    AXI4L = "axi4_lite"      # AXI4-Lite (registers)
```

SHM 바이너리 인코딩 값: `AXI4S=1, AXI4=2, AXI4L=3`

### 1.2 Role

```python
class Role(Enum):
    MASTER = "master"   # BFM이 버스 마스터 (데이터 구동)
    SLAVE  = "slave"    # BFM이 버스 슬레이브 (DUT 요청에 응답)
```

SHM 인코딩: `MASTER=0, SLAVE=1`

### 1.3 Direction

```python
class Direction(Enum):
    HOST_TO_DEV   = "host_to_dev"     # 호스트→장치 (입력 텐서)
    DEV_TO_HOST   = "dev_to_host"     # 장치→호스트 (출력 텐서)
    BIDIRECTIONAL = "bidirectional"    # 양방향 (드문 사용)
```

SHM 인코딩: `HOST_TO_DEV=0, DEV_TO_HOST=1, BIDIRECTIONAL=2`

### 1.4 OpCode (IR Command)

```python
class OpCode(Enum):
    LOAD      = 1    # Host → Data Region (Runtime이 SHM 제출 전 처리)
    PUSH      = 2    # BFM이 텐서 데이터를 DUT에 제공
    PULL      = 3    # BFM이 DUT에서 텐서 데이터 캡처
    STORE     = 4    # Data Region → Host (Backend DONE 후 Host가 읽음)
    WRITE_REG = 5    # AXI-Lite로 레지스터 쓰기
    READ_REG  = 6    # AXI-Lite로 레지스터 읽기 (결과 reg_value에 기록)
    POLL_REG  = 7    # (value & mask) == expected 될 때까지 반복 읽기
    BARRIER   = 8    # 전역 동기화 펜스 — 이전 모든 커맨드 commit 후 다음 issue
    COMPARE   = 9    # 버퍼와 golden_buffer의 비트별 비교 (probe 모드)
```

### 1.5 OpKind (Record Phase)

```python
class OpKind(Enum):
    LOAD_TENSOR     = "load_tensor"
    STORE_TENSOR    = "store_tensor"
    PUSH_TENSOR     = "push_tensor"
    PULL_TENSOR     = "pull_tensor"
    WRITE_REGISTER  = "write_register"
    READ_REGISTER   = "read_register"
    POLL_REGISTER   = "poll_register"
    CONFIGURE       = "configure"
    BARRIER         = "barrier"
    SEND_TENSOR     = "send_tensor"     # = load + push (자동)
    RECV_TENSOR     = "recv_tensor"     # = pull + store (자동)
```

### 1.6 MappingType (CompositeKernel 인터페이스 매핑)

```python
class MappingType(Enum):
    EXTERNAL       = "external"         # 외부 인터페이스 → BFM 생성
    EXTERNAL_BANK  = "external_bank"    # 외부 + 레지스터 뱅크 오프셋
    INTERNAL       = "internal"         # RTL 내부 와이어 → BFM 없음
    INTERNAL_PROBE = "internal_probe"   # 내부 + 패시브 모니터 부착
```

### 1.7 CommandStatus (Stats Region)

```python
class CommandStatus(Enum):
    PENDING   = 0   # 초기 상태
    ISSUED    = 1   # BFM에 디스패치됨
    ACTIVE    = 2   # 데이터 전송 진행 중
    COMMITTED = 3   # 완료 (의존성 그래프에서 해제)
    ERROR     = 4   # 에러 발생
```

---

## 2. Terminology Hierarchy

vTen 실행 모델의 4단계 용어 계층. 모든 스펙 문서가 이 정의를 따른다.

### 2.1 Definitions

| Term | Scope | Definition |
|------|-------|------------|
| **Command** | Lowest | 하나의 IR 명령. 64바이트 SHM 슬롯 1개를 차지한다. BFM dispatch 또는 내부 스케줄러 액션과 1:1 대응. 예: PUSH 1개, WRITE_REG 1개. |
| **Operation** | DSL | DSL 메서드 호출 1회. IR lowering 시 여러 Command로 확장될 수 있다. 예: `send_tensor()` → LOAD + PUSH (2 commands). `configure()` → N × WRITE_REG. |
| **Invocation** | Execution cycle | 하나의 가속기 실행 사이클: configure → 입력 전달 → 연산 → 출력 수집. 특정 파라미터 세트로 DUT를 한 번 "사용"하는 것에 해당. 하나의 Invocation은 보통 5–15개 Operation으로 구성. |
| **Batch** | Host↔Backend | 하나의 SHM 제출 단위: 두 `ctx.run()` 호출 사이의 모든 Command (또는 시작부터 첫 `ctx.run()`까지). Batch 당 정확히 하나의 Host↔Backend 세마포어 핸드셰이크. 하나의 Batch에 하나 이상의 Invocation을 포함할 수 있다. §11.4, §11.5에서는 **Kernel Task**를 동의어로 사용 (후방호환). |

### 2.2 Hierarchy Diagram

```
Batch (1 Host↔Backend handshake)
├── Invocation 0 (accelerator run with config A)
│   ├── Operation: send_tensor(ifm)        → Commands: LOAD, PUSH
│   ├── Operation: send_tensor(weight)     → Commands: LOAD, PUSH
│   ├── Operation: configure(kernel)       → Commands: WRITE_REG ×N
│   ├── Operation: write_register(start=1) → Command:  WRITE_REG
│   ├── Operation: recv_tensor(ofm)        → Command:  PULL  (aliased: STORE skipped)
│   └── Operation: poll_register(done)     → Command:  POLL_REG
├── Invocation 1 (accelerator run with config B)
│   ├── Operation: send_tensor(ifm)        → Command:  PUSH  (aliased: LOAD skipped)
│   ├── Operation: send_tensor(weight)     → Commands: LOAD, PUSH
│   │   ...
│   └── Operation: poll_register(done)     → Command:  POLL_REG
└── ...more Invocations...
```

### 2.3 Usage Rules

- **"layer"**는 vTen 용어가 아니다. 신경망 도메인에 속한다. 테스트 코드 주석에서 비공식적으로 사용할 수 있지만, DSL API에서는 사용하지 않는다. 올바른 vTen 용어는 **Invocation**.
- **"Kernel Task"**는 Backend/SHM 프로토콜 맥락(§11.4, §11.5)에서 Batch의 동의어로 유효하다. 신규 스펙 텍스트에서는 "Batch"를 선호한다.
- **"operation"** (소문자)는 DSL 수준 액션을 지칭한다. **"Command"**는 IR/SHM 수준 명령을 지칭한다. 이 둘을 혼용하지 않는다.

---

## 3. Tensor Class

Kernel 클래스 내에서 텐서를 선언하고, 런타임 과정에서 데이터와 메타데이터를 누적하는 핵심 타입.

### 3.1 Lifecycle & Timing

텐서의 상태는 두 단계에서 설정된다:

```
[1] instantiate() 시점 (eager resolution):
    - 파라미터 해결 → _resolved_shape 설정
    - _element_count 계산
    → generate_inputs()와 forward()에서 즉시 사용 가능

[2] compile() 시점 (재검증 + 직렬화):
    - Stage 1-2: _resolved_shape 재검증 (instantiate 결과와 일치 확인)
    - Stage 3: 직렬화 → _serialized 바이트
    - Stage 4: 주소 할당 → _address
```

이 설계의 근거:
- `generate_inputs()`와 `forward()`는 `compile()` **이전**에 호출됨
- 사용자는 `self.ifm.fill_random()`이나 `torch.randn(self.N, self.C, ...)` 등으로
  해결된 형상/파라미터에 접근해야 함
- `instantiate()`에서 runtime_params + project_params + kernel_spec params가
  모두 알려져 있으므로 eager resolution 가능

### 3.2 Class Definition

```python
import torch
import math

class Tensor:
    """커널의 텐서 선언. 선언 시에는 shape/dtype/interface만 설정하고,
    instantiate() 시점에 파라미터가 해결되면 _resolved_shape가 채워진다."""

    # ── 선언 시 설정 (Kernel 클래스 본문) ──

    def __init__(self, shape: tuple, dtype: torch.dtype, interface: str,
                 direction: Direction | None = None):
        """
        Args:
            shape: 텐서 형상. 문자열 차원("${C}")은 파라미터 해결 후 정수로 변환.
                   예: ("${N}", "${C}", "${D}", "${H}", "${W}")
            dtype: 데이터 타입. torch.int8, torch.int32, torch.float32 등.
            interface: 이 텐서가 바인딩되는 kernel_spec.yaml의 인터페이스 이름.
                       예: "ifm_stream", "data_port"
            direction: 데이터 전송 방향 (Optional).
                       AXI4-Stream: 생략 가능 — protocol/role에서 자동 추론.
                       AXI4 (MM): 같은 포트에 읽기/쓰기 텐서가 혼재할 수 있으므로
                                  명시 권장. 생략 시 HOST_TO_DEV로 기본값.
        """
        self.shape = shape          # 원본 (미해결) 형상 — 문자열 포함 가능
        self.dtype = dtype
        self.interface = interface
        self.direction = direction  # None이면 Stage 0에서 protocol/role로 추론
        self.name: str = ""         # Kernel 메타클래스에 의해 속성명으로 자동 설정

        # ── instantiate() 시점에 설정 (eager resolution) ──
        self._resolved_shape: tuple[int, ...] | None = None
        self._element_count: int = 0

        # ── compile() 시점에 설정 ──
        self.data: torch.Tensor | None = None          # 실제 텐서 데이터
        self._address: int | None = None               # Stage 4 후

    def _resolve_shape(self, resolver: 'ParameterResolver'):
        """instantiate() 시점에 호출. 파라미터를 해결하여 형상을 확정."""
        self._resolved_shape = tuple(
            resolver.resolve(dim) for dim in self.shape
        )
        self._element_count = math.prod(self._resolved_shape)

    # ── 데이터 조작 메서드 ──

    def fill_random(self, generator: torch.Generator | None = None):
        """dtype에 맞는 랜덤 데이터로 채움.
        instantiate() 후 호출 가능 (_resolved_shape가 설정된 상태)."""
        if self._resolved_shape is None:
            raise RuntimeError(
                f"Tensor '{self.name}': shape not resolved yet. "
                f"Ensure instantiate() was called before generate_inputs()."
            )
        if self.dtype in (torch.int8, torch.uint8):
            self.data = torch.randint(
                -128, 127, self._resolved_shape,
                dtype=self.dtype, generator=generator
            )
        elif self.dtype == torch.int32:
            self.data = torch.randint(
                -2**15, 2**15, self._resolved_shape,
                dtype=self.dtype, generator=generator
            )
        elif self.dtype in (torch.float32, torch.float16):
            self.data = torch.randn(
                self._resolved_shape, dtype=self.dtype, generator=generator
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def to_float(self) -> torch.Tensor:
        """연산을 위해 float32로 변환 (golden reference 계산용)."""
        if self.data is None:
            raise RuntimeError(f"Tensor '{self.name}': no data")
        return self.data.float()

    def set_address(self, addr: int):
        """사용자가 명시적으로 물리 주소를 강제 설정."""
        self._address = addr

    def numel(self) -> int:
        """해결된 형상의 총 원소 수."""
        if self._resolved_shape is None:
            raise RuntimeError(f"Tensor '{self.name}': shape not resolved")
        return math.prod(self._resolved_shape)
```

**Kernel 디스크립터 등록:** Kernel 클래스 정의 시 `ifm = Tensor(...)` 형태로 선언하면, `__init_subclass__`가 `ifm.name = "ifm"`을 자동 설정한다. 상세 구현은 §2.3 참조.

**주의: Tensor는 클래스 변수이다.** 하나의 Kernel 클래스에서 여러 인스턴스를 만들 때 텐서 상태(`data`, `_resolved_shape` 등)가 공유되는 문제를 방지하려면, `instantiate()` 시점에 각 인스턴스별로 텐서를 **복제(`copy.copy`)**해야 한다. 이는 KernelInstance에서 처리한다 (§6.4 참조).

### 3.3 Kernel Descriptor Registration (구현 명세)

**방식: `__init_subclass__` 기반** (메타클래스보다 단순하며, Python 3.6+ 표준)

```python
class Kernel:
    """Unit Kernel 베이스 클래스."""
    spec: str = ""
    
    # 프레임워크가 관리하는 클래스 레벨 레지스트리
    _tensor_descriptors: dict[str, Tensor] = {}
    _register_handles: dict[str, RegisterHandle] = {}

    def __init_subclass__(cls, **kwargs):
        """Kernel 서브클래스 정의 시 자동 호출.
        클래스 본문의 Tensor와 RegisterHandle을 자동 등록한다.
        
        호출 시점: 클래스 본문 평가 직후 (인스턴스 생성 전).
        
        수행 작업:
        1. 클래스 네임스페이스에서 Tensor 인스턴스 탐색
        2. 각 Tensor에 속성명 설정 (tensor.name = attr_name)
        3. _tensor_descriptors에 등록
        4. RegisterHandle도 동일하게 처리
        """
        super().__init_subclass__(**kwargs)
        
        # 부모의 레지스트리를 상속하지 않도록 새로 생성
        cls._tensor_descriptors = {}
        cls._register_handles = {}
        
        for attr_name, attr_value in vars(cls).items():
            if isinstance(attr_value, Tensor):
                attr_value.name = attr_name
                cls._tensor_descriptors[attr_name] = attr_value
            elif isinstance(attr_value, RegisterHandle):
                cls._register_handles[attr_name] = attr_value
    
    def tensors(self) -> list[Tensor]:
        """이 커널(인스턴스)에 등록된 모든 Tensor 반환.
        인스턴스화 전: 클래스 변수(공유).
        인스턴스화 후: KernelInstance.initialize()에서 복제된 인스턴스 변수."""
        return [getattr(self, name) for name in self.__class__._tensor_descriptors]
    
    def get_tensor(self, name: str) -> Tensor:
        if name not in self.__class__._tensor_descriptors:
            raise AttributeError(f"No tensor '{name}' in {self.__class__.__name__}")
        return getattr(self, name)
```

**인스턴스 격리 전략 (KernelInstance.initialize에서 처리):**

```python
# KernelInstance.initialize() 내부 (§6.4 참조):
import copy

self.kernel_class_instance = self.kernel_class()

# 클래스 변수 Tensor를 인스턴스 변수로 복제
for tensor_name in self.kernel_class._tensor_descriptors:
    class_tensor = self.kernel_class._tensor_descriptors[tensor_name]
    instance_tensor = copy.copy(class_tensor)  # shallow copy 충분
    # copy.copy: shape, dtype, interface, direction, name은 공유 (불변)
    # data, _resolved_shape, _element_count, _address는 None으로 초기화됨
    setattr(self.kernel_class_instance, tensor_name, instance_tensor)
    instance_tensor._resolve_shape(self._resolver)
```

**`copy.copy` vs `copy.deepcopy` 결정:**
- `copy.copy` (shallow) 채택: Tensor의 `shape` 튜플, `dtype`, `interface` 문자열은 불변이므로 공유 안전
- `data`(`torch.Tensor | None`)는 `None`으로 시작하고 `generate_inputs()`에서 새로 할당되므로 shallow copy 충분
- `_resolved_shape`는 `_resolve_shape()`에서 새 튜플로 교체되므로 shallow copy 충분

**CompositeKernel에 대한 동일 처리:**

```python
class CompositeKernel(Kernel):
    """복합 커널 베이스 클래스."""
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # 추가: SubKernelBinding과 ExposedTensorDef 수집
        cls._sub_kernel_bindings = {}
        cls._exposed_tensor_defs = {}
        cls._connections = []
        
        for attr_name, attr_value in vars(cls).items():
            if isinstance(attr_value, SubKernelBinding):
                attr_value._attr_name = attr_name
                cls._sub_kernel_bindings[attr_name] = attr_value
            elif isinstance(attr_value, ExposedTensorDef):
                cls._exposed_tensor_defs[attr_name] = attr_value
            elif attr_name == 'connections' and isinstance(attr_value, list):
                cls._connections = attr_value
```

---

## 4. Register Handle

`register()` 헬퍼 함수와 그 반환 타입.

```python
class RegisterHandle:
    """Kernel에서 register("ctrl") 호출 시 반환되는 핸들.
    ExecutionContext의 write_register/poll_register에서 인터페이스 참조로 사용."""

    def __init__(self, interface_name: str):
        self.interface_name = interface_name

    def __repr__(self):
        return f"RegisterHandle('{self.interface_name}')"


def register(interface_name: str) -> RegisterHandle:
    """Kernel 클래스 본문에서 레지스터 인터페이스를 선언하는 헬퍼.

    Usage:
        class Conv3DKernel(Kernel):
            ctrl = register("ctrl")

        # 이후 테스트에서:
        ctx.write_register(kernel.ctrl, {"start": 1})
        ctx.poll_register(kernel.ctrl, "done")
    """
    return RegisterHandle(interface_name)
```

**DSL에서의 사용 패턴:**

- `ctx.write_register(kernel.ctrl, {"start": 1})` → `kernel.ctrl.interface_name` = `"ctrl"` 사용
- `ctx.poll_register(kernel.ctrl, "done")` → `"done"`은 레지스터의 `fields` 내 필드명

---

## 5. Kernel Base Classes

### 5.1 Kernel (Unit Kernel)

```python
class Kernel:
    """단일 RTL 모듈에 대한 검증 단위.
    텐서 선언, 입력 생성, golden reference를 하나로 묶는다.
    
    Tensor/RegisterHandle 자동 등록은 __init_subclass__에서 처리 (§2.3 참조).
    인스턴스 격리(텐서 복제)는 KernelInstance.initialize()에서 처리 (§6.4 참조)."""

    spec: str = ""  # kernel_spec.yaml 경로 (클래스 변수로 선언)
    
    # __init_subclass__에 의해 설정됨 (§2.3)
    _tensor_descriptors: dict[str, 'Tensor'] = {}
    _register_handles: dict[str, 'RegisterHandle'] = {}

    def __init_subclass__(cls, **kwargs):
        """서브클래스 정의 시 Tensor/RegisterHandle 자동 등록. 상세: §2.3."""
        super().__init_subclass__(**kwargs)
        cls._tensor_descriptors = {}
        cls._register_handles = {}
        for attr_name, attr_value in vars(cls).items():
            if isinstance(attr_value, Tensor):
                attr_value.name = attr_name
                cls._tensor_descriptors[attr_name] = attr_value
            elif isinstance(attr_value, RegisterHandle):
                cls._register_handles[attr_name] = attr_value

    def generate_inputs(self, seed: int | None = None):
        """입력 텐서 데이터 생성. 사용자 오버라이드.
        타일링, 양자화 등 가속기 특화 로직 포함 가능.

        이 메서드 호출 시점에 파라미터와 형상이 이미 해결되어 있다.
        (instantiate() 시점에 eager resolution 수행됨)
        따라서 self.ifm.fill_random()이나 직접 shape 참조가 모두 가능."""
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Golden reference 계산. 사용자 오버라이드.
        임의의 PyTorch/Python 코드 사용 가능."""
        raise NotImplementedError

    def verify(self, hw_output: torch.Tensor,
               golden: torch.Tensor) -> bool:
        """커스텀 비교 로직 (선택적 오버라이드).
        기본값: torch.allclose(hw_output, golden)"""
        return torch.allclose(hw_output, golden)

    # ── 프레임워크 제공 메서드 ──

    def tensors(self) -> list[Tensor]:
        """이 커널에 등록된 모든 Tensor 반환.
        인스턴스화 전: 클래스 디스크립터 (공유).
        인스턴스화 후: 인스턴스별 복제본 (§6.4)."""
        return [getattr(self, name) for name in self.__class__._tensor_descriptors]

    def get_tensor(self, name: str) -> Tensor:
        """이름으로 Tensor 검색."""
        if name not in self.__class__._tensor_descriptors:
            raise AttributeError(f"No tensor '{name}' in {self.__class__.__name__}")
        return getattr(self, name)

    # ── 서브커널 바인딩 (CompositeKernel에서 사용) ──

    @classmethod
    def bind(cls, interface_map: dict,
             params: dict | None = None) -> 'SubKernelBinding':
        """이 Kernel 클래스를 CompositeKernel의 서브커널로 바인딩.
        CompositeKernel 클래스 본문에서 호출.

        Kernel 베이스 클래스에 정의되는 이유: 바인딩 **대상**이 일반 Kernel
        서브클래스(DMAKernel, MACKernel 등)이므로, 이들이 bind()에 접근
        가능해야 한다. CompositeKernel에 두면 Kernel 서브클래스에서
        MRO상 접근 불가.

        Usage:
            class NPUTopKernel(CompositeKernel):
                dma_ifm = DMAKernel.bind(
                    interface_map={
                        "axi_master": "ddr_port",
                        "stream_out": Internal(),
                        "ctrl_lite": ("ctrl", "dma_ifm"),
                    }
                )

        Returns:
            SubKernelBinding — 메타클래스가 수집하여 CompositeKernel에 등록.
        """
        return SubKernelBinding(
            kernel_class=cls,
            interface_map=interface_map,
            params=params,
        )
```

### 5.2 TensorProxy & expose() Mechanism

`SubKernelBinding`에서 서브커널의 텐서에 접근하고 `expose()`를 호출하는 프록시 체인.
이 코드는 **클래스 본문 평가 시점**(인스턴스 생성 전)에 동작해야 한다.

```python
class TensorProxy:
    """SubKernelBinding.__getattr__이 반환하는 프록시.
    아직 인스턴스화되지 않은 서브커널의 텐서를 참조한다.

    두 가지 용도:
    1. expose(interface=...) → ExposedTensorDef 생성
    2. Connect()의 인자로 사용 → 소스/목적 텐서 식별
    """

    def __init__(self, binding_attr_name: str, tensor_name: str,
                 kernel_class: type):
        """
        Args:
            binding_attr_name: CompositeKernel 클래스 내 바인딩 속성명 (예: "dma_ifm")
            tensor_name: 서브커널의 텐서 이름 (예: "src")
            kernel_class: 서브커널 클래스 (예: DMAKernel)
        """
        self.binding_attr_name = binding_attr_name
        self.tensor_name = tensor_name
        self.kernel_class = kernel_class

    def expose(self, interface: str) -> 'ExposedTensorDef':
        """이 텐서를 CompositeKernel의 최상위 텐서로 노출.

        Usage:
            class NPUTopKernel(CompositeKernel):
                dma_ifm = DMAKernel.bind(interface_map={...})
                ifm = dma_ifm.src.expose(interface="ddr_port")
        """
        return ExposedTensorDef(
            origin_sub_kernel=self.binding_attr_name,
            origin_name=self.tensor_name,
            top_interface=interface,
        )


@dataclass
class ExposedTensorDef:
    """expose() 호출의 결과. 메타클래스가 수집하여 FlattenedKernelView에 등록."""
    origin_sub_kernel: str   # "dma_ifm"
    origin_name: str         # "src"
    top_interface: str       # "ddr_port"
```

### 5.3 SubKernelBinding

```python
@dataclass
class SubKernelBinding:
    """Kernel.bind()의 결과. CompositeKernel 클래스 본문에서 속성으로 저장된다."""
    kernel_class: type           # Kernel 서브클래스 (예: DMAKernel)
    interface_map: dict          # {sub_iface: "top_iface" | ("top", "bank") | Internal()}
    params: dict | None = None   # 파라미터 포워딩 표현식

    # 메타클래스에 의해 설정됨 (CompositeKernel 클래스 정의 시)
    _attr_name: str = ""         # CompositeKernel 내 속성 이름 (예: "dma_ifm")

    def __getattr__(self, name: str) -> TensorProxy:
        """서브커널의 텐서에 프록시로 접근.
        클래스 본문 평가 시점에서 호출됨.

        Usage:
            dma_ifm = DMAKernel.bind(...)
            ifm = dma_ifm.src.expose(interface="ddr_port")
            #     ^^^^^^^^^ __getattr__("src") → TensorProxy

        검증: kernel_class에 해당 이름의 Tensor가 존재하는지 확인.
        """
        # dataclass 내부 필드 접근은 정상 처리 (무한 재귀 방지)
        if name.startswith('_') or name in ('kernel_class', 'interface_map',
                                             'params'):
            raise AttributeError(name)

        # 서브커널 클래스에서 텐서 검색
        attr = getattr(self.kernel_class, name, None)
        if isinstance(attr, Tensor):
            return TensorProxy(
                binding_attr_name=self._attr_name,
                tensor_name=name,
                kernel_class=self.kernel_class,
            )
        raise AttributeError(
            f"Sub-kernel '{self.kernel_class.__name__}' has no tensor '{name}'. "
            f"Available: {[k for k, v in vars(self.kernel_class).items() if isinstance(v, Tensor)]}"
        )
```

**메타클래스의 역할:** CompositeKernel 서브클래스 정의 시, 메타클래스(또는 `__init_subclass__`)가
모든 `SubKernelBinding` 속성을 찾아 `_attr_name`을 설정한다:

```python
# 메타클래스 의사 코드
for attr_name, attr_value in namespace.items():
    if isinstance(attr_value, SubKernelBinding):
        attr_value._attr_name = attr_name  # "dma_ifm", "mac", ...
```

### 5.4 Connect (프록시 기반)

```python
class Connect:
    """CompositeKernel의 connections에서 RTL 내부 와이어를 서술.
    실제 RTL 와이어를 생성하지 않음 — golden 체이닝과 probe 수집용.

    TensorProxy 2개를 받아서 소스/목적의 서브커널 이름과 텐서 이름을 추출한다.
    """

    def __init__(self, source: TensorProxy, dest: TensorProxy,
                 transform: callable | None = None):
        """
        Args:
            source: 소스 텐서 프록시 (예: dma_ifm.dst)
            dest: 목적 텐서 프록시 (예: mac.ifm)
            transform: 중간 변환 함수 (선택)

        Usage:
            connections = [
                Connect(dma_ifm.dst, mac.ifm),
                Connect(mac.ofm, dma_ofm.src, transform=lambda x: quantize(x, 8)),
            ]
        """
        if not isinstance(source, TensorProxy):
            raise TypeError(
                f"Connect source must be TensorProxy, got {type(source).__name__}. "
                f"Use sub_kernel_binding.tensor_name syntax.")
        if not isinstance(dest, TensorProxy):
            raise TypeError(
                f"Connect dest must be TensorProxy, got {type(dest).__name__}.")

        self.source_sub: str = source.binding_attr_name   # "dma_ifm"
        self.source_name: str = source.tensor_name         # "dst"
        self.dest_sub: str = dest.binding_attr_name        # "mac"
        self.dest_name: str = dest.tensor_name             # "ifm"

        # probe key 생성을 위해 소스 텐서의 인터페이스 이름도 추출
        source_tensor = getattr(source.kernel_class, source.tensor_name, None)
        self.source_interface: str | None = (
            source_tensor.interface if isinstance(source_tensor, Tensor) else None
        )

        self.transform = transform
```

### 5.5 Internal

```python
class Internal:
    """interface_map에서 내부 와이어(BFM 없음)를 표시."""
    def __init__(self, probe: bool = False):
        self.probe = probe
```

### 5.6 CompositeKernel

```python
class CompositeKernel(Kernel):
    """여러 서브커널을 조합한 상위 검증 단위.
    
    bind()는 Kernel 베이스 클래스에 정의되어 있다 (§4.1).
    DMAKernel.bind(...)처럼 일반 Kernel 서브클래스에서 호출된다.
    """

    connections: list[Connect] = []

    def bindings(self) -> list[tuple[str, SubKernelBinding]]:
        """선언된 모든 서브커널 바인딩 반환.
        Returns: [(name, SubKernelBinding), ...]"""
        return [(k, v) for k, v in vars(self.__class__).items()
                if isinstance(v, SubKernelBinding)]

    def exposed_tensor_defs(self) -> list[tuple[str, ExposedTensorDef]]:
        """expose()로 노출된 텐서 정의 반환.
        Returns: [(name, ExposedTensorDef), ...]"""
        return [(k, v) for k, v in vars(self.__class__).items()
                if isinstance(v, ExposedTensorDef)]

    # ── Probe support ──

    def __init__(self):
        super().__init__()
        self._probe_data: dict[str, torch.Tensor] = {}

    def probe(self, interface_key: str, data: torch.Tensor):
        """probe golden 데이터 등록.
        interface_key: "sub_kernel_name.interface_name"
        forward_with_intermediates() 내에서 호출."""
        self._probe_data[interface_key] = data.clone()

    def forward_with_intermediates(self) -> dict[str, torch.Tensor]:
        """Probe golden 데이터를 명시적으로 제공하려면 오버라이드.
        self.probe()를 각 Internal(probe=True) 인터페이스에 호출해야 함.
        미구현 시 Runtime이 connections DAG를 자동 순회."""
        raise NotImplementedError
```

### 5.7 전체 프록시 체인 예시

```python
# ── 서브커널 정의 ──
class DMAKernel(Kernel):
    spec = "specs/dma.yaml"
    src = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="axi_master")
    dst = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="stream_out")
    ctrl = register("ctrl_lite")

# ── CompositeKernel 정의 (클래스 본문 평가 시점) ──
class NPUTopKernel(CompositeKernel):
    spec = "specs/npu_top.yaml"

    # Step 1: bind() → SubKernelBinding 생성
    #   DMAKernel.bind()는 Kernel.bind()를 호출 (Issue 1 해결)
    #   메타클래스가 _attr_name = "dma_ifm" 설정
    dma_ifm = DMAKernel.bind(
        interface_map={
            "axi_master": "ddr_port",
            "stream_out": Internal(),
            "ctrl_lite": ("ctrl", "dma_ifm"),
        }
    )

    # Step 2: dma_ifm.src → SubKernelBinding.__getattr__("src")
    #   DMAKernel에서 Tensor "src"를 찾아 TensorProxy 반환
    #   TensorProxy(binding_attr_name="dma_ifm", tensor_name="src", ...)

    # Step 3: .expose(interface="ddr_port") → ExposedTensorDef 생성
    #   ExposedTensorDef(origin_sub_kernel="dma_ifm", origin_name="src",
    #                    top_interface="ddr_port")
    ifm = dma_ifm.src.expose(interface="ddr_port")

    mac = MACKernel.bind(interface_map={...})
    dma_ofm = DMAKernel.bind(interface_map={...})

    weight = dma_weight.src.expose(interface="ddr_port")
    ofm = dma_ofm.dst.expose(interface="ddr_port")

    # Step 4: Connect(dma_ifm.dst, mac.ifm) → TensorProxy 2개 받음
    #   Connect가 내부에서 source_sub="dma_ifm", source_name="dst",
    #   dest_sub="mac", dest_name="ifm"을 추출
    connections = [
        Connect(dma_ifm.dst, mac.ifm),
        Connect(dma_weight.dst, mac.weight),
        Connect(mac.ofm, dma_ofm.src),
    ]
```

---

## 6. KernelSpec Model (YAML Parse Result)

`kernel_spec.yaml`을 파싱한 결과 생성되는 데이터 클래스들. 상세 파싱 규칙과 검증 로직은 `03_kernel_spec_schema.md` 참조.

### 6.1 PackingScheme

```python
@dataclass
class CustomField:
    name: str           # 필드 이름 (예: "data_a")
    bits: tuple[int, int]  # (lo_bit, hi_bit) inclusive

@dataclass
class PackingScheme:
    """인터페이스의 비트 패킹 정의."""
    element_width: int           # 각 원소의 비트 폭
    elements_per_beat: int       # 버스 비트당 원소 수
    bit_order: str = "lsb_first" # "lsb_first" | "msb_first"
    alignment: str = "packed"    # "packed" | "aligned" (바이트 경계)
    byte_order: str = "little"   # "little" | "big"
    mode: str = "standard"       # "standard" | "custom"
    custom_fields: list[CustomField] | None = None

    @property
    def bus_width(self) -> int:
        """패킹된 한 비트의 총 비트 수."""
        if self.mode == "custom" and self.custom_fields:
            return max(f.bits[1] for f in self.custom_fields) + 1
        if self.alignment == "packed":
            return self.element_width * self.elements_per_beat
        else:
            elem_bytes = (self.element_width + 7) // 8
            return elem_bytes * 8 * self.elements_per_beat

    def validate_custom_fields(self) -> None:
        """custom mode일 때 필드 간 비트 겹침 검사.

        파서(spec/parser.py)가 PackingScheme 생성 직후 호출한다.
        겹치는 비트 범위가 있으면 ValidationError를 raise한다.
        allow_overlap 옵션은 현재 미지원 — 명시적 공유가 필요하면 향후 확장.
        """
        if self.mode != "custom" or not self.custom_fields:
            return
        occupied: dict[int, str] = {}  # bit_pos → field_name
        for field in self.custom_fields:
            lo, hi = field.bits
            for bit in range(lo, hi + 1):
                if bit in occupied:
                    raise ValidationError(
                        f"custom_fields overlap: field '{field.name}' bits {field.bits} "
                        f"conflicts with '{occupied[bit]}' at bit {bit}.")
                occupied[bit] = field.name
```

### 6.2 SplitSpec

```python
@dataclass
class PortDef:
    name: str
    base_addr: int

@dataclass
class InterleaveSpec:
    unit: int  # 라운드 로빈 단위 (바이트)

@dataclass
class SplitSpec:
    """멀티포트 분할 사양 (예: HBM 채널)."""
    mode: str                      # "channel_interleave" | "block_split"
    ports: list[PortDef]
    interleave: InterleaveSpec | None = None
```

### 6.3 AutoBindSpec

```python
@dataclass
class AutoBindSpec:
    """레지스터의 auto_bind 사양."""
    tensor: str | None = None    # 참조 텐서 이름
    value: str | None = None     # "address" | "size_bytes" | "size_beats" | "size_elements"
    bits: str | None = None      # "31:0" | "63:32" 등 (address 분할)
    param: str | None = None     # "${C}" 등 파라미터 표현식
    expr: str | None = None      # "${N}*${K}" 등 산술 표현식
```

### 6.4 RegisterSpec

```python
@dataclass
class RegisterField:
    name: str
    bits: str  # "0:0", "1:1" 등

@dataclass
class RegisterSpec:
    """단일 레지스터 정의."""
    name: str
    offset: int
    fields: dict[str, str] | None = None     # {field_name: "hi:lo"}
    auto_bind: AutoBindSpec | None = None
    interface_name: str = ""  # 소속 인터페이스 (파서가 설정)
```

### 6.5 MemoryRegion

```python
@dataclass
class MemoryRegion:
    """메모리 영역 정의."""
    name: str
    base: int
    size: int
    alignment: int = 4096
```

### 6.6 RegisterBankSpec

```python
@dataclass
class RegisterBankSpec:
    """AXI-Lite 인터페이스의 레지스터 뱅크 오프셋."""
    name: str           # 뱅크 이름 (예: "dma_ifm")
    base_offset: int    # 베이스 오프셋 (예: 0x000)
```

### 6.7 InterfaceSpec

```python
@dataclass
class InterfaceSpec:
    """kernel_spec.yaml의 단일 인터페이스 정의."""
    name: str                                # 인터페이스 이름 (YAML 키)
    rtl_port: str                            # RTL 포트 접두사
    protocol: Protocol
    data_width: int | None = None            # AXI4/AXI4S 데이터 폭 (비트)
    addr_width: int | None = None            # AXI4/AXI4L 주소 버스 폭 (비트)
    memory_region: str | None = None         # AXI4: 매핑된 메모리 영역
    tensor: str | None = None                # 단일 텐서 이름 (AXI4S)
    tensors: list[str] | None = None         # 복수 텐서 (AXI4 공유 포트)
    packing: PackingScheme | None = None
    split: SplitSpec | None = None
    registers: list[RegisterSpec] | None = None    # AXI4L
    register_banks: list[RegisterBankSpec] | None = None  # 복수 서브커널 뱅크
```

**`addr_width` 프로토콜별 기본값:**
- AXI4: `addr_width` 미지정 시 **64** (파서가 기본값 적용)
- AXI4-Lite: `addr_width` 미지정 시 **32** (파서가 기본값 적용)
- AXI4-Stream: `addr_width` 해당 없음 (지정 시 무시)

### 6.8 KernelSpec

```python
@dataclass
class KernelSpec:
    """kernel_spec.yaml 파싱 결과. 단일 RTL 모듈의 전체 인터페이스 사양."""
    kernel_name: str
    rtl_top: str
    parameters: dict[str, str | int]         # {"C": "${C}", "H": 32}
    memory_regions: dict[str, MemoryRegion]   # {"ddr": MemoryRegion(...)}
    interfaces: dict[str, InterfaceSpec]

    def get_interface(self, name: str) -> InterfaceSpec:
        """인터페이스 이름으로 조회."""
        if name not in self.interfaces:
            raise KeyError(f"Interface '{name}' not found in {self.kernel_name}")
        return self.interfaces[name]

    def get_registers(self, interface_name: str) -> list[RegisterSpec]:
        """특정 인터페이스의 레지스터 목록."""
        iface = self.get_interface(interface_name)
        return iface.registers or []

    def interface_names(self) -> list[str]:
        """모든 인터페이스 이름."""
        return list(self.interfaces.keys())

    def get_bank_offset(self, interface_name: str,
                        bank_name: str) -> int:
        """레지스터 뱅크의 베이스 오프셋 조회."""
        iface = self.get_interface(interface_name)
        if not iface.register_banks:
            raise ValueError(
                f"Interface '{interface_name}' has no register banks"
            )
        for bank in iface.register_banks:
            if bank.name == bank_name:
                return bank.base_offset
        raise ValueError(
            f"Bank '{bank_name}' not found in interface '{interface_name}'"
        )
```

---

## 7. FlattenedKernelView & Related Structures

CompositeKernel 플래트닝 후 모든 파이프라인 스테이지가 조작하는 중간 표현.
Unit Kernel도 `_self` 서브커널 하나로 래핑되어 동일한 구조를 사용한다.

### 7.1 InterfaceMapping

```python
@dataclass
class InterfaceMapping:
    """interface_map 적용 결과. 서브커널 인터페이스 → 최상위 매핑."""
    sub_kernel: str              # "dma_ifm" 또는 "_self" (Unit)
    sub_interface: str           # "axi_master"
    mapping_type: MappingType
    top_interface: str | None    # "ddr_port" 또는 None (Internal)
    bank_name: str | None        # "dma_ifm" 또는 None
    bank_offset: int = 0         # 해결된 뱅크 오프셋 (예: 0x000)
```

### 7.2 ExposedTensor

```python
@dataclass
class ExposedTensor:
    """CompositeKernel에서 expose()로 노출된 텐서.
    원본 Tensor로 위임하면서 직렬화 상태를 관리."""
    name: str                    # 최상위 이름 (예: "ifm")
    origin_path: str             # "dma_ifm.src" (서브커널 경로)
    origin_tensor: Tensor        # 실제 Tensor 객체 참조
    top_interface: str           # "ddr_port"
    direction: Direction

    # ── 컴파일 중 설정되는 변경 가능 상태 ──
    _serialized: bytes | None = None
    _serialized_size: int = 0
    _split_buffers: dict[str, bytes] | None = None  # 멀티포트 분할 시

    # ── 원본 위임 ──
    @property
    def data(self):
        return self.origin_tensor.data

    @data.setter
    def data(self, value):
        self.origin_tensor.data = value

    @property
    def shape(self):
        return self.origin_tensor._resolved_shape

    @property
    def element_count(self):
        return self.origin_tensor._element_count

    @property
    def address(self):
        return self.origin_tensor._address

    def set_address(self, addr):
        self.origin_tensor._address = addr
```

### 7.3 ProbePoint

```python
@dataclass
class ProbePoint:
    """Internal(probe=True)인 인터페이스의 golden 데이터 컨테이너."""
    connection: Connect
    interface_mapping: InterfaceMapping
    golden_data: torch.Tensor | None = None
    serialized_golden: bytes | None = None
    golden_buffer_id: int | None = None
```

### 7.4 KernelInstance

```python
@dataclass
class KernelInstance:
    """instantiate() 시 생성되는 커널 인스턴스.
    
    Eager resolution: 생성 시점에 파라미터와 텐서 형상을 즉시 해결하여
    generate_inputs()와 forward()에서 사용 가능하게 한다.
    """
    name: str
    spec: KernelSpec
    kernel_class: type
    kernel_class_instance: Kernel | None = None  # forward() 호출용
    runtime_params: dict = field(default_factory=dict)
    _resolver: 'ParameterResolver | None' = None

    def initialize(self, project_params: dict):
        """인스턴스 초기화: 파라미터 해결 + 텐서 형상 해결 + Kernel 인스턴스 생성.
        
        ExecutionContext.instantiate()에서 호출된다.
        이 메서드 실행 후 generate_inputs()와 forward()가 호출 가능하다.
        """
        import copy

        # 1. 파라미터 해결
        self._resolver = ParameterResolver(
            project_params,              # Priority 3
            self.spec.parameters,        # Priority 2
            self.runtime_params,         # Priority 1
        )

        # 2. Kernel 클래스 인스턴스 생성 + 텐서 복제 (shallow copy, §2.3 참조)
        self.kernel_class_instance = self.kernel_class()
        for tensor in self.kernel_class_instance.tensors():
            # 클래스 변수 Tensor를 인스턴스별 복제로 교체
            instance_tensor = copy.copy(tensor)
            setattr(self.kernel_class_instance, tensor.name, instance_tensor)

            # 3. 텐서 형상 해결 (eager)
            instance_tensor._resolve_shape(self._resolver)

        # 4. 해결된 파라미터를 인스턴스 속성으로 노출
        #    → generate_inputs()에서 self.C, self.N 등으로 접근 가능
        for key, value in self._resolver.namespace.items():
            if not hasattr(self.kernel_class_instance, key):
                setattr(self.kernel_class_instance, key, value)

    def tensors(self) -> list[Tensor]:
        """이 인스턴스의 모든 Tensor (딥카피된 인스턴스별 텐서)."""
        if self.kernel_class_instance:
            return self.kernel_class_instance.tensors()
        return []

    def get_tensor(self, name: str) -> Tensor:
        if self.kernel_class_instance:
            return self.kernel_class_instance.get_tensor(name)
        raise RuntimeError(f"KernelInstance '{self.name}' not initialized")
```

**Eager resolution이 compile() Stage 1-2와 중복되는 이유:**

1. `instantiate()` → eager resolution: **사용자 코드**(`generate_inputs`, `forward`)에서 해결된 형상 사용
2. `compile()` Stage 1-2 → re-validation: **런타임 파이프라인**에서 일관성 재확인
   - 예: 사용자가 `generate_inputs()`에서 `tensor.data`를 다른 형상으로 교체했을 수 있음
   - 이 경우 Stage 2의 `prod(resolved_shape) == tensor.data.numel()` 검증이 에러를 잡음

`compile()` 시에는 `_resolver`가 이미 설정되어 있으므로, Stage 1은 기존 resolver를 재사용하거나 재생성하여 결과가 일치하는지 확인한다.

### 7.5 FlattenedKernelView

```python
@dataclass
class FlattenedKernelView:
    """플래트닝 결과. 모든 파이프라인 스테이지의 작업 대상."""
    name: str
    top_spec: KernelSpec
    sub_kernels: dict[str, KernelInstance]   # Unit: {"_self": ...}
    interface_mappings: list[InterfaceMapping]
    exposed_tensors: dict[str, ExposedTensor]
    probe_points: list[ProbePoint]
    connections: list[Connect]

    # ── 컴파일 중 설정 ──
    _top_resolver: 'ParameterResolver | None' = None
    _register_bindings: list['RegisterBindingEntry'] | None = None

    # ── 쿼리 메서드 ──

    def external_interfaces(self) -> list[str]:
        """BFM 생성이 필요한 최상위 인터페이스 이름들."""
        seen = set()
        result = []
        for m in self.interface_mappings:
            if m.mapping_type in (MappingType.EXTERNAL,
                                  MappingType.EXTERNAL_BANK):
                if m.top_interface not in seen:
                    seen.add(m.top_interface)
                    result.append(m.top_interface)
        return result

    def tensors_for_interface(self, top_iface: str) -> list[ExposedTensor]:
        """특정 최상위 인터페이스에 바인딩된 모든 노출 텐서."""
        return [t for t in self.exposed_tensors.values()
                if t.top_interface == top_iface]

    def registers_for_interface(self, top_iface: str
                                ) -> list[tuple[str, RegisterSpec, int]]:
        """최상위 인터페이스의 모든 레지스터.
        Returns: [(sub_kernel_name, register_spec, absolute_offset)]"""
        result = []
        for m in self.interface_mappings:
            if m.top_interface != top_iface:
                continue
            if m.mapping_type not in (MappingType.EXTERNAL,
                                      MappingType.EXTERNAL_BANK):
                continue
            sub = self.sub_kernels[m.sub_kernel]
            for reg in sub.spec.get_registers(m.sub_interface):
                abs_offset = m.bank_offset + reg.offset
                result.append((m.sub_kernel, reg, abs_offset))
        return result

    def resolve_auto_bind_tensor(self, sub_kernel_name: str,
                                  tensor_name: str) -> ExposedTensor:
        """역방향 조회: 서브커널 텐서 이름 → ExposedTensor.
        auto_bind 해결(Stage 5)에서 사용."""
        origin_path = f"{sub_kernel_name}.{tensor_name}"
        for exposed in self.exposed_tensors.values():
            if exposed.origin_path == origin_path:
                return exposed
        raise BindingError(
            f"auto_bind references tensor '{tensor_name}' in sub-kernel "
            f"'{sub_kernel_name}', but no matching exposed tensor found. "
            f"Ensure the tensor is exposed via .expose() in the "
            f"CompositeKernel definition."
        )
```

---

## 8. Operation & OperationHandle (Record Phase)

Record-then-Compile 아키텍처의 Pass 1에서 사용되는 경량 기록 구조.

```python
@dataclass
class Operation:
    """DSL 호출 한 번이 기록하는 연산."""
    kind: OpKind
    tensor: Tensor | None = None
    kernel: 'KernelInstance | None' = None   # configure()에서 사용 (복수 커널 지원)
    register_interface: str | None = None
    register_fields: dict | None = None
    register_field_name: str | None = None   # poll_register용
    dep: list['OperationHandle'] = field(default_factory=list)
    commit_dep: list['OperationHandle'] = field(default_factory=list)
    probe: bool = False
    sync: bool = False
    golden: torch.Tensor | None = None       # verify()에서 설정
    verify: bool = False


@dataclass
class OperationHandle:
    """사용자에게 반환되는 경량 래퍼. add_commit_dependency() 지원."""
    op: Operation

    def add_commit_dependency(self, other: 'OperationHandle'):
        """이 연산의 commit 시점을 other의 commit 이후로 지연.
        retroactive 수정 — Record-then-Compile이 이를 가능하게 함."""
        self.op.commit_dep.append(other)
```

---

## 9. Execution IR: Command

Runtime이 생성하는 백엔드 독립적 명령 포맷. SHM에 64바이트 슬롯으로 패킹됨.

```python
@dataclass
class Command:
    """Execution IR의 단일 커맨드.
    """
    op: OpCode               # LOAD | PUSH | PULL | STORE | WRITE_REG |
                             # READ_REG | POLL_REG | BARRIER | COMPARE
    cmd_id: int                  # 0부터 시작하는 고유 ID
    interface_id: int = 0        # Binding Table 인덱스
    buffer_id: int = 0           # SHM 데이터 버퍼 인덱스
    protocol: Protocol = Protocol.AXI4S
    phys_addr: int = 0           # 메모리맵 연산의 물리 주소
    size: int = 0                # 전송 크기 (바이트)
    role: Role = Role.MASTER     # BFM 역할

    # 의존성
    dep: list[int] = field(default_factory=list)          # Issue dep cmd_id (최대 4)
    commit_dep: list[int] = field(default_factory=list)   # Commit dep cmd_id (최대 4)

    # 레지스터 연산
    reg_offset: int = 0
    reg_value: int = 0
    reg_mask: int = 0
    reg_expected: int = 0

    # Probe
    probe: bool = False
    golden_buf: int = 0          # golden 데이터 SHM 버퍼 ID

    # 동기화
    sync: bool = False           # flags[0]: SYNC 비트

    # 멀티포트
    port: str = ""               # 분할 인터페이스의 포트 이름
```

**의존성 제한**: Issue dep 최대 4개, Commit dep 최대 4개. 초과 시 BARRIER 사용.

**Operation → Command 확장 규칙:**

| Operation Kind | 확장 결과 | cmd_id 할당 |
|---|---|---|
| LOAD_TENSOR | 1× LOAD | 1 |
| STORE_TENSOR | 1× STORE | 1 |
| PUSH_TENSOR | 1× PUSH per port (분할 시 N개) | 1 또는 N |
| PULL_TENSOR | 1× PULL per port | 1 또는 N |
| WRITE_REGISTER | 1× WRITE_REG per field | 1+ |
| READ_REGISTER | 1× READ_REG | 1 |
| POLL_REGISTER | 1× POLL_REG | 1 |
| CONFIGURE | N× WRITE_REG (모든 auto_bind) | N |
| BARRIER | 1× BARRIER | 1 |
| SEND_TENSOR | LOAD + PUSH (MM) 또는 PUSH만 (Stream) | 1-2 |
| RECV_TENSOR | PULL + STORE (MM) 또는 PULL만 (Stream) | 1-2 |

> ※ CONFIGURE는 `OpKind`로만 존재하며, IR lowering 시 개별 WRITE_REG Command로 확장된다.

---

## 10. Binding Table

Runtime이 구축하는 텐서-인터페이스-BFM 매핑 테이블.

```python
@dataclass
class BindingEntry:
    """텐서 바인딩 정보."""
    tensor_name: str           # "ifm"
    kernel_path: str           # "npu_top.dma_ifm.src"
    interface_name: str        # "ddr_port"
    protocol: Protocol
    phys_address: int          # 0x0000_0000
    size_bytes: int
    buffer_id: int             # SHM 버퍼 인덱스
    serializer: 'StreamSerializer'
    is_internal: bool
    probe: bool


@dataclass
class RegisterBindingEntry:
    """auto_bind 해결 결과."""
    register_name: str         # "dma_ifm.ifm_base_lo"
    kernel_path: str           # "npu_top.dma_ifm.ctrl"
    interface_name: str        # "ctrl"
    absolute_offset: int       # bank_offset + register_offset
    auto_bind: AutoBindSpec
    resolved_value: int


@dataclass
class BFMConfig:
    """BFM 인스턴스 생성에 필요한 설정."""
    interface_name: str        # "ddr_port"
    protocol: Protocol
    data_width: int = 256      # 비트 단위
    addr_width: int = 64       # 주소 버스 폭 (비트). AXI4=64, AXI4L=32
    role: str = "slave"
    address_ranges: list[tuple[int, int, int]] = field(default_factory=list)
        # [(addr, size, buf_id), ...] — AXI4용
    poll_interval: int = 1     # POLL_REG 폴링 간격 (사이클)
    poll_timeout: int = 100000 # POLL_REG 타임아웃 (사이클)


@dataclass
class BindingTable:
    """전체 바인딩 테이블."""
    tensors: dict[str, BindingEntry]
    registers: dict[str, RegisterBindingEntry]
    bfms: dict[str, BFMConfig]
```

---

## 11. SHM Constants & Structures

### 11.1 크기 상수

```python
CONTROL_SIZE     = 256    # 바이트
CMD_SLOT_SIZE    = 64     # 바이트
STATS_SLOT_SIZE  = 32     # 바이트
BUF_DESC_SIZE    = 24     # 바이트
CACHE_LINE       = 64     # 데이터 영역 정렬 단위
```

### 11.2 Magic Number & Protocol

```python
SHM_MAGIC        = 0x5654454E   # "VTEN" (little-endian)
PROTOCOL_VERSION = 0x00000003   # v0.4 protocol
```

### 11.3 ControlHeader

```
Offset  Size   Field                 Description
──────────────────────────────────────────────────────────────────
0x00    4B     magic                 0x5654454E ("VTEN", LE)
0x04    4B     version               Protocol version (0x00000003)
0x08    4B     host_status           Host → Backend 시그널
0x0C    4B     backend_status        Backend → Host 시그널
0x10    4B     num_commands          Command 슬롯 수
0x14    4B     num_buffers           데이터 버퍼 수
0x18    8B     cmd_region_offset     Command Region 시작 오프셋
0x20    8B     stats_region_offset   Stats Region 시작 오프셋
0x28    8B     buf_desc_offset       Buffer Descriptor Table 오프셋
0x30    8B     data_region_offset    Data Region 시작 오프셋
0x38    8B     total_shm_size        전체 SHM 크기 (바이트)
0x40    4B     error_code            에러 코드 (0 = 정상)
0x44    4B     error_cmd_id          에러 발생 커맨드 ID
0x48    64B    error_message         Null-terminated 에러 문자열
0x88    4B     flags                 비트 플래그 (아래 참조)
0x8C    4B     timeout_ms            Backend sem_timedwait 값 (0=기본10s)
0x90    4B     sim_frequency_hz      시뮬레이션 클럭 주파수
0x94    4B     session_seq           세션 시퀀스 번호 (재시작 시 증가)
0x98    104B   reserved              향후 확장 (0으로 초기화)
```

**Total: 256 bytes.**

### 11.4 host_status 값

| 값 | 이름 | 설명 |
|----|------|------|
| 0 | IDLE | 초기 상태 / 완료 ACK |
| 1 | CMD_READY | 커맨드 배치 실행 준비 완료 |
| 2 | ACK | 결과 수신 확인 |
| 3 | SHUTDOWN | 백엔드 종료 요청 |

### 11.5 backend_status 값

| 값 | 이름 | 설명 |
|----|------|------|
| 0 | IDLE | 호스트 대기 중 |
| 1 | RUNNING | 커맨드 배치 실행 중 |
| 2 | DONE | 배치 성공 완료 |
| 3 | ERROR | 에러 발생 (error_code/error_cmd_id/error_message 참조) |

### 11.6 Control flags (offset 0x88)

| 비트 | 이름 | 설명 |
|------|------|------|
| 0 | STATS_ENABLED | 백엔드가 커맨드별 통계 기록 |
| 1 | PROGRESS_ENABLED | 백엔드가 커맨드별 상태 실시간 업데이트 |
| 2 | WAVEFORM_DUMP | 파형 기록 활성화 |
| 3 | WAVEFORM_ON_FAIL | 검증 실패 시에만 파형 기록 |
| 4-31 | reserved | 0이어야 함 |

### 11.7 Command Slot Layout (64 bytes)

```
Offset  Size   Field              Description
──────────────────────────────────────────────────────────────────
0x00    2B     opcode             OpCode 값 (§1.4)
0x02    2B     cmd_id             커맨드 ID (0-based)
0x04    2B     interface_id       Binding Table 인덱스
0x06    1B     protocol           AXI4S=1, AXI4=2, AXI4L=3
0x07    1B     role               MASTER=0, SLAVE=1
0x08    2B     buffer_id          데이터 버퍼 인덱스
0x0A    1B     probe              0=off, 1=on
0x0B    1B     flags              Bit 0: SYNC (1=동기, 0=비동기)
0x0C    4B     size               전송 크기 (바이트)
0x10    8B     phys_addr          물리 주소 (메모리맵 연산)
0x18    4B     reg_offset         레지스터 오프셋 (레지스터 연산)
0x1C    4B     reg_value          레지스터 쓰기 값 / 읽기 결과
0x20    4B     reg_mask           POLL_REG 비교 마스크
0x24    4B     reg_expected       POLL_REG 기대 값
0x28    2B     golden_buf_id      Golden 버퍼 인덱스 (probe 모드)
0x2A    1B     num_deps           Issue 의존성 수 (최대 4)
0x2B    1B     num_commit_deps    Commit 의존성 수 (최대 4)
0x2C    8B     dep_ids            Issue dep cmd_id (4 × 2B)
0x34    8B     commit_dep_ids     Commit dep cmd_id (4 × 2B)
0x3C    4B     reserved           0이어야 함
```

미사용 의존성 슬롯은 `0xFFFF`로 설정.

### 11.8 Buffer Descriptor Layout (24 bytes)

```
Offset  Size   Field              Description
──────────────────────────────────────────────────────────────────
0x00    2B     buffer_id          버퍼 인덱스
0x02    1B     direction          HOST_TO_DEV=0, DEV_TO_HOST=1, BIDIRECTIONAL=2
0x03    1B     flags              Bit 0: GOLDEN (probe golden 데이터)
0x04    4B     size               버퍼 크기 (바이트)
0x08    8B     data_offset        Data Region 시작 기준 바이트 오프셋
0x10    8B     reserved           0이어야 함
```

### 11.9 Stats Entry Layout (32 bytes)

```
Offset  Size   Field                Description
──────────────────────────────────────────────────────────────────
0x00    1B     status              CommandStatus 값 (§1.7)
0x01    1B     reserved
0x02    2B     error_code          커맨드별 에러 코드 (0 = OK)
0x04    4B     issue_cycle         Issue 이벤트 시뮬레이션 사이클
0x08    4B     commit_cycle        Commit 이벤트 시뮬레이션 사이클
0x0C    4B     first_active_cycle  첫 유효 전송 사이클
0x10    4B     last_active_cycle   마지막 유효 전송 사이클
0x14    4B     active_cycles       valid && ready 어서트된 사이클 수
0x18    4B     total_beats         전송된 데이터 비트 수
0x1C    4B     stall_cycles        백프레셔 또는 소스 스톨 사이클 수
```

### 11.10 SHM 크기 계산

```python
def calculate_shm_size(num_commands: int,
                       num_buffers: int,
                       buffer_sizes: list[int]) -> int:
    size = CONTROL_SIZE
    size += CMD_SLOT_SIZE * num_commands
    size += STATS_SLOT_SIZE * num_commands
    size += BUF_DESC_SIZE * num_buffers

    for buf_size in buffer_sizes:
        size = (size + CACHE_LINE - 1) & ~(CACHE_LINE - 1)
        size += buf_size

    size = (size + CACHE_LINE - 1) & ~(CACHE_LINE - 1)
    return size
```

### 11.11 SHM Memory Layout 개요

```
SHM Base ("/vten_{session_id}")
│
├─ [0] Control Region            (256 bytes, 고정)
├─ [1] Command Region            (64 bytes × num_commands)
├─ [2] Stats Region              (32 bytes × num_commands)
├─ [3] Buffer Descriptor Table   (24 bytes × num_buffers)
└─ [4] Data Region               (가변, 64-byte 정렬)
```

### 11.12 Python Host SHM 유틸리티

```python
@dataclass
class BufferDescriptor:
    buffer_id: int
    direction: int
    flags: int
    size: int
    data_offset: int


class SHMBufferAllocator:
    CACHE_LINE = 64

    def __init__(self):
        self.next_offset = 0
        self.descriptors: list[BufferDescriptor] = []

    def allocate(self, buffer_id: int, size: int,
                 direction: int, flags: int = 0) -> int:
        aligned = (self.next_offset + self.CACHE_LINE - 1) \
                  & ~(self.CACHE_LINE - 1)
        desc = BufferDescriptor(
            buffer_id=buffer_id,
            direction=direction,
            flags=flags,
            size=size,
            data_offset=aligned,
        )
        self.descriptors.append(desc)
        self.next_offset = aligned + size
        return aligned

    @property
    def total_data_size(self) -> int:
        return (self.next_offset + self.CACHE_LINE - 1) \
               & ~(self.CACHE_LINE - 1)

    def get_descriptor(self, buffer_id: int) -> BufferDescriptor:
        for d in self.descriptors:
            if d.buffer_id == buffer_id:
                return d
        raise KeyError(f"Buffer {buffer_id} not found")


class CommandMetrics:
    """Stats Region 원시 데이터에서 계산되는 파생 지표."""
    def __init__(self, raw_stats: dict, bus_bytes: int):
        self._raw = raw_stats
        self._bus_bytes = bus_bytes

    @property
    def latency_cycles(self) -> int:
        return self._raw['commit_cycle'] - self._raw['issue_cycle']

    @property
    def active_window(self) -> int:
        if self._raw['first_active_cycle'] == 0:
            return 0
        return self._raw['last_active_cycle'] - self._raw['first_active_cycle'] + 1

    @property
    def utilization(self) -> float:
        if self.active_window == 0:
            return 0.0
        return self._raw['active_cycles'] / self.active_window

    @property
    def bus_efficiency(self) -> float:
        if self.latency_cycles == 0:
            return 0.0
        return self._raw['active_cycles'] / self.latency_cycles

    @property
    def throughput_bytes_per_cycle(self) -> float:
        if self.active_window == 0:
            return 0.0
        return (self._raw['total_beats'] * self._bus_bytes) / self.active_window
```

---

### 11.13 Backend Error Codes

SHM Control Region의 `error_code` 및 BFM `done_error_code` 신호에 사용되는 정수 값.

```python
class BackendErrorCode:
    """Backend 에러 코드 (SHM + SV 양쪽에서 동일 값 사용)."""
    OK                 = 0   # 정상
    ADDR_UNMATCH       = 1   # AXI4 BFM: DUT 주소가 active table에 없음 (DECERR)
    POLL_TIMEOUT       = 2   # AXI-Lite BFM: poll_register 타임아웃
    BFM_QUEUE_ERROR    = 3   # BFM: 내부 큐 에러
    SCHEDULER_ERROR    = 4   # Scheduler: 의존성 해결 불가 (데드락 감지 등)
    SHM_ACCESS_ERROR   = 5   # DPI-C: SHM 읽기/쓰기 실패
    UNKNOWN_OPCODE     = 6   # Scheduler: 알 수 없는 opcode
    BFM_MAP_ERROR      = 7   # Scheduler: interface_id → BFM 매핑 실패
    PROBE_MISMATCH     = 8   # Probe BFM: golden 불일치 (경고, 실행은 계속)
    TIMEOUT            = 9   # 전역 시뮬레이션 타임아웃
```

**에러 코드 사용 위치:**

| 필드 | 위치 | 범위 | 설명 |
|------|------|------|------|
| `error_code` (Control Header) | offset 0x40 | 전역 | 배치 전체의 첫 번째 에러. Scheduler가 설정. |
| `error_cmd_id` (Control Header) | offset 0x44 | 전역 | 에러 발생 커맨드 ID |
| `error_message` (Control Header) | offset 0x48 | 64B | Null-terminated 문자열 |
| `done_error_code` (BFM→Scheduler) | 인터페이스 신호 | BFM당 | 커맨드 단위 에러. Scheduler가 수집하여 Control Header로 전파. |
| `error_code` (Stats Entry) | offset 0x02 | 커맨드당 | 해당 커맨드의 에러 코드 (정상 시 0) |

**에러 코드 → Python 예외 매핑:**

```python
# error_code 정수 → BackendError 하위 클래스 매핑
BACKEND_ERROR_MAP: dict[int, type[BackendError]] = {
    BackendErrorCode.ADDR_UNMATCH:    BFMError,         # 1
    BackendErrorCode.POLL_TIMEOUT:    PollTimeoutError,  # 2
    BackendErrorCode.BFM_QUEUE_ERROR: BFMError,          # 3
    BackendErrorCode.SCHEDULER_ERROR: BackendError,      # 4
    BackendErrorCode.SHM_ACCESS_ERROR:BackendError,      # 5
    BackendErrorCode.UNKNOWN_OPCODE:  BackendError,      # 6
    BackendErrorCode.BFM_MAP_ERROR:   BackendError,      # 7
    BackendErrorCode.PROBE_MISMATCH:  BackendError,      # 8
    BackendErrorCode.TIMEOUT:         TimeoutError,      # 9
}

def raise_backend_error(code: int, cmd_id: int, message: str) -> None:
    """SHM error_code를 읽어 적절한 BackendError 하위 예외를 raise한다.

    매핑에 없는 코드는 BackendError(base)로 raise.
    """
    exc_cls = BACKEND_ERROR_MAP.get(code, BackendError)
    raise exc_cls(f"[cmd_id={cmd_id}] {message} (error_code={code})")
```

**에러 전파 흐름:**

```
BFM 에러 발생
    │
    ▼
BFM: done_valid=1, done_error=1, done_error_code=코드
    │
    ▼
Scheduler: report_error(cmd_id, code)
    │  ├── error_flag <= 1
    │  ├── error_cmd_id <= cmd_id
    │  └── error_code <= code
    ▼
Controller: S_ERROR 진입
    │  ├── vten_signal_error(code, message)
    │  │   ├── ctrl->backend_status = ERROR
    │  │   ├── ctrl->error_code = code
    │  │   ├── ctrl->error_cmd_id = cmd_id
    │  │   └── snprintf(ctrl->error_message, 64, ...)
    │  └── sem_post(b2h)
    ▼
Python Host (XsimBackend.submit_batch):
    backend_status == ERROR → raise_backend_error(code, cmd_id, message)
    backend_status == DONE  → return BackendResult(status=DONE, ...)
    sem_timedwait 타임아웃  → raise TimeoutError(...)
```

**부분 실패(partial failure) 정책:**
- `backend_status=ERROR` 시 `BackendResult`는 반환되지 않는다 (예외만 raise).
- Stats Region: 에러 발생 이전까지 완료된 커맨드의 stats는 유효하다.
  에러 커맨드 자체는 `CommandStatus.ERROR`(=4)로 기록된다.
- Data Region 읽기: `backend_status=ERROR` 후 `read_buffer()` 호출 금지.
  버퍼 내용이 불완전할 수 있음. 진단 목적으로만 허용 (결과 신뢰 불가).

### 11.14 SHM Command Slot → bfm_cmd_t 변환 테이블

Scheduler가 SHM에서 커맨드를 읽을 때 `bfm_cmd_t`(BFM 디스패치용)와 dependency 배열(의존성 추적용)로 분리한다.

| SHM Slot 필드 | Offset | Size | → `bfm_cmd_t` 필드 | → Scheduler 전용 | 비고 |
|---|---|---|---|---|---|
| opcode | 0x00 | 2B | opcode (하위 4비트) | — | |
| cmd_id | 0x02 | 2B | cmd_id | — | |
| interface_id | 0x04 | 2B | interface_id | — | |
| protocol | 0x06 | 1B | protocol | — | |
| role | 0x07 | 1B | role | — | |
| buffer_id | 0x08 | 2B | buffer_id | — | |
| probe | 0x0A | 1B | probe | — | |
| flags | 0x0B | 1B | sync = flags[0] | — | bit 1-7 reserved |
| size | 0x0C | 4B | size | — | |
| phys_addr | 0x10 | 8B | phys_addr | — | |
| reg_offset | 0x18 | 4B | reg_offset | — | |
| reg_value | 0x1C | 4B | reg_value | — | |
| reg_mask | 0x20 | 4B | reg_mask | — | |
| reg_expected | 0x24 | 4B | reg_expected | — | |
| golden_buf_id | 0x28 | 2B | golden_buf_id | — | |
| num_deps | 0x2A | 1B | — | cmd_num_dep[i] | BFM에 전달 안 됨 |
| num_commit_deps | 0x2B | 1B | — | cmd_num_commit_dep[i] | BFM에 전달 안 됨 |
| dep_ids | 0x2C | 8B | — | cmd_dep[i][0:3] | 4 × uint16, 미사용=0xFFFF |
| commit_dep_ids | 0x34 | 8B | — | cmd_commit_dep[i][0:3] | 4 × uint16, 미사용=0xFFFF |
| reserved | 0x3C | 4B | — | — | 0이어야 함 |

**분리 근거:** BFM은 의존성을 알 필요 없음(Scheduler가 모든 의존성 해결 후 디스패치). `bfm_cmd_t`를 작게 유지하여 SV 시뮬레이션 메모리 효율 향상. 관심사 분리: BFM = 프로토콜 실행, Scheduler = 의존성 관리.

---

## 12. Error Taxonomy

```
VTenError
├── ValidationError                       # Stage 0 (빌드 시점)
│   ├── ProtocolMismatchError             # 서브커널↔최상위 프로토콜 불일치
│   ├── BankOverlapError                  # 레지스터 뱅크 주소 겹침
│   └── ConnectionShapeMismatchError      # 연결 텐서 원소 수 불일치
├── CompilationError                      # Stages 1-7
│   ├── ParameterResolutionError          # ${...} 미해결
│   ├── ShapeMismatchError                # 선언 형상 ≠ 데이터 원소 수
│   ├── SerializationError                # 직렬화 실패 (데이터 없음 등)
│   ├── MemoryOverflowError               # 메모리 영역 용량 초과
│   ├── BindingError                      # auto_bind 텐서 미발견
│   ├── DependencyError                   # 순환 의존 등
│   ├── DependencyLimitError              # dep > 4 초과
│   ├── ProbeError                        # probe golden 데이터 미제공
│   └── AliasError                        # 버퍼 앨리어싱 제약 위반 (크기, 패킹 불일치 등)
├── BackendError                          # 실행 시점
│   ├── TimeoutError                      # 세마포어 타임아웃
│   ├── BFMError                          # BFM 주소 미매칭 등
│   └── PollTimeoutError                  # POLL_REG 타임아웃
└── VerificationError                     # 검증 실패
```

**모든 에러에 포함되는 필드:**

```python
class VTenError(Exception):
    kernel_path: str    # 에러 발생 커널/서브커널 경로
    stage: str          # 파이프라인 스테이지 이름
    context: dict       # 관련 파라미터, 텐서 이름, 인터페이스 이름 등
```

---

## 13. Compiled Result & Execution Types

RuntimeEngine.compile()의 최종 산출물 및 실행 관련 타입.

```python
@dataclass
class CompiledResult:
    commands: list[Command]
    shm_image: bytes
    bfm_configs: list[BFMConfig]
    buffer_ids: dict[str, int]          # tensor_name → buffer_id
    flattened_view: FlattenedKernelView
    probe_reports: list[ProbePoint]


@dataclass
class VerificationTask:
    op_handle: OperationHandle
    golden: torch.Tensor


@dataclass
class BackendResult:
    """Backend 실행 완료 후 반환.

    반환 정책:
    - backend_status == DONE  → BackendResult 반환 (status=2)
    - backend_status == ERROR → raise_backend_error() 호출 (예외 raise, 반환 없음)
    - sem_timedwait 타임아웃  → TimeoutError raise (반환 없음)

    즉, BackendResult가 반환된 경우는 항상 status == 2 (DONE) 이다.
    """
    status: int                # backend_status 값 (항상 2=DONE)
    error_code: int = 0        # 항상 0 (DONE 시)
    error_cmd_id: int = 0
    error_message: str = ""
    stats: list["CmdStats"] = field(default_factory=list)  # 커맨드별 통계 (STATS_ENABLED 시)

    def read_buffer(self, buffer_id: int) -> bytes:
        """SHM Data Region에서 특정 버퍼 읽기.

        backend_status=DONE 이후에만 호출 가능.
        ERROR 후 호출 시 동작 미정의 (결과 신뢰 불가).
        """
        ...


@dataclass
class BatchResult:
    """ctx.run() 호출의 반환값. 하나의 Batch 실행 결과를 캡슐화.

    Multi-Batch 실행 시, 각 ctx.run() 호출마다 하나의 BatchResult가 반환된다.
    Stats Region은 Batch마다 덮어쓰이므로, per-command stats는
    ctx.run() 반환 시점에 읽어서 이 객체에 저장된다.
    """
    status: str                           # "DONE" | "ERROR"
    total_cycles: int                     # Backend 클럭 사이클 수
    per_command_stats: list[CmdStats]     # 커맨드별 통계 (STATS_ENABLED 시)
    error: BackendError | None = None     # 에러 상세 (ERROR 시)
```

---

## 14. TestScenario & Test Discovery

### 14.1 TestScenario

```python
class TestScenario:
    """사용자가 작성하는 테스트 시나리오의 베이스 클래스.
    
    하나의 TestScenario는 하나의 검증 시나리오를 정의한다.
    ExecutionContext를 받아 DSL 연산을 기록하고 검증 조건을 설정한다.
    
    Usage:
        class TestConv3D(TestScenario):
            kernel = "conv3d_top"
            
            configs = [
                {"C": 32, "D": 4, "H": 4, "W": 4},
                {"C": 64, "D": 8, "H": 8, "W": 8},
            ]
            
            def run(self, ctx: 'ExecutionContext', cfg: dict):
                k = ctx.instantiate(self.kernel, **cfg)
                k.generate_inputs(seed=42)
                push1 = ctx.push_tensor(k.data_in)
                pull1 = ctx.pull_tensor(k.data_out, dep=push1)
                ctx.verify(pull1, k.forward())
    """
    
    kernel: str = ""               # 대상 커널 이름 (kernel_spec.yaml의 kernel 필드와 매칭)
    
    configs: list[dict] | None = None
    """파라미터 구성 목록. None이면 단일 실행 (vten.toml의 [parameters] 사용).
    리스트이면 각 dict로 run()을 반복 호출하여 multi-config 검증 수행."""
    
    def run(self, ctx: 'ExecutionContext', cfg: dict):
        """테스트 시나리오 정의. 서브클래스에서 반드시 오버라이드.
        
        Args:
            ctx: ExecutionContext — DSL 연산 기록 및 커널 인스턴스화에 사용.
            cfg: dict — 이번 실행의 파라미터 구성.
                 configs가 None이면 vten.toml의 [parameters]만 포함.
                 configs가 리스트이면 해당 dict가 추가 병합됨.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() must be implemented"
        )
```

### 14.2 Test Discovery

CLI(`vten run --test <name>`)가 테스트를 발견하는 메커니즘:

```python
def discover_test(test_name: str, tests_dir: str = "tests") -> TestScenario:
    """테스트 이름으로 TestScenario 서브클래스를 발견하고 인스턴스화.
    
    Discovery 규칙:
    1. tests/ 디렉토리에서 test_*.py 파일 스캔
    2. 각 파일을 import하여 TestScenario 서브클래스 수집
    3. 매칭 전략 (우선순위 순):
       a. 클래스명이 test_name과 일치 (case-insensitive)
          예: --test TestConv3D → TestConv3D
       b. 클래스명을 snake_case로 변환하여 test_name과 일치
          예: --test test_conv3d → TestConv3D
       c. 파일명 기반: test_{test_name}.py 내의 유일한 TestScenario 서브클래스
          예: --test conv3d → tests/test_conv3d.py → 파일 내 첫 번째 서브클래스
    4. 복수 매칭 시 AmbiguousTestError
    5. 미매칭 시 TestNotFoundError
    
    Args:
        test_name: CLI에서 전달된 테스트 식별자
        tests_dir: 테스트 디렉토리 경로 (기본: "tests")
    
    Returns:
        인스턴스화된 TestScenario 서브클래스
    """
    ...
```

### 14.3 Test Execution Flow

```
vten run --test test_conv3d --config C=32
    │
    ├── 1. discover_test("test_conv3d") → TestConv3D 인스턴스
    │
    ├── 2. CLI 파라미터 병합:
    │       base_cfg = vten.toml[parameters] + {"C": 32}  # --config 오버라이드
    │
    ├── 3. configs 결정:
    │       if scenario.configs:
    │           run_cfgs = [merge(base_cfg, c) for c in scenario.configs]
    │       else:
    │           run_cfgs = [base_cfg]
    │
    ├── 4. 각 cfg에 대해:
    │       ctx = ExecutionContext(backend, base_cfg)
    │       scenario.run(ctx, cfg)     # DSL 기록 + ctx.run() 호출 (1회 이상)
    │       ctx.finalize()             # SHM cleanup
    │
    │   Note: scenario.run() 내부에서 ctx.run()을 직접 호출한다.
    │   Single-Batch 시나리오는 ctx.run() 1회, Multi-Batch는 N회 호출.
    │   각 ctx.run()은 pending ops를 compile → submit → wait하고
    │   BatchResult를 반환한다. (상세: 02_runtime_engine.md §3)
    │
    └── 5. 결과 수집 → results/ 디렉토리
```

---

## Cross-Reference Index

| 타입 | 정의 위치 (이 문서) | 사용하는 스펙 파일 |
|------|---------------------|-------------------|
| Terminology Hierarchy | §2 | 모든 파일 |
| Tensor | §3 | 01, 02 |
| RegisterHandle | §4 | 01, 02 |
| Kernel, bind() | §5.1 | 01, 02 |
| TensorProxy, ExposedTensorDef | §5.2 | 01, 02 |
| SubKernelBinding | §5.3 | 01, 02 |
| Connect, Internal | §5.4-5.5 | 01, 02 |
| CompositeKernel | §5.6 | 01, 02 |
| KernelSpec, InterfaceSpec | §6 | 02, 03, 04, 06 |
| PackingScheme | §6.1 | 02, 03 |
| FlattenedKernelView | §7 | 02 |
| KernelInstance (eager resolution) | §7.4 | 02 |
| Operation, OperationHandle | §8 | 02 |
| Command | §9 | 02, 04 |
| BindingTable, BFMConfig | §10 | 02, 04, 05, 06 |
| SHM 상수 및 레이아웃 | §11 | 02, 04, 05 |
| Error Taxonomy (AliasError 포함) | §12 | 모든 파일 |
| CompiledResult, BatchResult | §13 | 02, 06 |
| TestScenario, discover_test | §14 | 06, 07 |
