# vTen Runtime Engine — Compilation Pipeline

**Version 0.5.0 — March 2026**
**참조 모델: `00_data_models.md` (모든 타입 정의)**
**소스: 서플리먼트 전체 + 메인 스펙 §6-9**

---

## Table of Contents

1. [Architectural Position](#1-architectural-position)
2. [Record-then-Compile Architecture](#2-record-then-compile-architecture)
3. [ExecutionContext — User-Facing API](#3-executioncontext--user-facing-api)
4. [8-Stage Compilation Pipeline Overview](#4-8-stage-compilation-pipeline-overview)
5. [Stage 0: Composite Kernel Flattening](#5-stage-0-composite-kernel-flattening)
6. [Stage 1: Parameter Resolution](#6-stage-1-parameter-resolution)
7. [Stage 2: Shape Resolution & Validation](#7-stage-2-shape-resolution--validation)
8. [Stage 2b: Direction Refinement from Operations](#8-stage-2b-direction-refinement-from-operations)
9. [Stage 3: Tensor Serialization](#9-stage-3-tensor-serialization)
10. [Stage 3b: Probe Golden Serialization](#10-stage-3b-probe-golden-serialization)
11. [Stage 4: Address Allocation](#11-stage-4-address-allocation)
12. [Stage 5: auto_bind & Bank Offset Resolution](#12-stage-5-auto_bind--bank-offset-resolution)
13. [Stage 6: IR Lowering](#13-stage-6-ir-lowering)
14. [Stage 6b: BFM Configuration Synthesis](#14-stage-6b-bfm-configuration-synthesis)
15. [Stage 7: SHM Packing](#15-stage-7-shm-packing)
16. [Unit vs Composite Unification](#16-unit-vs-composite-unification)
16. [Validation Rules](#16-validation-rules)

---

## 1. Architectural Position

```
User Code (TestScenario.run)
    │
    ▼
ExecutionContext          ← User-facing API (§3)
    │  records Operation list
    ▼
RuntimeEngine.compile()  ← 8-stage pipeline (§4)
    │  produces (Command[], SHMImage, BFMConfig[])
    ▼
Backend.submit()         ← SHM handshake (04_backend_xsim.md)
    │
    ▼
ExecutionContext._run_verification()
```

**RuntimeEngine**은 ExecutionContext의 내부 컴파일 엔진이다. 사용자가 직접 인스턴스화하지 않는다.

---

## 2. Record-then-Compile Architecture

### 2.1 Two-Pass Architecture

| Pass | 단계 | 내용 |
|------|------|------|
| **Pass 1: Record** | `run()` 실행 중 | 각 `ctx.push_tensor()`, `ctx.configure()` 등이 경량 `Operation` 객체를 내부 리스트에 추가. 컴파일/직렬화/SHM 할당 없음. |
| **Pass 2: Compile** | `ctx.run()` 또는 `ctx.submit()` 호출 시 | 전체 연산 리스트를 8-stage 파이프라인으로 컴파일하여 IR 커맨드, SHM 이미지, BFM 설정 생성. |

### 2.2 Rationale

세 가지 속성이 2-pass를 요구한다:

**R1 — configure()가 모든 텐서 주소를 필요로 함.** `configure()` 호출 시점에 출력 텐서(ofm)가 아직 DSL 연산으로 참조되지 않았을 수 있으나, 주소는 이미 계산되어야 한다. 2-pass에서 주소 할당(Stage 4)은 연산 리스트가 아닌 Kernel 정의의 모든 텐서를 처리한다.

**R2 — add_commit_dependency()가 소급적.** 이전에 생성된 연산을 수정한다. 2-pass에서 모든 의존성 수정이 IR lowering 전에 완료된다.

**R3 — SHM 크기가 전체 버퍼 지식을 필요로 함.** `calculate_shm_size()`는 모든 버퍼의 수와 크기를 미리 알아야 단일 `mmap()`을 수행할 수 있다.

---

## 3. ExecutionContext — User-Facing API

### 3.0 Scope: Single Primary Kernel

현재 `ExecutionContext`는 **단일 primary kernel** 시나리오만 지원한다.
`instantiate()`를 복수 호출할 수 있으나, `RuntimeEngine.compile()`은
`_get_primary_kernel()`로 하나의 커널만 선택하여 `FlattenedKernelView`를 구성한다.

- 복수 커널이 **독립적인 DUT**를 대상으로 하는 경우: 별도의 `ExecutionContext` 사용
- 복수 커널이 **하나의 DUT를 공유**하는 경우: `CompositeKernel`로 모델링

이 제약은 v1.0에서 재검토 대상이다.

### 3.1 Complete Interface

```python
class ExecutionContext:
    def __init__(self, backend: Backend, project_params: dict):
        self._pending_ops: list[Operation] = []
        self._kernels: dict[str, KernelInstance] = {}
        self._backend = backend
        self._project_params = project_params
        self._verifications: list[VerificationTask] = []
        self._alias_registry: AliasRegistry = AliasRegistry()
        self._shm_manager: SHMManager = SHMManager()
        self._last_compiled: CompiledResult | None = None
        self._last_backend_result: BackendResult | None = None

    # ── Kernel Lifecycle ──

    def instantiate(self, kernel_name: str, **params) -> KernelInstance:
        """커널 인스턴스 생성. Eager resolution으로 파라미터와 형상을 즉시 해결.
        이후 generate_inputs()와 forward()에서 해결된 형상 사용 가능."""
        spec = load_kernel_spec(kernel_name)
        instance = KernelInstance(
            name=kernel_name, spec=spec,
            kernel_class=find_kernel_class(kernel_name),
            runtime_params=params,
        )
        # Eager resolution: 파라미터 + 형상 즉시 해결
        instance.initialize(self._project_params)
        self._kernels[kernel_name] = instance
        return instance
```

**instantiate() 상세 흐름:**

```
instantiate("conv3d_top", C=64, D=32)
    │
    ├── 1. load_kernel_spec("conv3d_top")
    │       → specs/ 디렉토리에서 kernel 필드가 "conv3d_top"인 YAML 탐색
    │       → 파싱 → KernelSpec 반환
    │
    ├── 2. find_kernel_class("conv3d_top")
    │       → kernels/ 디렉토리에서 Kernel 서브클래스 탐색
    │       → 매칭 전략 (아래 참조)
    │       → Conv3DKernel 클래스 반환
    │
    ├── 3. KernelInstance 생성
    │       → name, spec, kernel_class, runtime_params 저장
    │
    ├── 4. instance.initialize(project_params)
    │       ├── 4a. ParameterResolver(project, spec.params, runtime) 생성
    │       ├── 4b. kernel_class() 인스턴스 생성
    │       ├── 4c. 각 Tensor: copy.copy → 인스턴스 변수로 설정
    │       ├── 4d. 각 Tensor: _resolve_shape(resolver)
    │       └── 4e. 해결된 파라미터를 인스턴스 속성으로 노출 (self.C=64 등)
    │
    └── 5. _kernels에 등록, instance 반환
        → 이후 instance.generate_inputs(), instance.forward() 호출 가능
```

**load_kernel_spec 구현:**

```python
def load_kernel_spec(kernel_name: str,
                     specs_dir: str = "specs") -> KernelSpec:
    """kernel_name에 해당하는 kernel_spec.yaml을 탐색하고 파싱.
    
    탐색 규칙:
    1. specs/{kernel_name}.yaml 존재 → 직접 파싱
    2. specs/ 디렉토리의 모든 *.yaml 스캔 → kernel 필드가 kernel_name인 파일
    3. 미발견 시 FileNotFoundError
    
    Returns: KernelSpec (03_kernel_spec_schema.md §14 참조)
    """
    # 직접 경로 시도
    direct = Path(specs_dir) / f"{kernel_name}.yaml"
    if direct.exists():
        return parse_kernel_spec(direct)
    
    # 전체 스캔
    for yaml_path in Path(specs_dir).glob("*.yaml"):
        raw = yaml.safe_load(yaml_path.read_text())
        if raw.get('kernel') == kernel_name:
            return parse_kernel_spec(yaml_path)
    
    raise FileNotFoundError(
        f"No kernel_spec.yaml found for kernel '{kernel_name}' "
        f"in {specs_dir}/"
    )
```

**find_kernel_class 구현:**

```python
def find_kernel_class(kernel_name: str,
                      kernels_dir: str = "kernels") -> type:
    """kernel_name에 해당하는 Kernel 서브클래스를 탐색.
    
    탐색 규칙:
    1. kernels/ 디렉토리의 *.py 파일을 import
    2. 각 모듈에서 Kernel 서브클래스(CompositeKernel 포함) 수집
    3. 매칭 전략 (우선순위 순):
       a. 클래스의 spec 속성 경로에서 kernel_name 추출하여 비교
          예: spec = "specs/conv3d_top.yaml" → "conv3d_top" == kernel_name
       b. 클래스명을 snake_case로 변환하여 비교
          예: Conv3DKernel → conv3d_kernel → kernel_name과 비교
       c. 클래스명에서 "Kernel" 접미사 제거 후 snake_case 비교
          예: Conv3DKernel → Conv3D → conv3d → kernel_name
    4. 미발견 시 ImportError
    
    Returns: Kernel 서브클래스 (클래스 자체, 인스턴스 아님)
    """
    ...
```

**인스턴스화 후 사용자 코드에서의 텐서 접근 패턴:**

```python
k = ctx.instantiate("conv3d_top", C=64)
# k는 KernelInstance이지만, 사용자는 k.ifm, k.weight처럼 접근
# 이를 가능하게 하기 위해 KernelInstance는 __getattr__로 위임:

class KernelInstance:
    def __getattr__(self, name):
        """커널 클래스 인스턴스의 속성에 위임.
        k.ifm → k.kernel_class_instance.ifm (복제된 Tensor)
        k.ctrl → k.kernel_class_instance.ctrl (RegisterHandle)
        k.generate_inputs → k.kernel_class_instance.generate_inputs
        k.forward → k.kernel_class_instance.forward
        """
        if self.kernel_class_instance is not None:
            return getattr(self.kernel_class_instance, name)
        raise AttributeError(
            f"KernelInstance '{self.name}' not initialized. "
            f"Attribute '{name}' not accessible."
        )
```

    # ── L1: Host ↔ Memory ──

    def load_tensor(self, tensor, dep=None) -> OperationHandle:
        return self._record(OpKind.LOAD_TENSOR, tensor=tensor, dep=dep)

    def store_tensor(self, tensor, dep=None) -> OperationHandle:
        return self._record(OpKind.STORE_TENSOR, tensor=tensor, dep=dep)

    # ── L2: Accel ↔ Memory ──

    def push_tensor(self, tensor, dep=None, probe=False) -> OperationHandle:
        return self._record(OpKind.PUSH_TENSOR, tensor=tensor,
                            dep=dep, probe=probe)

    def pull_tensor(self, tensor, dep=None, probe=False) -> OperationHandle:
        return self._record(OpKind.PULL_TENSOR, tensor=tensor,
                            dep=dep, probe=probe)

    # ── L3: Control ──

    def write_register(self, register, fields: dict,
                       dep=None) -> OperationHandle:
        """레지스터에 값 쓰기.
        
        Args:
            register: RegisterHandle (예: kernel.ctrl)
            fields: {레지스터_이름: 값} dict.
                레지스터 이름은 kernel_spec.yaml의 registers[].name과 매칭.
                단일 쓰기: {"start": 1}
                복수 쓰기: {"ifm_base_lo": 0x1000, "ifm_base_hi": 0}
            dep: 선행 의존성
        
        Resolution (§12.7):
            1. register.interface_name으로 최상위 인터페이스 식별
            2. fields의 각 key를 인터페이스의 registers[].name과 매칭
            3. 실패 시 registers[].fields의 필드 이름으로 재탐색
            4. 매칭된 레지스터의 absolute_offset과 value로 WRITE_REG 생성
        """
        return self._record(OpKind.WRITE_REGISTER,
                            register_interface=register.interface_name,
                            register_fields=fields, dep=dep)

    def read_register(self, register, field_name, dep=None) -> OperationHandle:
        return self._record(OpKind.READ_REGISTER,
                            register_interface=register.interface_name,
                            register_field_name=field_name, dep=dep)

    def poll_register(self, register, field_name: str,
                      dep=None) -> OperationHandle:
        """레지스터 필드를 폴링하여 조건 충족까지 대기.
        
        Args:
            register: RegisterHandle (예: kernel.ctrl)
            field_name: 레지스터 필드 이름 (예: "done").
                kernel_spec.yaml의 registers[].fields 내 키와 매칭.
            dep: 선행 의존성
        
        Resolution (§12.9):
            1. register.interface_name으로 최상위 인터페이스 식별
            2. 인터페이스의 모든 registers를 순회하여 fields에 field_name이
               있는 레지스터 탐색
            3. 해당 레지스터의 offset과 필드 비트 범위에서 mask/expected 생성:
               - fields: { done: "0:0" } → mask=0x1, expected=0x1
               - fields: { busy: "1:1" } → mask=0x2, expected=0x2
            4. 기본 expected = mask (해당 비트가 모두 1)
        """
        return self._record(OpKind.POLL_REGISTER,
                            register_interface=register.interface_name,
                            register_field_name=field_name, dep=dep)

    def configure(self, kernel, dep=None) -> OperationHandle:
        """kernel의 auto_bind 레지스터를 일괄 설정.
        Operation에 kernel 참조를 저장하여 IR lowering에서
        어떤 커널의 auto_bind를 처리할지 식별."""
        return self._record(OpKind.CONFIGURE, kernel=kernel, dep=dep)

    def barrier(self) -> OperationHandle:
        return self._record(OpKind.BARRIER)

    # ── Shorthands ──

    def send_tensor(self, tensor, dep=None) -> OperationHandle:
        return self._record(OpKind.SEND_TENSOR, tensor=tensor, dep=dep)

    def recv_tensor(self, tensor, dep=None,
                    chunks: int | list[int] | None = None,
                    ) -> OperationHandle | list[OperationHandle]:
        """chunks: 전송을 N개 청크로 분할. BFM이 청크 사이에 tready deassert.
        int → N등분, list[int] → explicit per-chunk element counts.
        Returns list[OperationHandle] when chunks is specified."""
        if chunks is None:
            return self._record(OpKind.RECV_TENSOR, tensor=tensor, dep=dep)
        # Creates N separate RECV_TENSOR ops, each with chunk_index/chunk_total
        return [self._record(OpKind.RECV_TENSOR, tensor=tensor, dep=dep,
                             chunk_index=i, chunk_total=N, chunks_spec=chunks)
                for i in range(N := len(chunks) if isinstance(chunks, list) else chunks)]

    # ── Verification ──

    def verify(self, op_handle, golden):
        self._verifications.append(VerificationTask(
            op_handle=op_handle, golden=golden))

    # ── Buffer Aliasing ──

    def alias(self, src: TensorRef, dst: TensorRef) -> None:
        """dst가 src의 SHM 버퍼를 직접 재사용하도록 선언.
        상세: 01_kernel_and_dsl.md §3.7 참조."""
        self._alias_registry.register(src, dst)

    # ── Execution ──

    def run(self) -> 'BatchResult':
        """Pending Operations를 하나의 Batch로 컴파일 → 제출 → 완료 대기.

        여러 번 호출 가능. 각 호출은 직전 run() 이후 기록된 Operation만
        컴파일한다. BatchResult를 반환한다 (00_data_models.md §13).

        동작:
            1. pending ops → 8-stage 파이프라인 컴파일.
               alias registry를 참조하여 LOAD/STORE 자동 skip.
            2. Commands → SHM Command Region 직렬화.
            3. 새 LOAD 버퍼만 SHM Data Region에 기록.
               이전 Batch에서 유효한 버퍼(alias)는 재기록하지 않음.
            4. Buffer Descriptor Table 기록 (이번 Batch 버퍼만).
            5. Control Header에 num_commands, num_buffers 기록.
            6. host_status = CMD_READY; sem_post(h2b).
            7. sem_timedwait(b2h, timeout) — Backend 완료 대기.
            8. Stats Region 읽기 → BatchResult 구성.
            9. 내부 상태 초기화:
               - pending_ops = []
               - cmd_id 카운터 0으로 리셋
               - Command Region, Stats Region은 stale (다음 run()이 덮어씀)
               - Data Region은 보존 (cross-Batch alias 지원)
               - Alias registry 보존 (cross-Batch alias 유효)
            10. BatchResult 반환.

        후방호환:
            Single-Invocation 테스트에서 run()을 1회 호출하면
            기존 동작과 동일. 코드 변경 불필요.
        """
        engine = RuntimeEngine(
            kernels=self._kernels,
            ops=self._pending_ops,
            project_params=self._project_params,
            alias_registry=self._alias_registry,
            shm_manager=self._shm_manager,
        )
        compiled = engine.compile()
        backend_result = self._backend.submit(
            shm_image=compiled.shm_image,
            bfm_configs=compiled.bfm_configs,
        )
        self._last_compiled = compiled
        self._last_backend_result = backend_result
        self._pending_ops = []

        # Deferred verification: run() 전에 등록된 verify() 실행
        self._run_deferred_verifications(compiled, backend_result)

        return BatchResult(
            status="DONE",
            total_cycles=backend_result.stats_total_cycles(),
            per_command_stats=backend_result.stats,
        )

    # ── Verification ──

    def verify(self, op_handle, golden):
        """검증 수행.

        ctx.run() 이전에 호출: deferred VerificationTask로 기록.
            → run() 종료 시 자동 실행 (후방호환).
        ctx.run() 이후에 호출: 즉시(eager) 검증 수행.
            → SHM에서 버퍼 데이터를 읽어 golden과 비교.
            → VerificationError 시 즉시 raise.
        """
        if self._last_compiled is not None:
            # Eager mode: run() 이후 호출
            self._verify_immediate(op_handle, golden)
        else:
            # Deferred mode: run() 이전 호출 (후방호환)
            self._verifications.append(VerificationTask(
                op_handle=op_handle, golden=golden))

    # ── Internal ──

    def _record(self, kind, **kwargs) -> OperationHandle:
        dep = kwargs.pop('dep', None)
        op = Operation(
            kind=kind,
            dep=self._normalize_deps(dep),
            commit_dep=[],
            probe=kwargs.pop('probe', False),
            sync=kwargs.pop('sync', False),
            golden=None, verify=False,
            **kwargs,
        )
        self._pending_ops.append(op)
        return OperationHandle(op)

    def _normalize_deps(self, dep):
        if dep is None: return []
        if isinstance(dep, OperationHandle): return [dep]
        return list(dep)

    def _verify_immediate(self, op_handle, golden):
        """Eager verification: run() 이후 호출 시 즉시 비교."""
        tensor_name = op_handle.op.tensor.name
        buffer_id = self._last_compiled.buffer_ids[tensor_name]
        raw_bytes = self._last_backend_result.read_buffer(buffer_id)

        exposed = self._last_compiled.flattened_view.exposed_tensors[tensor_name]
        iface = self._last_compiled.flattened_view.top_spec.get_interface(
            exposed.top_interface)
        deserializer = StreamSerializer(iface.packing)
        hw_output = deserializer.deserialize(
            raw_bytes,
            exposed.origin_tensor._element_count,
            exposed.origin_tensor._resolved_shape,
        )
        if not self._compare(hw_output, golden, exposed):
            raise VerificationError(
                tensor=tensor_name,
                shape=exposed.origin_tensor._resolved_shape,
                max_diff=self._max_diff(hw_output, golden),
            )

    def _run_deferred_verifications(self, compiled, result):
        """Deferred verification: run() 전에 등록된 verify() 일괄 실행."""
        for task in self._verifications:
            self._verify_immediate(task.op_handle, task.golden)
        self._verifications.clear()
```

---

## 4. 8-Stage Compilation Pipeline Overview

```python
class RuntimeEngine:
    def compile(self) -> CompiledResult:
        kernel = self._get_primary_kernel()

        # Stage 0: Flatten (Composite) or Wrap (Unit)
        view = (self._flatten_composite(kernel)
                if isinstance(kernel, CompositeKernel)
                else self._wrap_unit_as_flat(kernel))

        self._resolve_parameters(view)          # Stage 1
        self._resolve_shapes(view)              # Stage 2
        self._serialize_tensors(view)           # Stage 3
        self._serialize_probe_golden(view)      # Stage 3b
        self._allocate_addresses(view)          # Stage 4
        self._resolve_auto_binds(view)          # Stage 5

        commands, buffer_ids = self._lower_to_ir(view, self._ops)  # Stage 6
        bfm_configs = self._synthesize_bfm_configs(view, commands) # Stage 6b
        shm_image = self._pack_shm(view, commands, buffer_ids)    # Stage 7

        return CompiledResult(
            commands=commands, shm_image=shm_image,
            bfm_configs=bfm_configs, buffer_ids=buffer_ids,
            flattened_view=view, probe_reports=view.probe_points,
        )
```

### Inter-Stage Data Dependencies

```
Stage 0 ──► FlattenedKernelView (불변 구조)
Stage 1 ──► 서브커널별 resolver (view에 저장)
Stage 2 ──► 해결된 형상, 원소 수 (텐서에 저장)
Stage 3 ──► 직렬화된 바이트, 크기, _port_buffers (ExposedTensor에 저장)
Stage 3b ──► golden 바이트 (ProbePoint에 저장)
Stage 4 ──► 물리 주소 (ExposedTensor → origin_tensor에 저장)
Stage 5 ──► 레지스터 바인딩 (view._register_bindings에 저장)
Stage 6 ──► Command[], buffer_ids 매핑
Stage 6b ──► BFMConfig[]
Stage 7 ──► 최종 SHM 바이트 이미지
```

**Critical ordering constraints:**
- Stage 4는 Stage 3 필요 (할당에 serialized_size 필요)
- Stage 5는 Stage 4 필요 (auto_bind에 주소 필요)
- Stage 6은 Stage 5 필요 (WRITE_REG에 레지스터 값 필요)
- Stage 3b는 Stage 2 필요 (golden 직렬화에 해결된 형상 필요)
- Stage 3b는 Stage 4와 독립 (golden은 서브커널 패킹 사용)

### 4.2 Multi-Config Compilation (compile_multi)

단일 배치에 여러 config 그룹을 묶어 컴파일한다. `config_boundary()`로 구분된 N개 config를 하나의 SHM 이미지로 합성.

```python
@staticmethod
def compile_multi(engines: list[RuntimeEngine], target: str = "sim") -> CompiledResult:
    """여러 config 그룹을 단일 배치로 컴파일.

    각 engine은 독립 compile()을 수행한 후:
    1. cmd_id를 누적 오프셋으로 재배정
    2. buffer_id를 'cfg{idx}:{name}' 접두사로 prefix
    3. config 간 BARRIER 커맨드 자동 삽입
    4. 모든 SHM 영역을 하나의 이미지로 합성
    """
```

**Buffer naming convention**: multi-config에서 각 config의 버퍼는 `cfg{idx}:{tensor_name}` 키로 구분된다. 예: `cfg0:data_in`, `cfg1:data_in`, `cfg2:data_out`.

**Verify 시 config_group 참조**: `_verify_immediate()`는 `Operation.config_group`을 참조하여 해당 config의 `views[config_group]`과 `cfg{config_group}:` prefix를 사용해 올바른 버퍼 데이터에 접근한다.

---

## 5. Stage 0: Composite Kernel Flattening

### 5.1 Algorithm

```python
def _flatten_composite(self, composite):
    top_spec = load_kernel_spec(composite.spec)

    # Phase A: 서브커널 인스턴스화
    sub_kernels = {}
    for name, binding in composite.bindings():
        sub_spec = load_kernel_spec(binding.kernel_class.spec)
        sub_kernels[name] = KernelInstance(
            name=name, spec=sub_spec,
            kernel_class=binding.kernel_class,
            params=binding.params or {},
        )

    # Phase B: 인터페이스 매핑 구축
    interface_mappings = []
    for name, binding in composite.bindings():
        sub_spec = sub_kernels[name].spec
        for sub_iface_name in sub_spec.interface_names():
            if sub_iface_name not in binding.interface_map:
                raise ValidationError(
                    f"Sub-kernel '{name}': interface '{sub_iface_name}' "
                    f"has no mapping in interface_map.")
            interface_mappings.append(
                self._parse_mapping(name, sub_iface_name,
                                    binding.interface_map[sub_iface_name],
                                    top_spec))

    # Phase C: 노출 텐서 수집
    exposed_tensors = {}
    for name, tensor_def in composite.exposed_tensor_defs():
        origin_tensor = sub_kernels[tensor_def.origin_sub_kernel].get_tensor(
            tensor_def.origin_name)
        exposed_tensors[name] = ExposedTensor(
            name=name,
            origin_path=f"{tensor_def.origin_sub_kernel}.{tensor_def.origin_name}",
            origin_tensor=origin_tensor,
            top_interface=tensor_def.top_interface,
            direction=self._infer_direction(
                tensor_def.origin_sub_kernel, origin_tensor, interface_mappings),
        )

    # Phase D: Probe 포인트 수집
    probe_points = []
    for mapping in interface_mappings:
        if mapping.mapping_type == MappingType.INTERNAL_PROBE:
            conn = self._find_connection_for_interface(
                composite.connections, mapping.sub_kernel, mapping.sub_interface)
            probe_points.append(ProbePoint(
                connection=conn, interface_mapping=mapping))

    # Phase E: 빌드 시점 검증
    self._validate_flattened(interface_mappings, exposed_tensors,
                             composite.connections, top_spec, sub_kernels)

    return FlattenedKernelView(
        name=top_spec.kernel_name, top_spec=top_spec,
        sub_kernels=sub_kernels, interface_mappings=interface_mappings,
        exposed_tensors=exposed_tensors, probe_points=probe_points,
        connections=composite.connections,
    )
```

### 5.2 Mapping Parser

```python
def _parse_mapping(self, sub_kernel_name, sub_iface_name,
                   mapping_value, top_spec):
    if isinstance(mapping_value, Internal):
        return InterfaceMapping(
            sub_kernel=sub_kernel_name, sub_interface=sub_iface_name,
            mapping_type=(MappingType.INTERNAL_PROBE
                         if mapping_value.probe
                         else MappingType.INTERNAL),
            top_interface=None, bank_name=None, bank_offset=0)

    elif isinstance(mapping_value, str):
        self._validate_protocol_compat(
            sub_kernel_name, sub_iface_name, mapping_value, top_spec)
        return InterfaceMapping(
            sub_kernel=sub_kernel_name, sub_interface=sub_iface_name,
            mapping_type=MappingType.EXTERNAL,
            top_interface=mapping_value, bank_name=None, bank_offset=0)

    elif isinstance(mapping_value, tuple) and len(mapping_value) == 2:
        top_iface, bank_name = mapping_value
        bank_offset = top_spec.get_bank_offset(top_iface, bank_name)
        self._validate_protocol_compat(
            sub_kernel_name, sub_iface_name, top_iface, top_spec)
        return InterfaceMapping(
            sub_kernel=sub_kernel_name, sub_interface=sub_iface_name,
            mapping_type=MappingType.EXTERNAL_BANK,
            top_interface=top_iface, bank_name=bank_name,
            bank_offset=bank_offset)

    else:
        raise ValidationError(
            f"Sub-kernel '{sub_kernel_name}', interface "
            f"'{sub_iface_name}': invalid mapping value {mapping_value!r}")
```

### 5.3 Direction Resolution

텐서의 데이터 전송 방향은 다음 우선순위로 결정한다:

**1단계 — 명시값 우선:** `Tensor(direction=Direction.HOST_TO_DEV)` 등으로
명시된 경우 해당 값을 그대로 사용한다. 추론하지 않는다.

**2단계 — 프로토콜/역할 추론 (direction=None인 경우):**
- AXI4-Stream: role=MASTER (BFM이 송신) → HOST_TO_DEV,
               role=SLAVE (BFM이 수신) → DEV_TO_HOST
- AXI4-Lite: 레지스터 전용이므로 텐서 방향 해당 없음

**AXI4 (Memory-Mapped) 특별 규칙:**
같은 AXI4 포트에 읽기/쓰기 텐서가 혼재할 수 있으므로
AXI4 인터페이스에 바인딩된 텐서는 `direction`을 명시해야 한다.
생략 시 `HOST_TO_DEV`로 기본값을 사용하지만, 경고를 발생시킨다.

```python
# 예시: 같은 ddr_port에 입출력 텐서 혼재
ifm = Tensor(shape, torch.int8, "ddr_port", direction=Direction.HOST_TO_DEV)
ofm = Tensor(shape, torch.int8, "ddr_port", direction=Direction.DEV_TO_HOST)
```

---

## 6. Stage 1: Parameter Resolution

### 6.1 Hierarchical Scope Chain

| 우선순위 | 스코프 | 소스 | 예 |
|----------|--------|------|-----|
| 1 (최고) | Runtime Object | 테스트 `Config()` | `Config(C=32, D=4)` |
| 2 | Kernel Template | `kernel_spec.yaml` | `K: 128, STRIDE: 1` |
| 3 (최저) | Project Defaults | `vten.toml [parameters]` | `C=64, BUS_WIDTH=256` |

### 6.2 Composite 동작

```python
def _resolve_parameters(self, view):
    top_resolver = ParameterResolver(
        self._project_params,
        view.top_spec.parameters,
        self._kernels[view.name].runtime_params,
    )

    for name, sub in view.sub_kernels.items():
        if name == "_self":
            sub._resolver = top_resolver
            continue

        binding_params = {}
        binding = self._get_binding(name)
        if binding.params:
            for k, expr in binding.params.items():
                binding_params[k] = top_resolver.resolve(expr)

        sub._resolver = ParameterResolver(
            self._project_params,
            {**view.top_spec.parameters, **binding_params},
            sub.spec.parameters,
        )
    view._top_resolver = top_resolver
```

### 6.3 Expression Evaluation

```python
class ParameterResolver:
    def __init__(self, project_params, kernel_params, runtime_params):
        self.namespace = {}
        self.namespace.update(project_params)
        self.namespace.update(kernel_params)
        self.namespace.update(runtime_params)

    def resolve(self, expr):
        if not isinstance(expr, str) or "${" not in expr:
            return expr

        def _substitute(m):
            name = m.group(1)
            if name not in self.namespace:
                raise ParameterResolutionError(
                    f"Unresolved parameter '${{{name}}}' in expression '{expr}'. "
                    f"Available: {sorted(self.namespace)}")
            return str(self.namespace[name])

        resolved = re.sub(r'\$\{(\w+)\}', _substitute, expr)
        return safe_eval(resolved)  # 산술 연산만 허용
```

표현식 예시:
```python
shape=("${N}", "${C}", "${D}", "${H}", "${W}")                    # 단순 치환
shape=("${N}", "${K}", "(${D}-${KD})//${STRIDE}+1", ...)         # 산술
shape=("${N}", "${C}//${TILE_C}", "${D}", "${H}", "${W}", "${TILE_C}")  # 교차
```

---

## 7. Stage 2: Shape Resolution & Validation

```python
def _resolve_shapes(self, view):
    for name, sub in view.sub_kernels.items():
        for tensor in sub.tensors():
            tensor._resolved_shape = tuple(
                sub._resolver.resolve(dim) for dim in tensor.shape)
            tensor._element_count = math.prod(tensor._resolved_shape)

            if tensor.data is not None:
                actual = tensor.data.numel()
                if actual != tensor._element_count:
                    raise ShapeMismatchError(
                        f"Tensor '{name}.{tensor.name}': declared shape "
                        f"{tensor._resolved_shape} ({tensor._element_count} "
                        f"elements) but data has {actual} elements.")

    # Connection 형상 호환성 (Composite만 해당)
    for conn in view.connections:
        src = view.sub_kernels[conn.source_sub].get_tensor(conn.source_name)
        dst = view.sub_kernels[conn.dest_sub].get_tensor(conn.dest_name)
        if src._element_count != dst._element_count:
            raise ConnectionShapeMismatchError(
                f"Connection {conn.source_sub}.{conn.source_name} → "
                f"{conn.dest_sub}.{conn.dest_name}: "
                f"{src._element_count} vs {dst._element_count} elements.")
```

---

## 8. Stage 2b: Direction Refinement from Operations

Shape validation 이후, DSL 연산 목록에서 텐서의 방향(Direction)을 추론 또는 확인한다.

### 8.1 추론 규칙

| DSL Operation | 추론 방향 |
|---------------|----------|
| `push_tensor`, `send_tensor` | `HOST_TO_DEV` |
| `pull_tensor`, `recv_tensor` | `DEV_TO_HOST` |
| `configure` | 해당 없음 (register only) |

### 8.2 동작

```python
def _refine_directions_from_ops(self, view):
    for op in self._ops:
        if op.tensor is None:
            continue
        inferred = _direction_from_op(op.kind)
        if inferred is None:
            continue
        if op.tensor.direction is not None and op.tensor.direction != inferred:
            raise DirectionConflictError(
                f"Tensor '{op.tensor.name}': explicit direction "
                f"{op.tensor.direction} conflicts with {op.kind}")
        op.tensor.direction = inferred
```

### 8.3 설계 근거

- `Tensor.direction`은 선택적 필드 (`None` → 추론)
- 명시적으로 설정된 direction과 DSL 연산이 충돌하면 에러
- Stage 3 직렬화 전에 방향이 확정되어야 LOAD/STORE 결정 가능

---

## 9. Stage 3: Tensor Serialization

### 8.1 외부 텐서만 직렬화

내부 인터페이스 텐서는 직렬화하지 않는다 — RTL 와이어로만 존재.

```python
def _serialize_tensors(self, view):
    for name, exposed in view.exposed_tensors.items():
        iface_spec = view.top_spec.get_interface(exposed.top_interface)
        packing = iface_spec.packing

        if exposed.direction == Direction.HOST_TO_DEV:
            if exposed.origin_tensor.data is None:
                raise SerializationError(
                    f"Tensor '{name}' has no data. "
                    f"Call generate_inputs() before run().")
            serializer = StreamSerializer(packing)
            exposed._serialized = serializer.serialize(exposed.origin_tensor.data)
            exposed._serialized_size = len(exposed._serialized)
        else:
            num_beats = math.ceil(
                exposed.origin_tensor._element_count / packing.elements_per_beat)
            exposed._serialized = None
            exposed._serialized_size = num_beats * (packing.bus_width // 8)

        # 멀티포트 분할 → _port_buffers (split)
        if iface_spec.split:
            split_spec = _parse_split_spec(iface_spec.split)
            if exposed._serialized is not None:
                splitter = MultiPortSerializer()
                exposed._port_buffers = splitter.split_tensor(
                    exposed._serialized, split_spec)
            else:
                # 출력 텐서: 포트별 빈 버퍼 할당
                n_ports = len(split_spec.ports)
                per_port_size = exposed._serialized_size // n_ports
                exposed._port_buffers = {
                    p.name: bytes(per_port_size) for p in split_spec.ports}
            exposed._port_mode = split_spec.mode
            if split_spec.interleave:
                exposed._interleave_unit = split_spec.interleave.unit

        # 멀티포트 분할 → _port_buffers (array, split이 없을 때)
        if iface_spec.array and not exposed._port_buffers:
            flat_names = iface_spec.array.flat_names(exposed.top_interface)
            exposed._port_buffers = _block_split_data(
                exposed._serialized, flat_names, exposed._serialized_size)
            exposed._port_mode = "block"
```

### 8.2 StreamSerializer

**직렬화 원소 순서: 항상 C-contiguous (row-major)**

`serialize()`는 `tensor_data.flatten()`(PyTorch 기본 = C-contiguous)으로 원소를 순서화한다.
즉 shape `[N, C, H, W]`일 때 W 축이 가장 빠르게 변한다.
`deserialize()`도 동일한 순서를 따른다.

```python
class StreamSerializer:
    def __init__(self, packing: PackingScheme):
        self.packing = packing

    def serialize(self, tensor_data) -> bytes:
        """텐서 → 바이트 스트림. 원소 순서: C-contiguous (row-major)."""
        flat = tensor_data.flatten()  # C-contiguous 보장
        beats = []
        for i in range(0, len(flat), self.packing.elements_per_beat):
            chunk = flat[i:i + self.packing.elements_per_beat]
            beat = self._pack_beat(chunk)
            beats.append(beat)
        return b''.join(beats)

    def _pack_beat(self, elements) -> bytes:
        beat_val = 0
        for idx, elem in enumerate(elements):
            raw = int(elem) & ((1 << self.packing.element_width) - 1)
            if self.packing.bit_order == 'lsb_first':
                shift = idx * self.packing.element_width
            else:
                shift = (self.packing.elements_per_beat - 1 - idx) \
                        * self.packing.element_width
            beat_val |= (raw << shift)
        num_bytes = (self.packing.bus_width + 7) // 8
        return beat_val.to_bytes(num_bytes, byteorder=self.packing.byte_order)

    def deserialize(self, raw_bytes, num_elements, shape=None):
        """역방향: 바이트 스트림 → Tensor. 원소 순서: C-contiguous."""
        # 구현: _pack_beat의 역순
        ...

    # ── 디버그 유틸리티 ──────────────────────────────────────────────

    def beat_index_to_coords(self, beat_index: int,
                             shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """beat_index → 해당 beat에 포함된 원소들의 텐서 좌표 목록 반환.

        직렬화 순서는 C-contiguous (row-major).
        마지막 beat가 elements_per_beat보다 적은 원소를 포함하는 경우
        실제 원소 수만큼만 반환한다.

        사용 예 (Probe mismatch 로그):
            coords = serializer.beat_index_to_coords(beat_idx, tensor.shape)
            # coords = [(0, 5, 0, 0), (0, 5, 0, 1), ...]
        """
        elem_start = beat_index * self.packing.elements_per_beat
        total_elems = math.prod(shape)
        coords = []
        for elem_idx in range(elem_start,
                              min(elem_start + self.packing.elements_per_beat,
                                  total_elems)):
            coords.append(_flat_to_coords(elem_idx, shape))
        return coords


def _flat_to_coords(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """C-contiguous flat index → 텐서 좌표 (shape 기준 row-major)."""
    coords = []
    for dim in reversed(shape):
        coords.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(coords))
```

### 8.3 MultiPortSerializer

```python
class MultiPortSerializer:
    def split_tensor(self, serialized, split_spec):
        if split_spec.mode == 'channel_interleave':
            return self._interleave_split(serialized, split_spec)
        elif split_spec.mode == 'block_split':
            return self._block_split(serialized, split_spec)

    def _interleave_split(self, data, spec):
        unit = spec.interleave.unit
        num_ports = len(spec.ports)
        result = {p.name: bytearray() for p in spec.ports}
        for i in range(0, len(data), unit):
            port_idx = (i // unit) % num_ports
            result[spec.ports[port_idx].name].extend(data[i:i+unit])
        return {k: bytes(v) for k, v in result.items()}

    @staticmethod
    def reassemble(port_data: dict[str, bytes], interleave_unit: int) -> bytes:
        """Reverse channel_interleave: round-robin 재조립."""
        ports = list(port_data.values())
        n_ports = len(ports)
        result = bytearray()
        offsets = [0] * n_ports
        while any(offsets[i] < len(ports[i]) for i in range(n_ports)):
            for i in range(n_ports):
                chunk = ports[i][offsets[i]:offsets[i] + interleave_unit]
                result.extend(chunk)
                offsets[i] += interleave_unit
        return bytes(result)
```

### 8.4 헬퍼 함수

```python
def _parse_split_spec(raw) -> SplitSpec:
    """raw dict 또는 SplitSpec → SplitSpec 변환."""
    if isinstance(raw, SplitSpec):
        return raw
    ports = [PortDef(name=p["name"], base_addr=p.get("base_addr", 0))
             for p in raw.get("ports", [])]
    interleave = None
    if "interleave" in raw:
        interleave = InterleaveSpec(unit=raw["interleave"]["unit"])
    return SplitSpec(mode=raw["mode"], ports=ports, interleave=interleave)

def _block_split_data(serialized, flat_names, serialized_size) -> dict[str, bytes]:
    """블록 분할: 직렬화 데이터를 포트별로 균등 분배."""
    n = len(flat_names)
    if serialized is not None:
        chunk_size = len(serialized) // n
        remainder = len(serialized) % n
        result, offset = {}, 0
        for i, fname in enumerate(flat_names):
            sz = chunk_size + (1 if i < remainder else 0)
            result[fname] = serialized[offset:offset + sz]
            offset += sz
        return result
    else:
        per_elem_size = serialized_size // n
        return {fname: bytes(per_elem_size) for fname in flat_names}
```

---

## 9. Stage 3b: Probe Golden Serialization

### 9.1 두 가지 전략

| 전략 | 트리거 | 장점 | 단점 |
|------|--------|------|------|
| **Explicit** | 사용자가 `forward_with_intermediates()` 정의 | 임의 Python 동작. 캡처 대상 제어. | 사용자 노력 필요. |
| **Automatic** | 미정의 시; Runtime이 `connections` DAG 순회 | 단순 선형 파이프라인에서 노력 제로. | 분기, 조건, 상태 있으면 실패. |

Explicit이 정의되면 우선. 미정의 시 Automatic 시도. golden 데이터 미발견 시 `ProbeError`.

### 9.2 Implementation

```python
def _serialize_probe_golden(self, view):
    if not view.probe_points:
        return

    kernel_instance = self._kernels[view.name]
    if hasattr(kernel_instance, 'forward_with_intermediates'):
        intermediates = kernel_instance.forward_with_intermediates()
    else:
        intermediates = self._auto_collect_intermediates(view)

    for probe in view.probe_points:
        key = (f"{probe.interface_mapping.sub_kernel}"
               f".{probe.interface_mapping.sub_interface}")
        if key not in intermediates:
            raise ProbeError(f"Probe point '{key}' has no golden data.")
        probe.golden_data = intermediates[key]

        sub = view.sub_kernels[probe.interface_mapping.sub_kernel]
        sub_iface = sub.spec.get_interface(probe.interface_mapping.sub_interface)
        serializer = StreamSerializer(sub_iface.packing)
        probe.serialized_golden = serializer.serialize(probe.golden_data)
```

### 9.3 Automatic Collection via Topological Sort

```python
def _auto_collect_intermediates(self, view):
    intermediates = {}
    probed = {(p.interface_mapping.sub_kernel, p.interface_mapping.sub_interface)
              for p in view.probe_points}
    order = self._topo_sort_sub_kernels(view.connections, view.sub_kernels)

    for sub_name in order:
        sub = view.sub_kernels[sub_name]
        output = sub.kernel_class_instance.forward()
        for conn in view.connections:
            if conn.source_sub != sub_name:
                continue
            data = conn.transform(output) if conn.transform else output
            key = (conn.source_sub, conn.source_interface)
            if key in probed:
                intermediates[f"{key[0]}.{key[1]}"] = data.clone()
            dest = view.sub_kernels[conn.dest_sub].get_tensor(conn.dest_name)
            dest.data = data
    return intermediates
```

---

## 10. Stage 4: Address Allocation

```python
def _allocate_addresses(self, view):
    allocators = {}
    for region_name, region in view.top_spec.memory_regions.items():
        allocators[region_name] = AddressAllocator(region)

    for name, exposed in view.exposed_tensors.items():
        iface = view.top_spec.get_interface(exposed.top_interface)
        if not iface.memory_region:
            continue  # 스트림 인터페이스 — 주소 불필요
        if exposed.origin_tensor._address is not None:
            continue  # 사용자 오버라이드
        addr = allocators[iface.memory_region].allocate(
            tensor_name=f"{view.name}.{name}",
            size=exposed._serialized_size)
        exposed.set_address(addr)
```

```python
class AddressAllocator:
    def __init__(self, region: MemoryRegion):
        self.region = region
        self.next_addr = region.base

    def allocate(self, tensor_name, size):
        aligned = self._align_up(self.next_addr, self.region.alignment)
        if aligned + size > self.region.base + self.region.size:
            raise MemoryOverflowError(f"{tensor_name} exceeds {self.region.name}")
        addr = aligned
        self.next_addr = aligned + size
        return addr

    def _align_up(self, addr, alignment):
        return (addr + alignment - 1) & ~(alignment - 1)
```

**핵심 설계:** 주소 할당은 `exposed_tensors`를 순회한다 (DSL 연산이 아님). Kernel 기반이므로 `configure()`가 모든 텐서 주소에 접근 가능 (R1 근거).

**Alias 텐서의 주소 해결:** alias target 텐서는 별도 주소를 할당받지 않는다. `AddressAllocator.resolve_phys_addr()`에서 alias source의 주소를 반환한다.

```python
class AddressAllocator:
    def resolve_phys_addr(self, tensor_path: str,
                          alias_registry: AliasRegistry) -> int:
        """alias된 텐서는 source의 phys_addr을 반환."""
        if alias_registry.is_alias_target(tensor_path):
            src_path = alias_registry.get_source(tensor_path)
            return self.resolve_phys_addr(src_path, alias_registry)
        return self._allocated_addrs[tensor_path]
```

---

## 11. Stage 5: auto_bind & Bank Offset Resolution

```python
def _resolve_auto_binds(self, view):
    view._register_bindings = []
    for top_iface_name in view.external_interfaces():
        for sub_name, reg, abs_offset in view.registers_for_interface(top_iface_name):
            if not reg.auto_bind:
                continue
            value = self._compute_auto_bind_value(reg.auto_bind, sub_name, view)
            view._register_bindings.append(RegisterBindingEntry(
                register_name=f"{sub_name}.{reg.name}",
                kernel_path=f"{view.name}.{sub_name}.{reg.interface_name}",
                interface_name=top_iface_name,
                absolute_offset=abs_offset,
                auto_bind=reg.auto_bind,
                resolved_value=value))

def _compute_auto_bind_value(self, bind_spec, sub_kernel_name, view):
    if bind_spec.value == 'address':
        exposed = view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)
        addr = exposed.address
        if addr is None:
            raise BindingError(f"Tensor has no address (stream interface?)")
        if bind_spec.bits:
            lo, hi = parse_bit_range(bind_spec.bits)
            return (addr >> lo) & ((1 << (hi - lo + 1)) - 1)
        return addr
    elif bind_spec.value == 'size_bytes':
        return view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)._serialized_size
    elif bind_spec.value == 'size_beats':
        exposed = view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)
        iface = view.top_spec.get_interface(exposed.top_interface)
        return exposed._serialized_size // (iface.data_width // 8)
    elif bind_spec.param:
        return view.sub_kernels[sub_kernel_name]._resolver.resolve(bind_spec.param)
    elif bind_spec.expr:
        return view.sub_kernels[sub_kernel_name]._resolver.resolve(bind_spec.expr)
```

**Bank Offset Resolution Chain 예시:**

```
auto_bind: { tensor: "src", value: address, bits: "31:0" }
  → sub_kernel = "dma_ifm", tensor_name = "src"
  → view.resolve_auto_bind_tensor("dma_ifm", "src")
  → ExposedTensor(name="ifm", origin_path="dma_ifm.src")
  → address = 0x0000_0000 (Stage 4에서 할당)
  → bits "31:0" → (addr >> 0) & 0xFFFFFFFF = 0x00000000
  → absolute_offset = bank_offset(0x000) + reg.offset(0x10) = 0x010
```

---

## 12. Stage 6: IR Lowering

### 12.0 Role Determination

Protocol × OpCode → BFM Role 매핑:

```python
def _determine_role(self, protocol: Protocol, opcode: OpCode) -> Role:
    """BFM이 마스터인지 슬레이브인지 결정.
    
    AXI4-Stream:
      PUSH → BFM MASTER (BFM이 tdata 구동, DUT가 수신)
      PULL → BFM SLAVE  (DUT가 tdata 구동, BFM이 캡처)
    
    AXI4 (Memory-Mapped):
      PUSH → BFM SLAVE (DUT=master가 읽기 요청, BFM이 응답)
      PULL → BFM SLAVE (DUT=master가 쓰기 요청, BFM이 수신)
    
    AXI4-Lite:
      항상 BFM MASTER (BFM이 레지스터 읽기/쓰기 구동)
    """
    if protocol == Protocol.AXI4L:
        return Role.MASTER

    if protocol == Protocol.AXI4S:
        return Role.MASTER if opcode == OpCode.PUSH else Role.SLAVE

    if protocol == Protocol.AXI4:
        return Role.SLAVE  # DUT가 항상 AXI master

    raise ValueError(f"Unknown protocol: {protocol}")
```

### 12.1 Buffer ID Allocation

텐서 이름 → SHM 버퍼 ID 매핑. Stage 6 시작 시 모든 exposed tensor에 순차 할당:

```python
def _allocate_buffer_ids(self, view: FlattenedKernelView) -> dict[str, int]:
    """exposed_tensors 순회 순서로 buffer_id를 0부터 순차 할당.

    Alias target 텐서는 source와 동일한 buffer_id를 받는다.
    해당 버퍼의 direction은 BIDIRECTIONAL로 승격된다.
    """
    buffer_ids = {}
    next_id = 0
    for name in view.exposed_tensors:
        if self._alias_registry.is_alias_target(name):
            # Alias target: source의 buffer_id 공유
            src_name = self._alias_registry.get_source(name)
            buffer_ids[name] = buffer_ids[src_name]
        else:
            buffer_ids[name] = next_id
            next_id += 1
    return buffer_ids
```

### 12.2 Main IR Lowering Loop

```python
def _lower_to_ir(self, view, ops):
    buffer_ids = self._allocate_buffer_ids(view)
    self._buffer_ids = buffer_ids  # 다른 lowering 메서드에서 참조

    commands = []
    next_cmd_id = 0
    op_to_cmd_range = {}  # {id(op): (first_cmd_id, last_cmd_id)}

    for op_handle_idx, op in enumerate(ops):
        first_cmd_id = next_cmd_id
        new_cmds = []

        if op.kind == OpKind.LOAD_TENSOR:
            new_cmds, next_cmd_id = self._lower_load(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.STORE_TENSOR:
            new_cmds, next_cmd_id = self._lower_store(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.PUSH_TENSOR:
            new_cmds, next_cmd_id = self._lower_push(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.PULL_TENSOR:
            new_cmds, next_cmd_id = self._lower_pull(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.WRITE_REGISTER:
            new_cmds, next_cmd_id = self._lower_write_reg(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.READ_REGISTER:
            new_cmds, next_cmd_id = self._lower_read_reg(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.POLL_REGISTER:
            new_cmds, next_cmd_id = self._lower_poll_reg(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.CONFIGURE:
            new_cmds, next_cmd_id = self._lower_configure(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.BARRIER:
            new_cmds, next_cmd_id = self._lower_barrier(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.SEND_TENSOR:
            new_cmds, next_cmd_id = self._lower_send_tensor(
                op, view, next_cmd_id, op_to_cmd_range)

        elif op.kind == OpKind.RECV_TENSOR:
            new_cmds, next_cmd_id = self._lower_recv_tensor(
                op, view, next_cmd_id, op_to_cmd_range)

        else:
            raise CompilationError(f"Unknown OpKind: {op.kind}")

        # commit_dep 반영: 확장된 마지막 커맨드에 적용
        if op.commit_dep and new_cmds:
            commit_dep_ids = self._resolve_commit_deps(op, op_to_cmd_range)
            new_cmds[-1].commit_dep = commit_dep_ids

        commands.extend(new_cmds)
        last_cmd_id = next_cmd_id - 1
        op_to_cmd_range[id(op)] = (first_cmd_id, last_cmd_id)

    return commands, buffer_ids
```

### 12.3 Individual Lowering Methods

**LOAD/STORE:**

```python
def _lower_load(self, op, view, next_cmd_id, op_to_cmd_range):
    exposed = view.exposed_tensors[op.tensor.name]
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)
    cmd = Command(
        op=OpCode.LOAD, cmd_id=next_cmd_id,
        buffer_id=self._buffer_ids[exposed.name],
        size=exposed._serialized_size,
        dep=dep_ids)
    return [cmd], next_cmd_id + 1

def _lower_store(self, op, view, next_cmd_id, op_to_cmd_range):
    exposed = view.exposed_tensors[op.tensor.name]
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)
    cmd = Command(
        op=OpCode.STORE, cmd_id=next_cmd_id,
        buffer_id=self._buffer_ids[exposed.name],
        size=exposed._serialized_size,
        dep=dep_ids)
    return [cmd], next_cmd_id + 1
```

**PUSH/PULL:**

```python
def _lower_push(self, op, view, next_cmd_id, op_to_cmd_range):
    exposed = view.exposed_tensors[op.tensor.name]
    iface = view.top_spec.get_interface(exposed.top_interface)
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)

    # 멀티포트 (array 또는 split) 처리
    if exposed._port_buffers:
        commands = []
        for port_name, port_data in exposed._port_buffers.items():
            cmd = Command(
                op=OpCode.PUSH, cmd_id=next_cmd_id,
                interface_id=self._get_iface_id(port_name),
                buffer_id=self._buffer_ids[f"{exposed.name}:{port_name}"],
                protocol=iface.protocol,
                phys_addr=exposed.address or 0,
                size=len(port_data),
                role=self._determine_role(iface.protocol, OpCode.PUSH),
                probe=op.probe,
                dep=dep_ids if not commands else [])
            commands.append(cmd)
            next_cmd_id += 1
        return commands, next_cmd_id

    cmd = Command(
        op=OpCode.PUSH, cmd_id=next_cmd_id,
        interface_id=self._get_iface_id(exposed.top_interface),
        buffer_id=self._buffer_ids[exposed.name],
        protocol=iface.protocol,
        phys_addr=exposed.address or 0,
        size=exposed._serialized_size,
        role=self._determine_role(iface.protocol, OpCode.PUSH),
        probe=op.probe,
        dep=dep_ids)
    return [cmd], next_cmd_id + 1
```

(`_lower_pull`은 `_lower_push`와 구조 동일, OpCode.PULL 사용)

### 12.4 Dependency Resolution

멀티 커맨드 확장 시 하류 의존은 확장의 **마지막** cmd_id를 참조:

```python
def _resolve_deps(self, op_deps, op_to_cmd_range):
    if not op_deps:
        return []
    cmd_deps = []
    for dep_handle in op_deps:
        op_id = id(dep_handle.op)
        _, last_cmd_id = op_to_cmd_range[op_id]
        cmd_deps.append(last_cmd_id)
    return cmd_deps
```

### 12.5 configure() Lowering

```python
def _lower_configure(self, op, view, next_cmd_id, op_to_cmd_range):
    """configure(kernel)의 IR lowering.
    
    op.kernel에 저장된 커널 참조를 사용하여 해당 커널의
    auto_bind 레지스터만 처리한다.
    
    현재 단일 primary kernel 시나리오에서는 view._register_bindings 전체를
    사용하지만, 복수 커널 확장 시 op.kernel로 필터링한다.
    """
    commands = []
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)

    # 단일 primary kernel: view._register_bindings 전체 사용
    # op.kernel 참조로 Composite 내 서브커널 범위 필터링 가능
    register_bindings = view._register_bindings
    
    if op.kernel is not None and op.kernel.name != view.name:
        # Composite 내 특정 서브커널의 auto_bind만 처리
        kernel_prefix = f"{view.name}.{op.kernel.name}"
        register_bindings = [
            b for b in view._register_bindings
            if b.kernel_path.startswith(kernel_prefix)
        ]

    for i, reg_binding in enumerate(register_bindings):
        iface_id = self._get_iface_id(reg_binding.interface_name)
        commands.append(Command(
            op=OpCode.WRITE_REG,
            cmd_id=next_cmd_id,
            interface_id=iface_id,
            protocol=Protocol.AXI4L,
            reg_offset=reg_binding.absolute_offset,
            reg_value=reg_binding.resolved_value,
            dep=dep_ids if i == 0 else [],  # 첫 커맨드만 dep 전달
        ))
        next_cmd_id += 1
    return commands, next_cmd_id
```

**첫 WRITE_REG만 dep을 갖는 이유:** 후속 WRITE_REG는 같은 AXI-Lite BFM 큐를 대상으로 하여 자연스럽게 직렬화된다.

### 12.6 Shorthand Expansion (Alias-Aware)

Shorthand(`send_tensor`, `recv_tensor`)는 alias registry를 참조하여 LOAD/STORE를 자동으로 skip한다. 사용자 DSL 코드 변경 없이 alias 선언만으로 최적화된 Command를 생성한다.

```python
def _lower_send_tensor(self, op, view, next_cmd_id, op_to_cmd_range):
    """send_tensor의 alias-aware lowering.

    - alias target (alias(src, t)가 선언됨): PUSH만 생성. LOAD skip.
      auto-dependency: src의 PULL cmd_id에 issue dep 추가.
    - alias 아님: LOAD + PUSH 생성 (기존 동작).
    """
    exposed = view.exposed_tensors[op.tensor.name]
    iface = view.top_spec.get_interface(exposed.top_interface)
    commands = []
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)

    is_alias_target = self._alias_registry.is_alias_target(exposed.name)

    # LOAD (alias target이면 skip)
    if not is_alias_target:
        load_cmd = Command(
            op=OpCode.LOAD, cmd_id=next_cmd_id,
            buffer_id=self._buffer_ids[exposed.name],
            size=exposed._serialized_size,
            dep=dep_ids)
        commands.append(load_cmd)
        load_id = next_cmd_id
        next_cmd_id += 1
        push_dep = [load_id]
    else:
        # Auto-dependency: src의 PULL cmd_id에 의존
        alias_src_name = self._alias_registry.get_source(exposed.name)
        src_write_cmd_id = self._alias_registry.last_write_cmd_id(alias_src_name)
        push_dep = dep_ids
        if src_write_cmd_id is not None:
            push_dep = list(set(push_dep + [src_write_cmd_id]))

    # PUSH
    push_cmd = Command(
        op=OpCode.PUSH, cmd_id=next_cmd_id,
        interface_id=self._get_iface_id(exposed.top_interface),
        buffer_id=self._buffer_ids[exposed.name],
        protocol=iface.protocol,
        phys_addr=exposed.address or 0,
        size=exposed._serialized_size,
        role=self._determine_role(iface.protocol, OpCode.PUSH),
        dep=push_dep)
    commands.append(push_cmd)
    next_cmd_id += 1

    return commands, next_cmd_id


def _lower_recv_tensor(self, op, view, next_cmd_id, op_to_cmd_range):
    """recv_tensor의 alias-aware lowering.

    - alias source (alias(t, dst)가 선언됨): PULL만 생성. STORE skip.
      데이터는 SHM에 남아서 downstream consumer가 사용.
    - alias 아님: PULL + STORE 생성 (기존 동작).
    """
    exposed = view.exposed_tensors[op.tensor.name]
    iface = view.top_spec.get_interface(exposed.top_interface)
    commands = []
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)

    is_alias_source = self._alias_registry.is_alias_source(exposed.name)

    # PULL
    pull_cmd = Command(
        op=OpCode.PULL, cmd_id=next_cmd_id,
        interface_id=self._get_iface_id(exposed.top_interface),
        buffer_id=self._buffer_ids[exposed.name],
        protocol=iface.protocol,
        phys_addr=exposed.address or 0,
        size=exposed._serialized_size,
        role=self._determine_role(iface.protocol, OpCode.PULL),
        dep=dep_ids)
    commands.append(pull_cmd)
    pull_id = next_cmd_id
    next_cmd_id += 1

    # PULL cmd_id를 alias registry에 기록 (auto-dependency용)
    self._alias_registry.record_write_cmd(exposed.name, pull_id)

    # STORE (alias source이면 skip; 메모리맵이 아닌 스트림도 기존대로 skip)
    if not is_alias_source and iface.protocol != Protocol.AXI4S:
        store_cmd = Command(
            op=OpCode.STORE, cmd_id=next_cmd_id,
            buffer_id=self._buffer_ids[exposed.name],
            dep=[pull_id])
        commands.append(store_cmd)
        next_cmd_id += 1

    return commands, next_cmd_id
```

**Cross-Batch alias에서의 auto-dependency:**

`ctx.run()`은 blocking이므로, 이전 Batch의 모든 Command는 이미 완료된 상태. Cross-Batch alias 시 auto-dependency injection은 불필요 — `src_write_cmd_id`가 `None`이면 (이전 Batch에서 완료되었으므로) 의존을 추가하지 않는다. 현재 Batch의 cmd_id 공간과 이전 Batch의 cmd_id 공간은 독립적이다.

### 12.7 write_register Lowering

```python
def _lower_write_reg(self, op, view, next_cmd_id, op_to_cmd_range):
    """write_register(register, {"reg_name": value, ...})의 IR lowering.
    
    fields dict의 각 key를 인터페이스의 레지스터 이름으로 해석하고,
    해당 레지스터의 absolute offset에 value를 쓰는 WRITE_REG 커맨드를 생성.
    """
    commands = []
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)
    
    iface_name = op.register_interface
    iface_id = self._get_iface_id(iface_name)
    
    for reg_name, value in op.register_fields.items():
        reg_spec, abs_offset = self._resolve_register_by_name(
            view, iface_name, reg_name
        )
        reg_value = self._encode_register_value(reg_spec, reg_name, value)
        
        commands.append(Command(
            op=OpCode.WRITE_REG,
            cmd_id=next_cmd_id,
            interface_id=iface_id,
            protocol=Protocol.AXI4L,
            reg_offset=abs_offset,
            reg_value=reg_value,
            dep=dep_ids if not commands else [],  # 첫 커맨드만 dep
        ))
        next_cmd_id += 1
    
    return commands, next_cmd_id


def _resolve_register_by_name(self, view, iface_name, reg_name):
    """인터페이스 내에서 레지스터 이름으로 검색.
    
    1차: registers[].name과 일치하는 레지스터 탐색
    2차: registers[].fields 내 키와 일치하는 필드 탐색 (필드 이름 해석)
    
    Returns: (RegisterSpec, absolute_offset)
    Raises: CompilationError
    """
    # 1차: 레지스터 이름 매칭
    for sub_name, reg_spec, abs_offset in \
            view.registers_for_interface(iface_name):
        if reg_spec.name == reg_name:
            return reg_spec, abs_offset
    
    # 2차: 필드 이름으로 재탐색
    for sub_name, reg_spec, abs_offset in \
            view.registers_for_interface(iface_name):
        if reg_spec.fields and reg_name in reg_spec.fields:
            return reg_spec, abs_offset
    
    available = [
        r.name for _, r, _ in view.registers_for_interface(iface_name)
    ]
    raise CompilationError(
        f"Register or field '{reg_name}' not found in interface "
        f"'{iface_name}'. Available registers: {available}"
    )


def _encode_register_value(self, reg_spec, key_name, value):
    """레지스터 값 인코딩.
    
    두 가지 해석 모드:
    1. key가 레지스터 이름 → value를 전체 레지스터 값으로 사용
       예: {"start": 1} (name="start") → reg_value = 1
    
    2. key가 필드 이름 → value를 해당 비트 범위에 시프트
       예: {"go": 1} (field "0:0") → reg_value = (1 << 0) = 0x1
       예: {"busy": 1} (field "1:1") → reg_value = (1 << 1) = 0x2
    """
    if key_name == reg_spec.name:
        return int(value)
    
    if reg_spec.fields and key_name in reg_spec.fields:
        hi, lo = parse_bit_range(reg_spec.fields[key_name])
        mask = ((1 << (hi - lo + 1)) - 1) << lo
        return (int(value) << lo) & mask
    
    return int(value)
```

**설계 결정:**
- **레지스터 이름 우선 해석.** `{"start": 1}`에서 `"start"`는 먼저 `registers[].name`과 매칭. 매칭 실패 시 `registers[].fields` 내 키로 재탐색.
- **Read-Modify-Write 미지원.** 동일 레지스터의 복수 필드를 개별 호출로 설정하면 나중 값이 덮어쓴다. 복수 필드를 한 번에 쓰려면 레지스터 이름으로 전체 값을 전달: `{"start": 0x03}`.

### 12.8 read_register Lowering

```python
def _lower_read_reg(self, op, view, next_cmd_id, op_to_cmd_range):
    """read_register(register, field_name)의 IR lowering.
    
    field_name에 해당하는 레지스터를 찾아 READ_REG 커맨드를 생성.
    결과는 Backend가 SHM Command Slot의 reg_value 필드에 기록.
    """
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)
    
    iface_name = op.register_interface
    iface_id = self._get_iface_id(iface_name)
    
    reg_spec, abs_offset = self._resolve_register_by_field_name(
        view, iface_name, op.register_field_name
    )
    
    cmd = Command(
        op=OpCode.READ_REG,
        cmd_id=next_cmd_id,
        interface_id=iface_id,
        protocol=Protocol.AXI4L,
        reg_offset=abs_offset,
        dep=dep_ids,
    )
    return [cmd], next_cmd_id + 1
```

### 12.9 poll_register Lowering

```python
def _lower_poll_reg(self, op, view, next_cmd_id, op_to_cmd_range):
    """poll_register(register, field_name)의 IR lowering.
    
    field_name을 레지스터의 비트 필드 정의에서 찾아 mask와 expected를 생성.
    Backend의 BFM이 (read_value & mask) == expected를 반복 확인한다.
    
    변환 예시:
        fields: { done: "0:0" }, field_name="done"
        → mask=0x1, expected=0x1 (비트 0이 1이 될 때까지 대기)
        
        fields: { status: "3:0" }, field_name="status"
        → mask=0xF, expected=0xF (비트 3:0이 모두 1이 될 때까지 대기)
    """
    dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)
    
    iface_name = op.register_interface
    iface_id = self._get_iface_id(iface_name)
    
    reg_spec, abs_offset, mask, expected = \
        self._resolve_poll_params(view, iface_name, op.register_field_name)
    
    cmd = Command(
        op=OpCode.POLL_REG,
        cmd_id=next_cmd_id,
        interface_id=iface_id,
        protocol=Protocol.AXI4L,
        reg_offset=abs_offset,
        reg_mask=mask,
        reg_expected=expected,
        dep=dep_ids,
    )
    return [cmd], next_cmd_id + 1


def _resolve_register_by_field_name(self, view, iface_name, field_name):
    """필드 이름으로 레지스터를 역탐색.
    
    인터페이스의 모든 레지스터를 순회하여 fields에 field_name이 있는
    레지스터를 찾는다.
    
    Returns: (RegisterSpec, absolute_offset)
    Raises: CompilationError
    """
    for sub_name, reg_spec, abs_offset in \
            view.registers_for_interface(iface_name):
        if reg_spec.fields and field_name in reg_spec.fields:
            return reg_spec, abs_offset
    
    available_fields = []
    for _, r, _ in view.registers_for_interface(iface_name):
        if r.fields:
            available_fields.extend(r.fields.keys())
    
    raise CompilationError(
        f"Field '{field_name}' not found in any register of interface "
        f"'{iface_name}'. Available fields: {available_fields}"
    )


def _resolve_poll_params(self, view, iface_name, field_name):
    """poll_register의 mask/expected를 비트 필드 정의에서 계산.
    
    Returns: (RegisterSpec, absolute_offset, mask, expected)
    
    기본 정책: expected = mask (해당 비트가 모두 1).
    이는 done, ready, complete 같은 상태 비트의 가장 일반적 패턴.
    커스텀 expected가 필요하면 향후 poll_register_raw() API 추가.
    """
    reg_spec, abs_offset = self._resolve_register_by_field_name(
        view, iface_name, field_name
    )
    
    bit_range_str = reg_spec.fields[field_name]  # "0:0", "3:0" 등
    hi, lo = parse_bit_range(bit_range_str)
    
    mask = ((1 << (hi - lo + 1)) - 1) << lo
    expected = mask
    
    return reg_spec, abs_offset, mask, expected
```

### 12.10 parse_bit_range 유틸리티

`auto_bind` 처리(§11 Stage 5)와 레지스터 lowering(§12.7-12.9)에서 공유하는 비트 범위 파서.

```python
def parse_bit_range(bit_range_str: str) -> tuple[int, int]:
    """비트 범위 문자열을 (hi, lo) 정수 튜플로 파싱.
    
    형식: "hi:lo" (hi >= lo)
    
    Examples:
        "0:0"   → (0, 0)    # 단일 비트
        "31:0"  → (31, 0)   # 32비트 전체
        "63:32" → (63, 32)  # 상위 32비트
        "1:1"   → (1, 1)    # 비트 1
    
    Raises: ValueError
    """
    parts = bit_range_str.strip().split(':')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid bit range '{bit_range_str}'. Expected 'hi:lo'."
        )
    hi, lo = int(parts[0]), int(parts[1])
    if hi < lo:
        raise ValueError(
            f"Invalid bit range '{bit_range_str}': hi ({hi}) < lo ({lo})."
        )
    return hi, lo
```

---

## 13. Stage 6b: BFM Configuration Synthesis

```python
def _synthesize_bfm_configs(self, view, commands):
    bfm_configs = {}
    for top_iface_name in view.external_interfaces():
        iface_spec = view.top_spec.get_interface(top_iface_name)

        if iface_spec.protocol in (Protocol.AXI4, Protocol.AXI4S):
            address_ranges = []
            for exposed in view.tensors_for_interface(top_iface_name):
                if exposed.address is not None:
                    address_ranges.append((
                        exposed.address, exposed._serialized_size,
                        self._buffer_ids[exposed.name]))
            bfm_configs[top_iface_name] = BFMConfig(
                interface_name=top_iface_name,
                protocol=iface_spec.protocol,
                data_width=iface_spec.data_width,
                role=self._determine_bfm_role(iface_spec),
                address_ranges=sorted(address_ranges))

        elif iface_spec.protocol == Protocol.AXI4L:
            bfm_configs[top_iface_name] = BFMConfig(
                interface_name=top_iface_name,
                protocol=Protocol.AXI4L,
                data_width=32, addr_width=iface_spec.addr_width,
                role="master",
                poll_interval=1, poll_timeout=100000)

    return list(bfm_configs.values())
```

---

## 14. Stage 7: SHM Packing

```python
def _pack_shm(self, view, commands, buffer_ids):
    self._validate_commands_before_packing(commands)

    shm_alloc = SHMBufferAllocator()

    # 데이터 버퍼 할당
    for name, exposed in view.exposed_tensors.items():
        bid = buffer_ids[name]
        direction = (0 if exposed.direction == Direction.HOST_TO_DEV else 1)
        shm_alloc.allocate(bid, exposed._serialized_size, direction)

    # Probe golden 버퍼
    for probe in view.probe_points:
        if probe.serialized_golden is not None:
            bid = self._next_buffer_id()
            shm_alloc.allocate(bid, len(probe.serialized_golden), 0, flags=0x01)
            probe.golden_buffer_id = bid

    # 크기 계산 및 이미지 생성
    total = calculate_shm_size(
        num_commands=len(commands),
        num_buffers=len(shm_alloc.descriptors),
        buffer_sizes=[d.size for d in shm_alloc.descriptors])

    image = bytearray(total)

    self._pack_control_header(image, commands, shm_alloc)   # §10.3 of 00

    cmd_offset = CONTROL_SIZE
    for cmd in commands:
        self._pack_command_slot(image, cmd_offset, cmd)      # §10.7 of 00
        cmd_offset += CMD_SLOT_SIZE

    stats_offset = cmd_offset
    for cmd in commands:
        if cmd.op == OpCode.LOAD:
            self._pack_stats_entry(image,
                stats_offset + cmd.cmd_id * STATS_SLOT_SIZE,
                status=COMMITTED)

    bufdesc_offset = stats_offset + len(commands) * STATS_SLOT_SIZE
    for desc in shm_alloc.descriptors:
        self._pack_buffer_descriptor(image, bufdesc_offset, desc)
        bufdesc_offset += BUF_DESC_SIZE

    # 입력 버퍼 데이터 적재
    for name, exposed in view.exposed_tensors.items():
        if exposed._serialized is not None:
            desc = shm_alloc.get_descriptor(buffer_ids[name])
            start = self._data_region_offset + desc.data_offset
            image[start:start + len(exposed._serialized)] = exposed._serialized

    # Probe golden 데이터
    for probe in view.probe_points:
        if probe.serialized_golden is not None:
            desc = shm_alloc.get_descriptor(probe.golden_buffer_id)
            start = self._data_region_offset + desc.data_offset
            image[start:start + len(probe.serialized_golden)] = probe.serialized_golden

    return bytes(image)
```

---

## 15. Unit vs Composite Unification

Unit 커널은 `_self` 서브커널 하나로 래핑하여 동일한 FlattenedKernelView 사용:

```python
def _wrap_unit_as_flat(self, kernel):
    mappings = [
        InterfaceMapping(
            sub_kernel="_self", sub_interface=iface.name,
            mapping_type=MappingType.EXTERNAL,
            top_interface=iface.name, bank_name=None, bank_offset=0)
        for iface in kernel.spec.interfaces.values()
    ]
    exposed = {}
    for tensor in kernel.tensors():
        exposed[tensor.name] = ExposedTensor(
            name=tensor.name, origin_path=f"_self.{tensor.name}",
            origin_tensor=tensor, top_interface=tensor.interface,
            direction=self._infer_direction(tensor, kernel.spec))

    return FlattenedKernelView(
        name=kernel.spec.kernel_name, top_spec=kernel.spec,
        sub_kernels={"_self": kernel},
        interface_mappings=mappings, exposed_tensors=exposed,
        probe_points=[], connections=[])
```

**근거:** Unit = 서브커널 1개, 모든 인터페이스 External, bank_offset=0인 CompositeKernel. 단일 파이프라인 코드 경로로 테스트 표면 ~50% 감소.

---

## 16. Validation Rules

### 16.1 Build-Time (Stage 0)

| 규칙 | 검사 | 에러 |
|------|------|------|
| V1: 완전 매핑 | 모든 서브커널 인터페이스가 interface_map에 존재 | ValidationError |
| V2: 프로토콜 호환 | 서브커널 인터페이스 프로토콜 = 최상위 인터페이스 프로토콜 | ProtocolMismatchError |
| V3: 뱅크 비겹침 | 같은 AXI-Lite 인터페이스의 레지스터 뱅크 주소 비겹침 | BankOverlapError |
| V4: 연결 호환 | 연결된 텐서 쌍의 원소 수 일치 (형상 해결 후) | ConnectionShapeMismatchError |

### 16.2 Compile-Time (Stages 1-7)

| 규칙 | 스테이지 | 검사 | 에러 |
|------|----------|------|------|
| V5: 파라미터 해결 | 1 | 모든 `${...}` 표현식이 정수로 평가 | ParameterResolutionError |
| V6: 형상 일치 | 2 | `prod(resolved_shape) == tensor.data.numel()` | ShapeMismatchError |
| V7: 입력 데이터 | 3 | HOST_TO_DEV 텐서에 non-None 데이터 | SerializationError |
| V8: 영역 용량 | 4 | 할당 주소가 메모리 영역 내 | MemoryOverflowError |
| V9: auto_bind 텐서 | 5 | 참조 텐서가 최상위에 노출 | BindingError |
| V10: 의존성 비순환 | 6 | 연산 그래프에 순환 없음 | DependencyError |
| V11: 의존성 한도 | 6 | ≤4 issue deps, ≤4 commit deps | DependencyLimitError |
| V12: Probe golden | 3b | 모든 probe 포인트에 golden 데이터 | ProbeError |
| V13: 레지스터 이름 존재 | 6 (write_reg) | `fields` dict의 모든 key가 인터페이스의 레지스터 이름 또는 필드 이름에 존재 | CompilationError |
| V14: Poll 필드 존재 | 6 (poll_reg) | `field_name`이 인터페이스의 어떤 레지스터의 fields에 존재 | CompilationError |
| V15: 비트 범위 유효 | 6 | `"hi:lo"` 형식이며 `hi >= lo` | ValueError |
