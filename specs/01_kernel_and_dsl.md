# vTen Kernel Abstraction & DSL Operations

**Version 0.5.0 — March 2026**
**참조 모델: `00_data_models.md` (Tensor, Kernel, CompositeKernel, OpKind, register())**
**소스: 메인 스펙 §3-5, §10 + 서플리먼트 §18 (errata)**

---

## Table of Contents

1. [RTL Interface Specification (kernel_spec.yaml)](#1-rtl-interface-specification)
2. [Kernel Class](#2-kernel-class)
3. [DSL Operations](#3-dsl-operations)
4. [Kernel Composition](#4-kernel-composition)

---

## 1. RTL Interface Specification (kernel_spec.yaml)

### 1.1 Zero RTL Intrusion

모든 의미론적 매핑(텐서↔포트 바인딩, dtype, 패킹)은 `kernel_spec.yaml`에 존재한다. RTL 소스는 수정하지 않는다. CLI의 `vten spec --detect rtl/top.sv`가 YAML 스켈레톤을 자동 생성하며, 사용자는 의미론 정보만 채운다.

### 1.2 kernel_spec.yaml 전체 예시

```yaml
kernel: conv3d_top
rtl_top: rtl/conv3d_top.sv

parameters:
  C: "${C}"                      # project/runtime 스코프에서 해결
  H: 32
  K: 128

memory_regions:
  ddr:
    base: 0x0000_0000
    size: 0x1_0000_0000          # 4GB
    alignment: 4096              # AXI burst 정렬

interfaces:
  ifm_stream:
    rtl_port: s_axis_ifm
    protocol: axi4_stream
    tensor: ifm
    packing:
      element_width: 8
      elements_per_beat: 32
      bit_order: lsb_first

  data_port:
    rtl_port: m_axi_data
    protocol: axi4
    data_width: 256
    addr_width: 64
    memory_region: ddr
    tensors: [ifm, weight, ofm]
    packing:
      element_width: 8
      elements_per_beat: 32
      alignment: packed

  ctrl:
    rtl_port: s_axilite_ctrl
    protocol: axi4_lite
    addr_width: 32
    registers:
      - name: start
        offset: 0x00
        fields: { go: "0:0" }
      - name: status
        offset: 0x04
        fields: { done: "0:0", busy: "1:1" }
      - name: ifm_base_lo
        offset: 0x10
        auto_bind: { tensor: ifm, value: address, bits: "31:0" }
      - name: ifm_base_hi
        offset: 0x14
        auto_bind: { tensor: ifm, value: address, bits: "63:32" }
      - name: weight_base_lo
        offset: 0x18
        auto_bind: { tensor: weight, value: address, bits: "31:0" }
      - name: weight_base_hi
        offset: 0x1C
        auto_bind: { tensor: weight, value: address, bits: "63:32" }
      - name: ofm_base_lo
        offset: 0x20
        auto_bind: { tensor: ofm, value: address, bits: "31:0" }
      - name: ofm_base_hi
        offset: 0x24
        auto_bind: { tensor: ofm, value: address, bits: "63:32" }
      - name: transfer_size
        offset: 0x28
        auto_bind: { tensor: ifm, value: size_beats }
      - name: input_ch
        offset: 0x30
        auto_bind: { param: "${C}" }

  ofm_hbm:
    rtl_port: m_axi_ofm
    protocol: axi4
    data_width: 256
    tensor: ofm
    split:
      mode: channel_interleave
      ports:
        - { name: hbm_ch0, base_addr: 0x00000000 }
        - { name: hbm_ch1, base_addr: 0x00000000 }
      interleave:
        unit: 4096
```

YAML 필드별 상세 스키마, 파싱 규칙, 검증 로직은 `03_kernel_spec_schema.md` 참조.

### 1.3 Packing Specification

스트림 인터페이스의 비트 레벨 패킹 정의. Runtime 직렬화기가 처리하며 Kernel 클래스에는 투명.

| 필드 | 타입 | 설명 |
|------|------|------|
| `element_width` | int (비트) | 각 데이터 원소의 비트 폭 |
| `elements_per_beat` | int | 버스 비트당 패킹되는 원소 수 |
| `bit_order` | `lsb_first` \| `msb_first` | 비트 내 배치 순서 |
| `alignment` | `packed` \| `aligned` | packed: 갭 없음; aligned: 바이트 경계 정렬 |
| `byte_order` | `little` \| `big` | 직렬화된 비트의 바이트 엔디안 |
| `mode: custom` | fields 리스트 | 혼합 필드 비트: 각 필드에 이름과 비트 범위 |

예시 — 24비트 버스에 4×6비트 원소, LSB-first:

```yaml
packing:
  element_width: 6
  elements_per_beat: 4
  bit_order: lsb_first
```

커스텀 필드(혼합 타입 비트):

```yaml
packing:
  mode: custom
  fields:
    - { name: data_a, bits: [0, 23] }
    - { name: data_b, bits: [24, 47] }
    - { name: valid_mask, bits: [48, 49] }
    - { name: reserved, bits: [50, 63] }
```

### 1.4 Multi-Port Split

단일 논리 텐서가 복수 물리 포트(예: HBM 채널)에 분산될 때의 분배 전략. Runtime이 직렬화된 데이터를 자동 분할하고 포트별 별도 IR 커맨드를 생성한다. DSL 사용자는 단일 `push_tensor`/`pull_tensor`만 작성.

- `channel_interleave`: 고정 단위 크기로 라운드 로빈 (예: 4KB 청크)
- `block_split`: 텐서 차원 또는 바이트 범위 기준 연속 분할

---

## 2. Kernel Class

### 2.1 구조

```python
from vten import Kernel, Tensor, register

class Conv3DKernel(Kernel):
    spec = "specs/conv3d_top.yaml"

    # ── 파라미터 기본값 ──
    default_params = {
        "N": 1, "C": 32, "D": 4, "H": 4, "W": 4, "K": 64,
    }

    # ── 텐서 선언 (logical shape, logical dtype) ──
    ifm = Tensor(
        shape=("${N}", "${C}", "${D}", "${H}", "${W}"),
        dtype=torch.int8,
        interface="ifm"
    )
    weight = Tensor(
        shape=("${K}", "${C}", 3, 3, 3),
        dtype=torch.int8,
        interface="weight_mem"
    )
    ofm = Tensor(
        shape=("${N}", "${K}", "${D}", "${H}", "${W}"),
        dtype=torch.int32,
        interface="ofm_hbm"
    )

    ctrl = register("ctrl")

    # ── 파생 파라미터 (선택적, instance method) ──
    def compute_derived_params(self) -> dict:
        return {"total_elements": self.N * self.C * self.D * self.H * self.W}

    # ── 입력 생성 (자유 형식 Python) ──
    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.ifm.fill_random(generator=rng)
        self.weight.fill_random(generator=rng)

    # ── Golden reference (logical 공간) ──
    def forward(self, ifm, weight) -> dict[str, torch.Tensor]:
        """인자명 = input tensor 이름, 반환값 = {output tensor 이름: logical data}"""
        return {"ofm": F.conv3d(ifm.to_float(), weight.to_float())}
```

### 2.2 Tiling Responsibility

레이아웃 타일링(예: NCDHW → N,C//32,D,H,W,32)은 `generate_inputs()`의 책임이며 직렬화 런타임의 책임이 아니다.

**근거:**
- 타일링 패턴은 가속기마다 다르며 기능적 참조 모델과 깊이 결합됨
- 사용자 코드에 유지하면 임의 재배열에 대한 최대 유연성 확보
- Runtime은 총 원소 수가 선언 형상과 일치하는지만 검증

```python
def generate_inputs(self, seed=None):
    raw = torch.randn(self.N, self.C, self.D, self.H, self.W)
    C2 = 32
    tiled = raw.reshape(self.N, self.C//C2, C2, self.D, self.H, self.W)
    tiled = tiled.permute(0, 1, 3, 4, 5, 2).contiguous()
    self.ifm.data = tiled  # 직렬화기가 타일링된 형태를 수신
```

### 2.3 Complex Reference Models

다단계 파이프라인, 복잡한 전처리, 데이터셋 기반 입력을 가진 가속기의 경우 Kernel 메서드는 제약 없는 Python이다:

```python
class ComplexPipelineKernel(Kernel):
    spec = "specs/pipeline.yaml"
    ifm = Tensor(...)

    def generate_inputs(self, seed=None):
        raw = load_from_dataset("imagenet", index=self.test_id)
        preprocessed = custom_preprocessing(raw)
        quantized = quantize(preprocessed, bits=8, scale=self.scale)
        C2 = 32
        tiled = quantized.reshape(N, C//C2, C2, D, H, W)
        self.ifm.data = tiled.permute(0,1,3,4,5,2).contiguous()

    def forward(self, ifm, weight) -> dict[str, torch.Tensor]:
        x = ifm.to_float()
        x = custom_padding(x, mode='replicate')
        x = F.conv3d(x, weight.to_float())
        x = batch_norm_reference(x, self.bn_params)
        return {"ofm": F.relu(x)}
```

---

## 3. DSL Operations

### 3.1 연산 분류

원본 논문의 `write_tensor`/`read_tensor`는 메모리맵 인터페이스에서 모호하다. 가속기가 버스 마스터인 경우 호스트↔메모리 전송과 가속기↔메모리 상호작용을 구분하는 2-레벨 분류를 사용한다.

| 레벨 | 연산 | 의미 | MM (AXI4) 동작 | Stream 동작 |
|------|------|------|----------------|-------------|
| L1: Host↔Mem | `load_tensor` | Host → Device Memory | SHM 적재 | N/A |
| | `store_tensor` | Device Memory → Host | SHM 읽기 | N/A |
| L2: Accel↔Mem | `push_tensor` | Accel이 텐서 소비 | BFM 슬레이브가 accel 읽기에 응답 | BFM이 데이터 구동 |
| | `pull_tensor` | Accel이 텐서 생성 | BFM 슬레이브가 accel 쓰기 수신 | BFM이 데이터 캡처 |
| L3: Control | `write_register` | 레지스터 설정 | AXI-Lite 쓰기 | — |
| | `read_register` | 레지스터 조회 | AXI-Lite 읽기 | — |
| | `poll_register` | 조건 충족까지 블록 | AXI-Lite 반복 읽기 | — |
| | `configure` | auto_bind 전체 레지스터 쓰기 | 모든 auto_bind WRITE_REG | — |
| | `barrier` | 전역 동기화 펜스 | 모든 진행 중 연산 대기 | — |
| Shorthand | `send_tensor` | = load + push (자동, alias 시 push만) | 두 단계 모두 | push만 |
| | `recv_tensor` | = pull + store (자동, alias source 시 pull만) | 두 단계 모두 | pull만 |
| | `alias` | 버퍼 재사용 선언 (Invocation 간) | 동일 | 동일 |
| | `config_boundary` | multi-config 그룹 경계 표시 | BARRIER 삽입 + config_group 증가 | 동일 |

### 3.2 Dependency Model

각 연산에는 두 수명주기 이벤트가 있다: **Issue** (DUT에 디스패치)와 **Commit** (DUT 실행 완료).

1. **Sync/Async 모드**: Async는 즉시 디스패치 허용 (Issue_n ≈ Issue_{n+1}). Sync는 직렬 실행 강제 (Commit_n ≺ Issue_{n+1}).
2. **Issue Dependencies**: 세밀한 제어: Commit_src ≺ Issue_dst. `dep=` 파라미터로 표현.
3. **Commit Dependencies**: 종료 경계: Commit_dst ≤ Commit_src. `add_commit_dependency()`로 표현.
4. **Barrier**: 전역 펜스. 이전 모든 연산이 commit된 후에야 후속 issue 가능.

### 3.3 사용 예시: Memory-Mapped Interface

가속기가 AXI 마스터인 경우의 전체 패턴:

```python
def run(self, ctx, cfg):
    kernel = ctx.instantiate("conv3d_top", **cfg)
    kernel.generate_inputs(seed=42)

    # Phase 1: Host → Device Memory
    l1 = ctx.load_tensor(kernel.ifm)
    l2 = ctx.load_tensor(kernel.weight)

    # Phase 2: Auto-configure (주소 레지스터, 형상 레지스터, BFM 주소 맵)
    ctx.configure(kernel)

    # Phase 3: 가속기 시작
    w0 = ctx.write_register(kernel.ctrl, {"start": 1}, dep=[l1, l2])

    # Phase 4: Accel이 메모리에서 읽고, 계산하고, 결과 기록
    push1 = ctx.push_tensor(kernel.ifm, dep=w0)
    push2 = ctx.push_tensor(kernel.weight, dep=w0)
    pull1 = ctx.pull_tensor(kernel.ofm, dep=[push1, push2])

    # Phase 5: done 대기
    p1 = ctx.poll_register(kernel.ctrl, "done", dep=w0)
    pull1.add_commit_dependency(p1)

    # Phase 6: Device Memory → Host + 검증
    s1 = ctx.store_tensor(kernel.ofm, dep=pull1)
    ctx.verify(s1)  # golden 자동 계산 (forward(**inputs) 호출)
```

### 3.4 사용 예시: Stream Interface (Shorthand)

```python
def run(self, ctx, cfg):
    kernel = ctx.instantiate("conv3d_top", **cfg)
    kernel.generate_inputs(seed=42)

    w0 = ctx.write_register(kernel.ctrl, {"start": 1})
    push1 = ctx.push_tensor(kernel.ifm, dep=w0)
    pull1 = ctx.pull_tensor(kernel.ofm, dep=push1)
    ctx.verify(pull1)  # golden 자동 계산
```

### 3.5 configure() 동작

`ctx.configure(kernel)`은 다음을 트리거한다:

1. 모든 텐서에 대한 주소 할당 (미할당 시)
2. 모든 `auto_bind` 레지스터에 대한 WRITE_REG IR 생성
3. BFM 주소 맵 구축

```python
def run(self, ctx, cfg):
    kernel = ctx.instantiate("conv3d_top", **cfg)
    kernel.generate_inputs(seed=42)

    ctx.load_tensor(kernel.ifm)
    ctx.load_tensor(kernel.weight)
    ctx.configure(kernel)       # ← auto_bind 레지스터 여기서 쓰기
    ctx.write_register(kernel.ctrl, {"start": 1})
    # ...
```

주소 강제 설정도 가능:

```python
kernel.ifm.set_address(0xDEAD_0000)
ctx.configure(kernel)  # 강제 설정된 주소 사용
```

### 3.6 auto_bind 값 유형

| 값 | 설명 |
|----|------|
| `address` | 텐서의 할당된 물리 주소. `bits`로 64비트 분할 (예: "31:0", "63:32") |
| `size_bytes` | 직렬화된 총 크기 (바이트) |
| `size_beats` | 버스 비트 단위 크기 (= size_bytes / (data_width / 8)) |
| `size_elements` | 총 원소 수 |
| `param` | 해결된 파라미터 값 (문자열 표현식) |
| `expr` | 파라미터에 대한 임의 산술 표현식 |

### 3.7 Buffer Aliasing

다중 Invocation을 순차 실행할 때, 이전 Invocation의 출력 버퍼를 다음 Invocation의 입력으로 직접 재사용한다. Host↔Device 왕복(STORE + re-LOAD)을 제거하여 성능을 극대화한다.

**용어:** Invocation, Batch 등의 정의는 `00_data_models.md §2 Terminology Hierarchy` 참조.

#### 3.7.1 `ctx.alias()` API

```python
def alias(self, src: TensorRef, dst: TensorRef) -> None:
    """dst가 src의 SHM 버퍼를 직접 재사용하도록 선언.

    Parameters:
        src: 이전 Invocation의 출력 텐서 (보통 recv_tensor 대상).
        dst: 이후 Invocation의 입력 텐서 (보통 send_tensor 소스).

    Effects:
        - dst는 src와 동일한 buffer_id를 받는다 (새 SHM 할당 없음).
        - IR lowering 시 자동으로:
          - recv_tensor(src): STORE를 건너뜀 (src가 alias source이므로).
          - send_tensor(dst): LOAD를 건너뜀 (dst가 alias target이므로).
        - Auto-dependency: dst를 소비하는 첫 Command는 src를 쓴
          마지막 Command(PULL)에 자동 의존.
        - Buffer Descriptor direction이 BIDIRECTIONAL로 승격.
        - Memory-mapped 인터페이스: dst의 auto_bind 주소 레지스터에
          src의 phys_addr이 할당됨 (동일 물리 주소).

    Constraints:
        - dst.size_bytes <= src.size_bytes (빌드 타임 검증).
        - src와 dst의 packing이 호환되어야 함 (동일 프로토콜 시
          bus_width와 element_width 일치).
        - dtype이 비호환이면 AliasError. Host Transform 패턴 사용 (§3.8.4).

    Raises:
        AliasError (00_data_models.md §12)
    """
```

#### 3.7.2 Alias-Aware 동작 (send_tensor / recv_tensor)

기존 `send_tensor`, `recv_tensor` API는 변경 없음. IR lowering 시 alias registry를 참조하여 자동으로 최적화한다.

| 상황 | send_tensor(t) | recv_tensor(t) |
|------|----------------|----------------|
| t가 alias target | PUSH만 (LOAD skip) | — |
| t가 alias source | — | PULL만 (STORE skip) |
| t가 alias 없음 | LOAD + PUSH (기존) | PULL + STORE (기존) |

사용자는 항상 `send_tensor`/`recv_tensor`만 사용한다. alias 여부에 따른 Command 생성은 Runtime이 결정한다.

#### 3.7.3 Multiple Consumers (Fan-out)

하나의 source 버퍼를 여러 downstream에 alias 가능:

```python
# Residual connection: layer0 output → layer1, layer2
ctx.alias(layer0.ofm, layer1.ifm)    # consumer 1
ctx.alias(layer0.ofm, layer2.skip)   # consumer 2
```

두 PUSH 모두 동일 `buffer_id` 참조. 읽기 전용이므로 충돌 없음.

#### 3.7.4 Build-Time Validation

```python
def validate_alias(src: TensorBinding, dst: TensorBinding):
    assert dst.size_bytes <= src.size_bytes  # 크기 제약
    if src.protocol == dst.protocol:
        assert src.packing.bus_width == dst.packing.bus_width
        assert src.packing.element_width == dst.packing.element_width
```

### 3.8 Multi-Invocation Patterns

`ctx.run()`을 여러 번 호출하여 다중 Batch를 생성하거나, alias를 사용하여 단일 Batch에 여러 Invocation을 포함할 수 있다. `ctx.run()` 확장 의미론은 `02_runtime_engine.md §3`을 참조.

#### 3.8.1 Pattern 1: Single Batch (alias only)

모든 Invocation을 단일 Batch에 포함. 최대 성능. Host↔Backend 핸드셰이크 1회.

```python
def run(self, ctx, cfg):
    layers = [ctx.instantiate("conv3d_top", **LAYER_CONFIGS[i]) for i in range(5)]
    for i, layer in enumerate(layers):
        layer.weight.fill_random(seed=100 + i)

    # ── Invocation 0: 외부 입력 ──
    layers[0].ifm.fill_random(seed=0)
    ctx.send_tensor(layers[0].ifm)
    ctx.send_tensor(layers[0].weight)
    ctx.configure(layers[0])
    w = ctx.write_register(layers[0].ctrl, {"start": 1})
    recv0 = ctx.recv_tensor(layers[0].ofm, dep=w)
    poll = ctx.poll_register(layers[0].ctrl, "done", dep=w)
    recv0.add_commit_dependency(poll)

    # ── Invocations 1–4: alias chaining ──
    for i in range(1, 5):
        ctx.alias(layers[i - 1].ofm, layers[i].ifm)
        ctx.send_tensor(layers[i].weight)
        ctx.configure(layers[i])
        w = ctx.write_register(layers[i].ctrl, {"start": 1}, dep=recv0)
        ctx.send_tensor(layers[i].ifm, dep=recv0)  # aliased → PUSH only
        recv0 = ctx.recv_tensor(layers[i].ofm, dep=w)
        poll = ctx.poll_register(layers[i].ctrl, "done", dep=w)
        recv0.add_commit_dependency(poll)

    # ── 실행 ──
    ctx.run()

    # ── Golden ──
    golden = layers[0].forward(ifm=layers[0].ifm.logical_data,
                                weight=layers[0].weight.logical_data)["ofm"]
    for i in range(1, 5):
        golden = layers[i].forward(ifm=golden,
                                    weight=layers[i].weight.logical_data)["ofm"]
    ctx.verify(recv0, golden)
```

#### 3.8.2 Pattern 2: Chunked Batch (SHM 용량 관리)

동시 활성 버퍼가 `/dev/shm` 한계를 초과할 때. Cross-Batch alias로 경계를 넘어 데이터 전달.

```python
def run(self, ctx, cfg):
    layers = [ctx.instantiate("conv3d_top", **LAYER_CONFIGS[i]) for i in range(10)]
    for i, layer in enumerate(layers):
        layer.weight.fill_random(seed=100 + i)
    layers[0].ifm.fill_random(seed=0)

    CHUNK_SIZE = 5
    prev_recv = None

    for chunk_start in range(0, 10, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, 10)

        for i in range(chunk_start, chunk_end):
            if i == 0:
                ctx.send_tensor(layers[0].ifm)
            else:
                ctx.alias(layers[i - 1].ofm, layers[i].ifm)
                ctx.send_tensor(layers[i].ifm,
                                dep=prev_recv if i > chunk_start else None)

            ctx.send_tensor(layers[i].weight)
            ctx.configure(layers[i])
            w = ctx.write_register(layers[i].ctrl, {"start": 1},
                                   dep=prev_recv if i > chunk_start else None)
            prev_recv = ctx.recv_tensor(layers[i].ofm, dep=w)
            poll = ctx.poll_register(layers[i].ctrl, "done", dep=w)
            prev_recv.add_commit_dependency(poll)

        ctx.run()  # chunk마다 Batch 제출, Data Region 보존

    golden = compute_golden_chain(layers)  # forward(**inputs) chain
    ctx.verify(prev_recv, golden)
```

Host↔Backend 핸드셰이크: ceil(10/5) = 2회.

#### 3.8.3 Pattern 3: Per-Invocation Batch (동적 제어 흐름)

각 Invocation을 별도 Batch로. Host가 중간 결과를 검사하여 분기 가능.

```python
def run(self, ctx, cfg):
    layers = [ctx.instantiate("conv3d_top", **LAYER_CONFIGS[i]) for i in range(5)]
    for i, layer in enumerate(layers):
        layer.weight.fill_random(seed=100 + i)
    layers[0].ifm.fill_random(seed=0)
    golden = None

    for i in range(5):
        if i == 0:
            ctx.send_tensor(layers[0].ifm)
        else:
            ctx.alias(layers[i - 1].ofm, layers[i].ifm)
            ctx.send_tensor(layers[i].ifm)    # cross-Batch alias → PUSH only

        ctx.send_tensor(layers[i].weight)
        ctx.configure(layers[i])
        w = ctx.write_register(layers[i].ctrl, {"start": 1})
        result = ctx.recv_tensor(layers[i].ofm, dep=w)
        poll = ctx.poll_register(layers[i].ctrl, "done", dep=w)
        result.add_commit_dependency(poll)

        stats = ctx.run()

        hw_output = result.to_tensor()
        if golden is None:
            golden = layers[i].forward(
                ifm=layers[i].ifm.logical_data,
                weight=layers[i].weight.logical_data)["ofm"]
        else:
            golden = layers[i].forward(
                ifm=golden,
                weight=layers[i].weight.logical_data)["ofm"]

        if compute_confidence(hw_output) > cfg["early_exit_threshold"]:
            ctx.verify(result, golden)
            return

    ctx.verify(result, golden)
```

#### 3.8.4 Pattern 4: Host Transform (강제 Batch 분리)

dtype 변환 등 host-side 처리가 필요할 때. alias 불가 — 데이터가 Host를 경유해야 함.

```python
def run(self, ctx, cfg):
    layer0 = ctx.instantiate("conv3d_top", **cfg["layer0"])  # ofm: int32
    layer1 = ctx.instantiate("conv3d_top", **cfg["layer1"])  # ifm: int8

    # ── Batch 0 ──
    ctx.send_tensor(layer0.ifm)
    ctx.send_tensor(layer0.weight)
    ctx.configure(layer0)
    w = ctx.write_register(layer0.ctrl, {"start": 1})
    result0 = ctx.recv_tensor(layer0.ofm, dep=w)
    poll = ctx.poll_register(layer0.ctrl, "done", dep=w)
    result0.add_commit_dependency(poll)
    ctx.run()

    # ── Host transform: int32 → int8 requantize ──
    ofm_int32 = result0.to_tensor()
    scale = ofm_int32.abs().max() / 127.0
    ifm_int8 = (ofm_int32.float() / scale).round().clamp(-128, 127).to(torch.int8)
    layer1.ifm.data = ifm_int8

    # ── Batch 1 ──
    ctx.send_tensor(layer1.ifm)       # NOT aliased → LOAD + PUSH
    ctx.send_tensor(layer1.weight)
    ctx.configure(layer1)
    w = ctx.write_register(layer1.ctrl, {"start": 1})
    result1 = ctx.recv_tensor(layer1.ofm, dep=w)
    poll = ctx.poll_register(layer1.ctrl, "done", dep=w)
    result1.add_commit_dependency(poll)
    ctx.run()

    golden1 = compute_requantized_golden(layer0, layer1, scale)  # forward(**inputs) chain
    ctx.verify(result1, golden1)
```

#### 3.8.5 Pattern Selection Guide

```
Q: 중간 결과에 대한 Host 개입이 필요한가?
├─ No  → Q: 모든 Invocation이 SHM에 동시 적재 가능한가?
│        ├─ Yes → Pattern 1 (Single Batch)
│        └─ No  → Pattern 2 (Chunked Batch)
└─ Yes → Q: 어떤 종류의 Host 개입인가?
         ├─ 동적 제어 흐름 (결과 기반 분기)      → Pattern 3
         ├─ 데이터 변환 (dtype 변환)              → Pattern 4
         └─ 디버그 검증 (per-Invocation golden)   → Pattern 3 + ctx.verify()
```

---

## 4. Kernel Composition

### 4.1 문제: Unit Test vs Integration Test

단위 테스트에서는 모든 인터페이스에 BFM이 붙는다. 합성 설계에서는 일부 인터페이스가 BFM 없는 내부 RTL 와이어다.

```
Unit Test (MACKernel):
  BFM_ifm ↔ [MAC.ifm_port]      ← BFM 필요
  BFM_weight ↔ [MAC.weight_port]  ← BFM 필요
  [MAC.ofm_port] ↔ BFM_ofm      ← BFM 필요

Composed (NPUTop):
  BFM_ddr ↔ [DMA_ifm] ──wire──→ [MAC.ifm_port]     ← 내부, BFM 없음
  BFM_ddr ↔ [DMA_weight] ──wire──→ [MAC.weight_port]  ← 내부
  [MAC.ofm_port] ──wire──→ [DMA_ofm] ↔ BFM_ddr     ← 내부
```

**핵심:** `connections`는 **이미 존재하는 RTL 내부 와이어**를 서술한다. vTen이 와이어를 생성하지 않으며, 어떤 인터페이스에 BFM이 붙고 어떤 것이 내부인지만 결정한다.

### 4.2 Sub-kernel 선언 — `bind()` 없이

v2에서는 `interface_map`과 `bind()` 메커니즘이 삭제된다. 서브커널은 단순히 인스턴스로 선언하고, `connections` 리스트와 자동 추론 규칙으로 Internal/External을 결정한다.

```python
# v1 (삭제됨):
# dma_ifm = DMAKernel.bind(interface_map={...})

# v2:
dma_ifm = DMAKernel()  # 인스턴스 선언만
```

### 4.3 CompositeKernel 정의

```python
class DMAKernel(Kernel):
    spec = "specs/dma.yaml"
    src = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="axi_master")
    dst = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="stream_out")
    ctrl = register("ctrl_lite")

    default_params = {"SIZE": 1024}

    def forward(self, src) -> dict[str, torch.Tensor]:
        return {"dst": src.clone()}

class MACKernel(Kernel):
    spec = "specs/mac.yaml"
    ifm = Tensor(..., interface="axis_ifm")
    weight = Tensor(..., interface="axis_weight")
    ofm = Tensor(..., interface="axis_ofm")
    ctrl = register("ctrl_lite")

    def forward(self, ifm, weight) -> dict[str, torch.Tensor]:
        return {"ofm": F.conv3d(ifm.to_float(), weight.to_float())}
```

```python
class NPUTopKernel(CompositeKernel):
    spec = "specs/npu_top.yaml"

    # ── 서브커널 선언 (인스턴스만) ──
    dma_ifm = DMAKernel()
    dma_weight = DMAKernel()
    mac = MACKernel()
    dma_ofm = DMAKernel()

    # ── 내부 연결: >> 연산자로 Connection 생성 ──
    connections = [
        dma_ifm.dst   >> mac.ifm,
        dma_weight.dst >> mac.weight,
        mac.ofm       >> dma_ofm.src,
    ]

    # ── forward() 자동 chain ──
    # connections graph 기반으로 framework가 자동으로 forward chain을 수행한다.
    # 오버라이드 없이도 CompositeKernel.forward()가 multi-round
    # dataflow evaluation으로 sub-kernel forward()를 순서대로 호출한다.
    # 사용자가 원하면 오버라이드 가능.
```

**자동 추론 규칙:**

| v1 수동 | v2 자동 규칙 |
|---------|------------|
| `interface_map={"stream_out": Internal()}` | `connections`에 등장하는 tensor의 interface → Internal |
| `interface_map={"axi_master": "ddr_port"}` | `connections`에 없는 tensor의 interface → auto expose |
| `register("ctrl")` 뱅크 매핑 | sub-kernel의 register interface → `{sub_name}_{iface_name}` 자동 prefix |
| `.expose("ddr_port")` | 자동 expose (이름 = `{sub_name}_{tensor_name}` 또는 원본 유지) |

위 예시에서 `connections`에 등장하지 않는 텐서 — `dma_ifm.src`, `dma_weight.src`, `dma_ofm.dst`, 각 `ctrl` — 는 자동으로 최상위에 expose된다.

### 4.4 `>>` 연산자와 Connection

`connections`는 RTL 와이어를 **생성하지 않는다**. 이미 존재하는 RTL 내부 와이어를 서술한다.

#### 4.4.1 TensorRef와 `>>` 연산자

CompositeKernel class body에서 `sub_kernel.tensor_name`에 접근하면 `TensorRef` 객체가 반환된다. `>>` 연산자로 두 TensorRef를 연결하여 `Connection` 객체를 생성한다:

```python
class TensorRef:
    """Sub-kernel tensor reference."""
    def __init__(self, sub_kernel_name: str, tensor_name: str, kernel_class: type):
        self.sub_kernel_name = sub_kernel_name
        self.tensor_name = tensor_name
        self.kernel_class = kernel_class

    def __rshift__(self, other: TensorRef) -> Connection:
        return Connection(source=self, dest=other)
```

`SubKernelRef` (서브커널 인스턴스 선언)가 `__getattr__`로 TensorRef를 반환한다:

```python
class SubKernelRef:
    """CompositeKernel class body에서 sub-kernel 참조."""
    def __init__(self, kernel_class: type):
        self.kernel_class = kernel_class

    def __set_name__(self, owner, name):
        self._attr_name = name

    def __getattr__(self, name) -> TensorRef:
        attr = getattr(self.kernel_class, name, None)
        if isinstance(attr, Tensor):
            return TensorRef(self._attr_name, name, self.kernel_class)
        raise AttributeError(...)
```

#### 4.4.2 connections의 역할

1. **Internal/External 결정**: connections에 등장하는 tensor → Internal (BFM 없음). 등장하지 않는 tensor → auto expose (BFM 생성).
2. **Golden 모델 체이닝**: `forward()` 데이터 흐름 순서 결정 + golden chain pool 저장 (선언적 probe의 golden 소스).
3. **Probe 포인트**: 모든 Internal 연결에 패시브 probe BFM 자동 생성. 선언적 `probes` API로 런타임 활성화.
4. **문서화**: RTL 내부 구조를 Python으로 서술.
5. **검증**: 런타임이 연결 쌍의 호환 형상/dtype/protocol 확인.

#### 4.4.3 CompositeKernel 내부 속성

`__init_subclass__`에서 자동으로 구성되는 내부 속성:

| 속성 | 타입 | 설명 |
|------|------|------|
| `_sub_kernel_refs` | `dict[str, SubKernelRef]` | 서브커널 이름 → SubKernelRef 매핑 |
| `_connections` | `list[Connection]` | 선언된 Connection 목록 |
| `_connected_tensors` | `set[tuple[str, str]]` | connections에 등장하는 `(sub_name, tensor_name)` 집합 |
| `_auto_exposed` | `dict[str, ExposedTensor]` | connections에 없는 tensor → 자동 expose 결과 |

### 4.5 Register Bank Offset

복수 서브커널이 하나의 AXI-Lite 인터페이스를 공유할 때, 각각 다른 오프셋 범위를 점유:

```yaml
# npu_top kernel_spec.yaml
interfaces:
  ctrl:
    rtl_port: s_axilite_ctrl
    protocol: axi4_lite
    register_banks:
      dma_ifm:    { base_offset: 0x000 }
      dma_weight: { base_offset: 0x100 }
      mac:        { base_offset: 0x200 }
      dma_ofm:    { base_offset: 0x300 }
      global:     { base_offset: 0x400 }
```

DMAKernel의 오프셋 `0x00` 레지스터 → `dma_ifm` 뱅크에서 `0x000 + 0x00 = 0x000`, `dma_weight` 뱅크에서 `0x100 + 0x00 = 0x100`.

### 4.6 Address Space Unification

복수 서브커널이 같은 최상위 AXI 인터페이스(`"ddr_port"`)에 매핑될 때:

1. 모든 텐서가 **동일 메모리 영역**에서 `AddressAllocator`로 할당
2. **단일 BFM**이 복수 주소 범위로 생성
3. 각 서브커널의 `auto_bind` 레지스터가 뱅크 오프셋을 통해 올바른 주소를 받음

```
ddr_port 메모리 레이아웃:
  0x0000_0000 ~ 0x0001_FFFF  →  dma_ifm.src (IFM)
  0x0002_0000 ~ 0x0002_7FFF  →  dma_weight.src (Weight)
  0x0004_0000 ~ 0x0005_FFFF  →  dma_ofm.dst (OFM)

단일 BFM이 세 범위 모두 서비스.
```

### 4.7 테스트 레벨에서의 합성

커널이 `run(self, ctx)` 메서드를 구현하면, TestScenario는 `kernel`과 선택적 `probes`만 선언하면 된다:

```python
class TestNPUTop(TestScenario):
    kernel = "npu_top"
    # default run: ki.run(ctx) 자동 호출

class TestNPUTopProbe(TestScenario):
    kernel = "npu_top"
    probes = ["mac.axis_ifm", "mac.axis_ofm", "data_out"]
    # 커널 정의 변경 없이 probe 적용
    # 모든 internal 연결(connections)에 probe BFM 자동 생성
    # golden chain pool에서 golden 자동 추출
```

`probes` 리스트의 각 항목은:
- `"data_out"`: 출력 텐서의 PULL op에 `probe=True` 자동 적용
- `"mac.axis_ifm"`: 서브커널 내부 텐서. INTERNAL → INTERNAL_PROBE 동적 업그레이드, golden chain에서 golden 자동 추출

**수동 run() override도 여전히 가능하다:**

```python
class TestNPUTopCustom(TestScenario):
    kernel = "npu_top"

    def run(self, ctx, cfg):
        npu = ctx.instantiate("npu_top", **cfg)
        npu.generate_inputs(seed=42)

        l1 = ctx.load_tensor(npu.ifm)
        l2 = ctx.load_tensor(npu.weight)
        ctx.configure(npu)

        w0 = ctx.write_register(npu.ctrl, {"start": 1}, dep=[l1, l2])

        push1 = ctx.push_tensor(npu.ifm, dep=w0)
        push2 = ctx.push_tensor(npu.weight, dep=w0)
        pull1 = ctx.pull_tensor(npu.ofm, dep=[push1, push2])

        p1 = ctx.poll_register(npu.ctrl, "done", dep=w0)
        pull1.add_commit_dependency(p1)

        s1 = ctx.store_tensor(npu.ofm, dep=pull1)
        ctx.verify(s1)  # golden 자동 계산 (forward(**inputs) chain)
```

MACKernel 코드는 **수정 없이** NPUTopKernel 내에서 재사용된다.

**참고:** 이 예시는 서플리먼트 §18.3의 errata 해결 결과를 반영한다. Memory-mapped 인터페이스에서도 `push_tensor`/`pull_tensor`를 명시적으로 사용해야 한다 (Interpretation A 채택).

### 4.8 Connection 검증

빌드 시점에 `ConnectionValidator`가 수행하는 검사:

- **프로토콜 호환성**: 연결된 두 텐서의 protocol이 호환되어야 함 (예: AXI4-Stream ↔ AXI4-Stream)
- **형상 호환성**: 연결된 텐서의 원소 수가 일치하거나 호환 가능해야 함
- **dtype 호환성**: source dtype과 dest dtype이 일치하거나 암묵적 변환 가능해야 함
- **메모리 영역 용량**: auto expose된 텐서가 메모리 영역 한계를 초과하지 않아야 함
- **레지스터 뱅크 겹침**: 동일 AXI-Lite 인터페이스를 공유하는 서브커널의 뱅크가 겹치지 않아야 함
- **댕글링 참조**: connections에서 참조하는 텐서가 실제로 서브커널에 존재해야 함

### 4.9 Cross-Kernel Parameter Sharing

CompositeKernel에서 상위 파라미터가 서브커널로 자동 전파된다. 서브커널의 `default_params`보다 상위 CompositeKernel의 resolved params가 우선한다:

```python
class NPUTopKernel(CompositeKernel):
    dma = DMAKernel()
    mac = MACKernel()
    # 둘 다 project/runtime 스코프에서 C, D, H, W를 상속

    default_params = {
        "TRANSFER_SIZE": "${N}*${C}*${D}*${H}*${W}",
    }
```

파라미터 해결 우선순위:

```
runtime (test config)  >  Kernel.default_params  >  vten.toml [parameters]
```

---

## IR Generation Flow (Full Example)

사용자 코드:

```python
ctx.load_tensor(kernel.ifm)
ctx.load_tensor(kernel.weight)
ctx.configure(kernel)
ctx.write_register(kernel.ctrl, {"start": 1})
ctx.push_tensor(kernel.ifm)
ctx.push_tensor(kernel.weight)
ctx.pull_tensor(kernel.ofm)
ctx.poll_register(kernel.ctrl, "done")
ctx.store_tensor(kernel.ofm)
```

Runtime이 생성하는 IR:

```
# 주소 할당: ifm @ 0x0, weight @ 0x20000, ofm @ 0x40000

LOAD(buf=0, size=131072)                          # ifm → SHM buf 0
LOAD(buf=1, size=65536)                            # weight → SHM buf 1

# configure (auto_bind)
WRITE_REG(offset=0x10, value=0x00000000)           # ifm_base_lo
WRITE_REG(offset=0x14, value=0x00000000)           # ifm_base_hi
WRITE_REG(offset=0x18, value=0x00020000)           # weight_base_lo
WRITE_REG(offset=0x1C, value=0x00000000)           # weight_base_hi
WRITE_REG(offset=0x20, value=0x00040000)           # ofm_base_lo
WRITE_REG(offset=0x24, value=0x00000000)           # ofm_base_hi
WRITE_REG(offset=0x28, value=512)                  # transfer_size
WRITE_REG(offset=0x30, value=64)                   # input_ch = C

WRITE_REG(offset=0x00, value=0x1)                  # start.go = 1

PUSH(iface=0, buf=0, proto=AXI4, addr=0x0, size=131072, role=SLAVE)
PUSH(iface=0, buf=1, proto=AXI4, addr=0x20000, size=65536, role=SLAVE)

PULL(iface=0, buf=2, proto=AXI4, addr=0x40000, size=131072, role=SLAVE)

POLL_REG(offset=0x04, mask=0x1, expected=0x1)

STORE(buf=2, size=131072)
```

BFM 주소 맵:

```
BFM "data_port" Address Map:
  0x0000_0000 ~ 0x0001_FFFF  →  SHM buffer 0 (IFM, accel이 읽기)
  0x0002_0000 ~ 0x0002_7FFF  →  SHM buffer 1 (Weight, accel이 읽기)
  0x0004_0000 ~ 0x0005_FFFF  →  SHM buffer 2 (OFM, accel이 쓰기)

DUT AXI Read Request: araddr=0x0000_1000, arlen=15
  → 조회: buffer 0, offset 0x1000
  → SHM buffer 0에서 16 비트 서빙

DUT AXI Write Request: awaddr=0x0004_0000, awlen=15
  → 조회: buffer 2, offset 0x0
  → SHM buffer 2에 wdata 저장
```
