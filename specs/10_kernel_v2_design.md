# Kernel v2 Design — 전체 설계 계획

## 설계 원칙

```
Kernel = logical 세계의 함수
layout/unlayout 메서드 = logical ↔ physical 변환 (커널에 종속)
Framework = 자동 배관 (pack, chain, verify)
run() = 사용자가 작성하는 HW 실행 프로토콜
```

---

## 1. Tensor — logical이 primary

### 1.1 변경 요약

| 항목 | 현재 | v2 |
|------|------|-----|
| `shape` | physical shape | **logical shape** |
| `dtype` | physical dtype | **logical dtype** |
| `data` | 사용자가 직접 설정 | framework 전용 (physical) |
| `logical_data` | optional, 수동 관리 | **primary 사용자 인터페이스** |
| `layout` | 별도 Layout 클래스 인스턴스 | **커널 메서드** `layout_{name}()` / `unlayout_{name}()` |
| `layout.pack()` | 사용자가 수동 호출 | **framework가 커널 메서드 자동 호출** |

### 1.2 Tensor 선언 (Before → After)

```python
# ── BEFORE ──
ifm_mem = Tensor(
    shape=("${ifm_total_bytes}",),        # physical (flat bytes)
    dtype=torch.uint8,                     # physical dtype
    interface="ifm_dma",
    direction=Direction.HOST_TO_DEV,
    layout=DDRIfmLayout(),                 # logical_shape는 Layout 안에
)

# ── AFTER ──
ifm_mem = Tensor(
    shape=("${in_ch}", "${in_depth}", "${in_height}", "${in_width}"),  # logical
    dtype=torch.int32,                                                  # logical
    interface="ifm_dma",
    # layout 없음 — 커널 메서드 layout_ifm_mem() / unlayout_ifm_mem()으로 정의
)
```

### 1.3 Layout은 커널 메서드

Layout 클래스를 별도로 만들지 않는다. **커널 메서드**로 정의:

```python
class FmapIOKernel(Kernel):
    ifm_mem = Tensor(
        shape=("${in_ch}", "${in_depth}", "${in_height}", "${in_width}"),
        dtype=torch.int32,
        interface="ifm_dma",
    )

    # layout_{tensor_name}: logical → physical
    def layout_ifm_mem(self, logical: torch.Tensor) -> torch.Tensor:
        return tiled_pack(logical, self.Ti, self.To)  # 유틸리티 호출

    # unlayout_{tensor_name}: physical → logical
    def unlayout_ifm_mem(self, physical: torch.Tensor) -> torch.Tensor:
        return tiled_unpack(physical, self.Ti, self.To)
```

**규칙:**
- `layout_{tensor_name}(self, logical) → physical` 메서드가 있으면 framework가 Stage 3에서 자동 호출
- `unlayout_{tensor_name}(self, physical) → logical` 메서드가 있으면 auto-verify 시 자동 호출
- 없으면 identity (logical == physical) — passthrough 커널 등
- `self.*`로 모든 param 접근 가능 — 별도 `params: dict` 불필요
- 재사용 가능한 변환 로직은 유틸리티 함수로 분리하고 메서드에서 호출

```python
# 같은 변환을 여러 tensor에 적용
def layout_ifm_mem(self, logical):
    return tiled_pack(logical, self.Ti, self.To)

def layout_wgt_mem(self, logical):
    return weight_tiled_pack(logical, self.Ti, self.To, self.in_depth)  # depth 반복 포함
```

Layout 없는 Tensor: logical == physical (identity). 현재 passthrough 커널 그대로.

### 1.4 Direction 추론

```
layout_{name} 메서드 있으면:
  - layout만 → HOST_TO_DEV (host가 pack해서 device에 보냄)
  - unlayout만 → DEV_TO_HOST (device 출력을 host가 unpack)
  - 양쪽 다 → BIDIRECTIONAL

layout 메서드 없으면:
  - 현재 로직 유지 (protocol/role 기반 추론)
```

Direction을 명시하면 추론보다 우선.

### 1.5 resolved_shape의 의미 변경

현재 `_resolved_shape`은 physical shape. v2에서는 **logical shape**를 resolve.
Physical shape는 Layout.pack()의 출력에서 자동으로 결정.

```python
# Tensor._resolve_shape(resolver)
# shape=("${in_ch}", "${in_depth}", "${in_height}", "${in_width}")
# → _resolved_shape = (32, 4, 4, 4)  ← logical
# → _element_count = 2048             ← logical
```

### 1.6 generate_inputs() 단순화

```python
# ── BEFORE ──
def generate_inputs(self, seed=None):
    ifm_raw = torch.randint(0, 100, (self.in_ch, self.in_depth, ...))
    self.ifm_mem.logical_data = ifm_raw
    self.ifm_mem.data = self.ifm_mem.layout.pack(ifm_raw, {
        "in_ch": self.in_ch, ...   # 수동 params dict 구성
    })
    self._ifm_ddr = ...  # golden용 별도 저장

# ── AFTER ──
def generate_inputs(self, seed=None):
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    self.ifm_mem.logical_data = torch.randint(
        0, 100, self.ifm_mem.resolved_shape,   # logical shape 사용
        dtype=self.ifm_mem.dtype, generator=rng,
    )
    # 끝. framework가 compile 시 자동으로:
    # tensor.data = self.layout_ifm_mem(tensor.logical_data)
```

### 1.7 Framework auto-pack (Stage 3 pre-step)

```python
# engine.py — Stage 3 직전에 삽입
def _auto_pack_tensors(self, kernel, view: FlattenedKernelView):
    """layout_{name} 메서드가 있는 tensor: logical_data → layout → data 자동 설정."""
    for exposed in view.exposed_tensors.values():
        tensor = exposed.origin_tensor
        layout_method = getattr(kernel, f"layout_{tensor.name}", None)
        if layout_method is not None and tensor.logical_data is not None:
            tensor.data = layout_method(tensor.logical_data)
```

---

## 2. forward() 통합

### 2.1 시그니처

```python
class Kernel:
    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """인자명 = input tensor 이름, 반환값 = {output tensor 이름: data}
        HW의 I/O를 그대로 재현하는 behavioral model."""
        raise NotImplementedError
```

### 2.2 규칙

1. **인자 이름** = input tensor의 `name` (Kernel class에서 선언한 attribute 이름)
2. **반환값** = `{output_tensor_name: data}` dict — HW 출력과 동일한 format
3. **HW behavioral model** — forward()는 HW의 동작을 그대로 서술한다.
   입출력 모두 HW I/O와 동일한 physical format을 사용.
   이로써 auto-chain에서 upstream.forward() 출력이 downstream에
   별도 변환 없이 직접 전달된다.
4. **params**는 `self.*`로 접근 (resolved params가 instance attrs로 노출)

### 2.3 Before → After 예시

```python
# ── BEFORE (act_quant) ──
golden_requires = ["psum_flat", "bias_data"]
golden_provides = {"quant_out": "quant_out"}

@classmethod
def golden(cls, psum_flat, bias_data, bias_shift, is_relu, out_ch, ...):
    return {"quant_out": cls.golden_quantize(psum_flat, bias_data, ...)}

def forward(self) -> torch.Tensor:
    return type(self).golden(
        psum_flat=self._psum_data, bias_data=self._bias_data,
        bias_shift=self.bias_shift, ...
    )["quant_out"]

# ── AFTER (act_quant) — HW behavioral model ──
def forward(self, **inputs) -> dict[str, torch.Tensor]:
    """Golden: psum + bias → ReLU → shift → clip."""
    psum_flat = inputs.get("psum_in", self.psum_in.logical_data)
    bias_data = inputs.get("bias_in", self.bias_in.logical_data)
    # ... ReLU → shift → clip ...
    return {"quant_out": (clipped & 0xFF).flatten().to(torch.uint8)}
```

### 2.3.1 HW Behavioral Model 예시

forward()는 HW의 I/O를 그대로 재현한다. 파이프라인에서 upstream의 출력이
downstream의 입력으로 직접 전달되므로, 중간 변환이 불필요하다.

```python
# ── mac_atu: MAC 연산 + serialization ──
class MacAtuKernel(Kernel):
    def _compute_einsum(self, **inputs) -> torch.Tensor:
        """Core MAC: physical streams → einsum tensor."""
        ifm = self._parse_physical_ifmap(inputs["ifmap"])
        wgt = self._parse_physical_weight(inputs["weight"])
        return torch.einsum("gcdhw,gcotuxyz->uxdgohwtyz", ifm, wgt)

    def _serialize_psum(self, mac_result: torch.Tensor) -> torch.Tensor:
        """Einsum tensor → 21-bit packed stream bytes (HW output format)."""
        ...  # PSUM_IN_DW 비트로 pack → (total_beats * ATU_AXIS_OUT, BYTES_PER_BEAT) uint8

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """HW behavioral model: physical IFM/weight → packed psum streams."""
        mac_result = self._compute_einsum(**inputs)
        packed = self._serialize_psum(mac_result)
        return {"partial_sum": packed}  # HW 출력 format 그대로

# ── psum_buffer: deserialization + accumulation ──
class PsumBufferKernel(Kernel):
    def _deserialize_psum(self, packed_bytes: torch.Tensor) -> torch.Tensor:
        """21-bit packed stream bytes → einsum tensor."""
        ...  # (total_beats * ATU_AXIS_OUT, BYTES_PER_BEAT) → einsum

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """HW behavioral model: packed psum streams → accumulated output."""
        mac_result = self._deserialize_psum(inputs["psum_in"])
        # ... accumulation, stride handling ...
        return {"psum_out": result.flatten().to(torch.int32)}
```

auto-chain에서의 데이터 흐름:
```
mac.forward(ifmap=physical, weight=physical)
  → {"partial_sum": packed_bytes}  (21-bit packed stream)
  → connection: mac.partial_sum >> psum.psum_in
  → psum.forward(psum_in=packed_bytes)  (변환 없이 직접 전달)
  → {"psum_out": accumulated_result}
```

### 2.4 삭제되는 것

- `golden()` classmethod
- `golden_requires`, `golden_provides`
- `golden_seeds`, `golden_output`
- `golden_quantize()` 같은 static helper (forward에 inline)
- `_psum_data`, `_ifm_raw` 등 `_*_data` state 패턴
- `forward_ifm_out()`, `forward_coo_out()` 같은 per-tensor forward 메서드

### 2.5 fmapIO — 부분 입력 처리

fmapIO는 IFM read 경로와 OFM write 경로가 동일 kernel에 있어서 cycle 발생:
```
fmapIO → mac → psum → act → fmapIO
```

해법: forward()가 **가용한 입력만으로 가능한 출력을 반환**:

```python
class FmapIOKernel(Kernel):
    def forward(self, ifm_mem=None, ofm_in=None) -> dict[str, torch.Tensor]:
        result = {}
        if ifm_mem is not None:
            # IFM read path: identity passthrough (semantic)
            result["ifm_out"] = ifm_mem
            result["coo_out"] = self._compute_coordinates()
        if ofm_in is not None:
            # OFM write path: identity passthrough (semantic)
            result["ofm_mem"] = ofm_in
        return result
```

Framework는 tensor-level dataflow로 multi-round evaluation:
```
Round 1: ifm_mem ready → fmap.forward(ifm_mem=...) → ifm_out, coo_out ready
         wgt_mem ready → wl.forward(wgt_mem=...) → wgt_out ready
Round 2: mac.forward(ifmap=ifm_out, weight=wgt_out) → partial_sum ready
Round 3: psum.forward(psum_in=partial_sum) → psum_out ready
         bias_mem ready → bias.forward(bias_mem=...) → bias_out ready
Round 4: act.forward(psum_in=psum_out, bias_in=bias_out) → quant_out ready
Round 5: fmap.forward(ofm_in=quant_out) → ofm_mem ready  ← 2nd call
```

### 2.6 Standalone forward 호출

```python
# Framework가 자동으로:
inputs = {}
for tensor in kernel.input_tensors():
    inputs[tensor.name] = tensor.logical_data
golden_outputs = kernel.forward(**inputs)
```

### 2.7 Composite forward 자동 chain

```python
# Framework가 connections graph + forward() 시그니처로 자동 chain
# CompositeKernel.forward() 오버라이드 불필요
# 단, 사용자가 원하면 오버라이드 가능
```

---

## 3. Parameter 통합

### 3.1 Parameter default를 Kernel class에 선언

```python
class ActQuantKernel(Kernel):
    spec = "kernels/act_quant/kernel_spec.yaml"

    default_params = {
        "in_depth": 4, "in_height": 4, "in_width": 4,
        "out_ch": 32, "bias_shift": 8, "is_relu": 1,
        "ifm_stride": 1, "ofm_stride": 1,
    }
```

### 3.2 Resolution 우선순위

```
runtime (test config)  >  Kernel.default_params  >  vten.toml [parameters]
```

### 3.3 kernel_spec.yaml에서 삭제

```yaml
# BEFORE — runtime_params 섹션
runtime_params:
  scale_factor:
    default: 2              # ← Kernel.default_params로 이동
    register: ctrl.scale_factor  # ← 이름 매칭 자동, 불필요

# BEFORE — register defaults
registers:
  - { name: in_depth, width: 8, default: 4 }  # ← default 삭제

# AFTER — 순수 HW 맵만
registers:
  - { name: in_depth, width: 8 }
  - { name: out_ch,   width: 9 }
```

### 3.4 compute_derived_params() — instance method로 변경

조건부 파생 (stride 분기 등)은 선언적으로 표현 불가능. 유지하되 **instance method**로:

```python
# BEFORE (static, dict 접근)
@staticmethod
def compute_derived_params(params: dict) -> dict:
    if params["ifm_stride"] == 2:
        eff_depth = params["in_depth"] // 2
    ...

# AFTER (instance method, self.* 접근)
def compute_derived_params(self) -> dict:
    if self.ifm_stride == 2:
        eff_depth = self.in_depth // 2
    elif self.ofm_stride == 2:
        eff_depth = self.in_depth * 2
    else:
        eff_depth = self.in_depth
    return {"eff_depth": eff_depth, ...}
```

가능한 이유: KernelInstance.initialize() 순서 변경
```
현재:  Resolver → compute_derived(params_dict) → Instance 생성 → attrs 설정
v2:    Resolver → Instance 생성 → attrs 설정 → self.compute_derived() → derived attrs 설정
```

### 3.5 Parameter access 완전 통일

**모든 곳에서 `self.param_name`으로 접근.** 예외 없음:
- `compute_derived_params(self)` — `self.out_ch`
- `generate_inputs(self)` — `self.in_ch`
- `forward(self, ...)` — `self.bias_shift`
- `run(self, ctx)` — `self.kernel_size`

삭제:
- `params: dict` 인자 패턴
- `**kw` (golden 삭제)
- `resolver.namespace` 직접 접근

---

## 4. Register 단순화

### 4.1 kernel_spec.yaml register — 순수 HW 맵

```yaml
interfaces:
  ctrl:
    protocol: axi4_lite
    role: slave
    data_width: 32
    rtl_port: s_axilite_control
    registers:
      # param register: 이름 매칭 → configure()에서 자동 write
      - { name: in_depth,   width: 8 }
      - { name: out_ch,     width: 9 }
      - { name: bias_shift, width: 5 }
      - { name: is_relu,    width: 1 }
      - { name: ifm_stride, width: 2 }
      - { name: ofm_stride, width: 2 }

      # auto_bind: tensor 속성에서 계산
      - { name: ifm_addr_lo, width: 32,
          auto_bind: { tensor: ifm_mem, value: address, bits: "31:0" } }

      # control: pulse → configure() skip, run()에서 수동
      - { name: vsync, width: 1, pulse: true }

      # read-only: write 안 함
      - { name: done, width: 1, access: ro }
```

### 4.2 configure() 통합 규칙

```python
def configure(kernel):
    for register in kernel_spec.registers:
        if register.auto_bind:
            value = compute_auto_bind(register.auto_bind)  # tensor addr/size
            write_register(register, value)
        elif register.pulse or register.access == "ro":
            continue  # skip
        elif register.name in param_namespace:
            value = param_namespace[register.name]
            write_register(register, value)
        # else: skip (no matching param)
```

### 4.3 삭제되는 메커니즘

- `runtime_params` 섹션 및 `RuntimeParamSpec` 클래스
- `register: ctrl.field_name` 매핑 구문
- `role: config` register 속성
- `config_map` (SubKernelBinding)
- `resolve_runtime_param_registers()`, `resolve_config_registers()`,
  `resolve_composite_config_registers()` — 하나의 `resolve_param_registers()`로 통합

### 4.4 RTL register 이름 통일 (Option B)

NPU RTL에서 이름 불일치 수정:
```
conv_ifm_stride → ifm_stride
conv_ofm_stride → ofm_stride
```

이로써 `config_map` 완전 불필요.

---

## 5. Composition 단순화

### 5.1 Sub-kernel 선언 — bind() 없이

```python
# ── BEFORE ──
wl = WeightLoaderKernel.bind(
    interface_map={
        "ctrl": ("wl_ctrl", "weight_loader"),
        "wgt_dma": "wgt_dma",
        "wgt_out": Internal(),
    },
    config_map={"conv_ifm_stride": "ifm_stride"},
)

# ── AFTER ──
wl = WeightLoaderKernel()  # 인스턴스 선언만
```

### 5.2 Connection → 자동 추론

```python
class NpuPipelineKernel(CompositeKernel):
    wl   = WeightLoaderKernel()
    fmap = FmapIOKernel()
    mac  = MacAtuKernel()
    psum = PsumBufferKernel()
    act  = ActQuantKernel()
    bias = BiasLoaderKernel()

    connections = [
        wl.wgt_out     >> mac.weight,
        fmap.ifm_out   >> mac.ifmap,
        fmap.coo_out   >> mac.ifmap_coord,
        mac.partial_sum >> psum.psum_in,
        mac.psum_coord  >> psum.psum_coord_in,
        psum.psum_out  >> act.psum_in,
        bias.bias_out  >> act.bias_in,
        act.quant_out  >> fmap.ofm_in,
    ]
```

### 5.3 자동 추론 규칙

| 현재 수동 | v2 자동 규칙 |
|----------|------------|
| `interface_map={"wgt_out": Internal()}` | connections에 등장하는 tensor의 interface → Internal |
| `interface_map={"wgt_dma": "wgt_dma"}` | connections에 없는 tensor의 interface → auto expose |
| `register("wl_ctrl")` | sub-kernel의 register interface → `{sub_name}_{iface_name}` 자동 prefix |
| `wl.wgt_mem.expose("wgt_dma")` | 자동 expose (이름 = `{sub_name}_{tensor_name}` 또는 원본 유지) |
| `golden_seeds/provides` | forward() 시그니처에서 자동 추론 |
| `config_map` | RTL 통일로 불필요, param 이름 매칭 자동 |

### 5.4 >> 연산자 구현

```python
class TensorRef:
    """Sub-kernel tensor reference (CompositeKernel class body에서 생성)."""
    def __init__(self, sub_kernel_name: str, tensor_name: str, kernel_class: type):
        self.sub_kernel_name = sub_kernel_name
        self.tensor_name = tensor_name
        self.kernel_class = kernel_class

    def __rshift__(self, other: TensorRef) -> Connection:
        return Connection(source=self, dest=other)
```

SubKernelRef (현재 SubKernelBinding 역할)가 `__getattr__`로 TensorRef 반환:

```python
class SubKernelRef:
    """CompositeKernel class body에서 sub-kernel 참조."""
    def __init__(self, kernel_class: type):
        self.kernel_class = kernel_class
        self._attr_name = ""  # __set_name__에서 설정

    def __set_name__(self, owner, name):
        self._attr_name = name

    def __getattr__(self, name) -> TensorRef:
        # kernel_class의 Tensor descriptor인지 확인
        attr = getattr(self.kernel_class, name, None)
        if isinstance(attr, Tensor):
            return TensorRef(self._attr_name, name, self.kernel_class)
        raise AttributeError(...)
```

### 5.5 Exposed tensor 접근

CompositeKernel 인스턴스에서 exposed tensor 접근:
```python
# 자동 expose 규칙:
# connections에 등장하지 않는 tensor → exposed
# 이름: sub_kernel_name을 prefix로 (충돌 시)

# 접근 예:
self.wl.wgt_mem       # sub-kernel의 exposed tensor (ExposedTensor 프록시)
self.fmap.ifm_mem     # sub-kernel의 exposed tensor
self.fmap.ofm_mem     # sub-kernel의 exposed tensor
self.bias.bias_mem    # sub-kernel의 exposed tensor
```

`__init_subclass__`에서 exposed tensor를 자동으로 composite의 attribute로 등록.

### 5.6 Composite generate_inputs()

```python
class NpuPipelineKernel(CompositeKernel):
    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        # exposed tensor에 logical_data만 설정
        self.wl.wgt_mem.logical_data = torch.randint(
            -100, 100, self.wl.wgt_mem.resolved_shape,
            dtype=self.wl.wgt_mem.dtype, generator=rng,
        )
        self.fmap.ifm_mem.logical_data = torch.randint(...)
        self.bias.bias_mem.logical_data = torch.randint(...)
        # framework auto-pack
```

### 5.6.1 Auto-chain generate_inputs (Composite Registry)

Standalone 커널 검증 시 upstream 체인을 자동 실행하여 입력을 생성한다.

**메커니즘:**
1. `__init_subclass__`에서 `_composite_registry`에 sub-kernel → composite 매핑 등록
2. `Kernel.generate_inputs()`가 자체 구현 없으면:
   - `_lookup_composite()`로 registry 조회 (class identity tolerance)
   - 실패 시 `_discover_composite()`로 sibling 커널 디렉토리 자동 스캔
3. `_generate_inputs_for()`에서:
   - `_same_kernel_class()`로 target의 sub-ref name 매칭 (re-import 허용)
   - reverse-BFS로 **upstream 의존성만** 수집 (target 제외 → 사이클 회피)
   - target의 resolved params를 upstream에 전달
   - upstream 순회: `generate_inputs()` → pool에서 connected inputs 설정 → `forward()`
4. target 커널의 connected input 텐서에 `data` + `logical_data` 설정

```python
# source 커널만 generate_inputs 구현
class WeightLoaderKernel(Kernel):
    def generate_inputs(self, seed=None): ...  # ✅ source — 직접 구현
    def forward(self, **inputs): ...

class MacAtuKernel(Kernel):
    # generate_inputs 불필요 — auto-chain이 해결
    def forward(self, **inputs): ...           # ✅ forward만 구현 (HW behavioral model)

class PsumBufferKernel(Kernel):
    def forward(self, **inputs): ...           # ✅ forward만 구현 (HW behavioral model)

class ActQuantKernel(Kernel):
    def forward(self, **inputs): ...           # ✅ forward만 구현

# standalone 검증:
psum = ctx.instantiate(PsumBufferKernel, **params)
psum.generate_inputs(seed=42)
# → _lookup_composite() → NpuPipelineKernel 발견
# → reverse-BFS: upstream = {wl, fmap, mac}
# → upstream-only topo sort: [wl, fmap, mac]
# → wl.generate_inputs() + wl.forward()
# → fmap.generate_inputs() + fmap.forward()
# → mac.forward() → packed psum streams (HW format)
# → psum.psum_in.data = packed psum streams
```

**데이터 흐름:**
- Exposed inputs: `logical_data` → `layout_{name}()` → physical
- Connected inputs: forward() 출력 (HW physical format) → 직접 전달
- target에 `data` + `logical_data` 모두 설정

**Auto-discovery (`_discover_composite`):**
Registry가 비어 있을 때 (CompositeKernel이 아직 import되지 않은 경우),
커널의 소스 파일 위치에서 sibling 디렉토리를 스캔하여 CompositeKernel을
자동 발견한다. conftest.py에서 수동 import할 필요 없음.

**Class identity tolerance:**
같은 .py 파일이 다른 모듈 이름으로 로드될 때 (`_vten_kernel_X` vs `X.X_kernel`),
`_same_kernel_class()`가 클래스 이름 + 소스 파일 경로로 동일성을 판단하고,
`_lookup_composite()`가 fallback 매칭 + 캐싱을 수행한다.

### 5.7 Composite forward() — 자동 chain

CompositeKernel.forward() 기본 구현:
```python
class CompositeKernel(Kernel):
    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Connection graph 기반 자동 forward chain."""
        # 1. exposed input tensor → pool에 넣기
        pool = {}  # (sub_name, tensor_name) → logical data
        for name, tensor in self._exposed_inputs():
            pool[(sub_name, tensor_name)] = inputs[name]

        # 2. Multi-round dataflow evaluation
        for round in range(MAX_ROUNDS):
            progress = False
            for sub_name, sub_kernel in self._sub_kernels():
                # 이 sub_kernel의 input tensor 중 pool에 있는 것들 수집
                available_inputs = self._collect_available_inputs(sub_name, pool)
                if not available_inputs:
                    continue
                # 이미 계산된 output은 skip
                if self._all_outputs_computed(sub_name, pool):
                    continue
                # forward() 호출
                outputs = sub_kernel.forward(**available_inputs)
                for tensor_name, data in outputs.items():
                    pool[(sub_name, tensor_name)] = data
                    # connection을 통해 downstream으로 전파
                    for conn in self._outgoing_connections(sub_name, tensor_name):
                        pool[(conn.dest_sub, conn.dest_tensor)] = data
                progress = True
            if not progress:
                break

        # 3. exposed output tensor 수집
        return {name: pool[(sub, tensor)]
                for name, sub, tensor in self._exposed_outputs()}
```

사용자가 오버라이드 가능하지만, 대부분의 경우 자동 chain으로 충분.

### 5.8 Composite run()

run()은 **항상 사용자 작성**. HW 실행 프로토콜은 자동화 불가:
```python
def run(self, ctx):
    h_cfg = ctx.configure(self)
    ctx.push_tensor(self.wl.wgt_mem, dep=h_cfg)
    ctx.push_tensor(self.fmap.ifm_mem, dep=h_cfg)
    ctx.push_tensor(self.bias.bias_mem, dep=h_cfg)
    h_ofm = ctx.pull_tensor(self.fmap.ofm_mem, dep=h_cfg)
    for ctrl in [self.psum_ctrl, self.act_ctrl, self.bias_ctrl,
                  self.wl_ctrl, self.fmap_ctrl]:
        ctx.write_register(ctrl, {"vsync": 1}, dep=h_cfg)
```

---

## 6. Verification 자동화

### 6.1 ctx.run(verify=True) — auto-golden 검증

```python
# BEFORE (deprecated — ctx.verify() 삭제됨)
golden = self.forward()
ctx.verify(h_ofm, golden)

# AFTER — run(verify=True)로 자동 검증
ctx.run(verify=True)  # framework가 forward()를 호출하여 golden 자동 계산
```

### 6.2 Framework auto-verify 동작 (구현)

`vten/runtime/context.py`의 `_run_auto_verification()`:

```python
def run(self, verify: bool = False):
    # ... compile + execute ...
    if verify:
        self._run_auto_verification(compiled, result)
    # auto-golden은 _compute_auto_golden()으로 각 D2H 텐서에 대해 계산
```

### 6.3 Auto-golden 계산 파이프라인

```
op_handle → tensor_name → _find_kernel_for_tensor()
  → kernel_inst 찾기 (_kernels 순회, tensors() 및 _auto_exposed 매칭)
  → _run_forward(kernel_inst) 호출 (결과는 _golden_cache에 캐싱)
    → CompositeKernel: forward() no args (auto-chain이 layout + 연결 처리)
    → Simple Kernel: H2D 텐서 수집 + layout 적용 + forward(**inputs)
  → fwd_result[tensor_name] 추출
  → 형식 변환 (필요시)
  → chunk 슬라이싱 (필요시)
  → golden 반환
```

### 6.4 형식 변환 규칙

forward() 출력 dtype과 tensor dtype이 다를 때 자동 변환:

| forward() dtype | tensor dtype | 처리 |
|---|---|---|
| 동일 | 동일 | 그대로 비교 |
| uint8 (packed) | int32 (unpacked) | serialize → deserialize 왕복 (packing scheme 적용) |
| 다름 | 다름 | dtype cast |

**mac_atu 케이스**: forward() → packed uint8, tensor dtype = int32
→ auto-golden이 forward() 출력을 `StreamSerializer.serialize()` → `deserialize()`로 왕복
→ 21-bit truncation이 반영된 int32 golden 생성
→ SHM deserialize 결과와 정확히 일치

### 6.5 chunk 지원

`pull_tensor(tensor, chunks=N)`으로 분할된 출력도 auto-golden 지원:
```python
psum_handles = ctx.pull_tensor(self.partial_sum, chunks=self.in_depth)
# run(verify=True)가 각 chunk에 대해 golden 자동 슬라이싱
```

golden 전체를 계산한 뒤, `chunk_index`와 `chunk_total`로 구간 슬라이싱.

### 6.6 golden 캐싱

동일 커널의 여러 출력 텐서에 대해 forward()를 한 번만 호출:
```python
# _golden_cache: dict[int, dict[str, torch.Tensor]]
# key = id(kernel_inst), value = forward() 결과 dict
# run() 완료 후 clear
```

### 6.7 NPU 커널 run() 간소화 결과

```python
# ── weight_loader: 수동 layout + forward → auto-golden ──
# BEFORE:
physical = self.layout_wgt_mem(self.wgt_mem.data)
ctx.verify(h_out, self.forward(wgt_mem=physical)["wgt_out"])
# AFTER:
ctx.run(verify=True)  # auto-golden 자동 계산

# ── mac_atu: 수동 einsum + per-depth 슬라이싱 → auto-golden + chunk ──
# BEFORE (12줄):
mac_result = self._compute_einsum(ifmap=self.ifmap.data, weight=self.weight.data)
for d in range(self.in_depth):
    golden_slices = []
    for u in range(mac_result.shape[0]):
        for x in range(mac_result.shape[1]):
            golden_slices.append(mac_result[u, x, d].flatten())
    golden_d = torch.cat(golden_slices).to(torch.int32)
    ctx.verify(psum_handles[d], golden_d)
# AFTER (1줄):
ctx.run(verify=True)  # chunk별 golden 자동 슬라이싱

# ── fmapIO: 수동 layout + 2개 verify → auto-golden ──
# BEFORE:
physical = self.layout_ifm_mem(self.ifm_mem.data)
golden = self.forward(ifm_mem=physical)
ctx.verify(h_ifm, golden["ifm_out"])
ctx.verify(h_coo, golden["coo_out"])
# AFTER:
ctx.run(verify=True)  # 모든 D2H 텐서 자동 검증

# ── composite (mac_psum, npu_pipeline) ──
# BEFORE:
ctx.verify(h_out, self.forward()["psum_out"])
# AFTER:
ctx.run(verify=True)
```

---

## 7. kernel_spec.yaml v2

### 7.1 최소 스키마

```yaml
kernel: act_quant
rtl_top: act_quant_core

clock: { name: ap_clk }
reset: { name: ap_aresetn, active_low: true }

interfaces:
  ctrl:
    protocol: axi4_lite
    rtl_port: s_axilite_control
    data_width: 32
    registers:
      - { name: in_depth,    width: 8 }
      - { name: in_height,   width: 8 }
      - { name: in_width,    width: 8 }
      - { name: out_ch,      width: 9 }
      - { name: bias_shift,  width: 5 }
      - { name: is_relu,     width: 1 }
      - { name: ifm_stride,  width: 2 }
      - { name: ofm_stride,  width: 2 }
      - { name: vsync,       width: 1, pulse: true }

  psum_in:
    protocol: axi4_stream
    role: slave
    data_width: 64
    packing: { element_width: 32, elements_per_beat: 2 }

  bias_in:
    protocol: axi4_stream
    role: slave
    data_width: 64
    packing: { element_width: 32, elements_per_beat: 2 }

  quant_out:
    protocol: axi4_stream
    role: master
    data_width: 16
    packing: { element_width: 8, elements_per_beat: 2 }
```

### 7.2 삭제된 필드

- `runtime_params` 섹션 전체
- `registers.*.default` (→ Kernel.default_params)
- `registers.*.role` (→ 규칙 기반: auto_bind/pulse/ro/else)
- `registers.*.alias` (→ RTL 통일로 불필요)
- `interfaces.*.tensor` (→ Tensor 선언에서 interface 참조)

---

## 8. 완성된 Kernel v2 예시

### 8.1 ActQuantKernel (leaf kernel)

```python
import torch
from vten import Kernel, Tensor, register

class ActQuantKernel(Kernel):
    spec = "kernels/act_quant/kernel_spec.yaml"
    ctrl = register("ctrl")

    default_params = {
        "in_depth": 4, "in_height": 4, "in_width": 4,
        "out_ch": 32, "bias_shift": 8, "is_relu": 1,
        "ifm_stride": 1, "ofm_stride": 1,
    }

    psum_in = Tensor(
        shape=("${total_psum_elems}",),
        dtype=torch.int32,
        interface="psum_in",
    )
    bias_in = Tensor(
        shape=("${out_ch}",),
        dtype=torch.int32,
        interface="bias_in",
    )
    quant_out = Tensor(
        shape=("${total_psum_elems}",),
        dtype=torch.uint8,
        interface="quant_out",
    )

    def compute_derived_params(self) -> dict:
        och_groups = (self.out_ch + self.To - 1) // self.To
        d, h, w = self.in_depth, self.in_height, self.in_width
        if self.ifm_stride == 2:
            d, h, w = d // 2, h // 2, w // 2
        elif self.ofm_stride == 2:
            d, h, w = d * 2, h * 2, w * 2
        total = d * och_groups * h * w * (self.To // self.OUT_GROUP) * self.OUT_GROUP
        return {"total_psum_elems": total,
                "eff_depth": d, "eff_height": h, "eff_width": w}

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.psum_in.logical_data = torch.randint(
            -50000, 50000, self.psum_in.resolved_shape,
            dtype=torch.int32, generator=rng)
        self.bias_in.logical_data = torch.randint(
            -10000, 10000, self.bias_in.resolved_shape,
            dtype=torch.int32, generator=rng)

    def forward(self, psum_in, bias_in) -> dict[str, torch.Tensor]:
        To = self.To
        och_groups = (self.out_ch + To - 1) // To
        och_pad = (To - self.out_ch % To) % To
        psum = psum_in.to(torch.int64).reshape(
            self.eff_depth, och_groups, self.eff_height * self.eff_width, To)
        bias = bias_in.to(torch.int64)
        if och_pad > 0:
            bias = torch.nn.functional.pad(bias, (0, och_pad))
        biased = psum + bias.reshape(1, och_groups, 1, To)
        if self.is_relu:
            activated = torch.clamp(biased, min=0)
            shifted = activated >> self.bias_shift
            clipped = shifted.clamp(0, 255)
        else:
            shifted = biased >> self.bias_shift
            clipped = shifted.clamp(-128, 127)
        return {"quant_out": (clipped & 0xFF).flatten().to(torch.uint8)}

    def run(self, ctx):
        h_bias = ctx.push_tensor(self.bias_in)
        h_out = ctx.pull_tensor(self.quant_out, dep=h_bias)
        h_cfg = ctx.configure(self, dep=h_bias)
        ctx.write_register(self.ctrl, {"vsync": 1}, dep=h_cfg)
        ctx.push_tensor(self.psum_in, dep=h_cfg)
```

### 8.2 NpuPipelineKernel (composite)

```python
from vten import CompositeKernel, register

class NpuPipelineKernel(CompositeKernel):
    spec = "kernels/npu_pipeline/kernel_spec.yaml"

    default_params = {
        "in_ch": 32, "out_ch": 32,
        "in_depth": 4, "in_height": 4, "in_width": 4,
        "kernel_size": 3, "ifm_stride": 1, "ofm_stride": 1,
        "bias_shift": 8, "is_relu": 1,
    }

    # Sub-kernels
    wl   = WeightLoaderKernel()
    fmap = FmapIOKernel()
    mac  = MacAtuKernel()
    psum = PsumBufferKernel()
    act  = ActQuantKernel()
    bias = BiasLoaderKernel()

    # Internal connections (>> 연산자)
    connections = [
        wl.wgt_out      >> mac.weight,
        fmap.ifm_out    >> mac.ifmap,
        fmap.coo_out    >> mac.ifmap_coord,
        mac.partial_sum >> psum.psum_in,
        mac.psum_coord  >> psum.psum_coord_in,
        psum.psum_out   >> act.psum_in,
        bias.bias_out   >> act.bias_in,
        act.quant_out   >> fmap.ofm_in,
    ]

    # 자동 추론:
    # - Exposed tensors: wl.wgt_mem, fmap.ifm_mem, fmap.ofm_mem, bias.bias_mem
    # - Internal: wgt_out, ifm_out, coo_out, weight, ifmap, ...
    # - Registers: wl_ctrl, fmap_ctrl, mac_ctrl, psum_ctrl, act_ctrl, bias_ctrl
    # - forward(): connection chain 자동 실행

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.wl.wgt_mem.logical_data = torch.randint(
            -100, 100, self.wl.wgt_mem.resolved_shape,
            dtype=self.wl.wgt_mem.dtype, generator=rng)
        self.fmap.ifm_mem.logical_data = torch.randint(
            0, 100, self.fmap.ifm_mem.resolved_shape,
            dtype=self.fmap.ifm_mem.dtype, generator=rng)
        self.bias.bias_mem.logical_data = torch.randint(
            -10000, 10000, self.bias.bias_mem.resolved_shape,
            dtype=self.bias.bias_mem.dtype, generator=rng)

    def run(self, ctx):
        h_cfg = ctx.configure(self)
        ctx.push_tensor(self.wl.wgt_mem, dep=h_cfg)
        ctx.push_tensor(self.fmap.ifm_mem, dep=h_cfg)
        ctx.push_tensor(self.bias.bias_mem, dep=h_cfg)
        h_ofm = ctx.pull_tensor(self.fmap.ofm_mem, dep=h_cfg)
        for ctrl in [self.psum_ctrl, self.act_ctrl, self.bias_ctrl,
                      self.wl_ctrl, self.fmap_ctrl]:
            ctx.write_register(ctrl, {"vsync": 1}, dep=h_cfg)
```

### 8.3 Test (변경 거의 없음)

```python
from vten import TestScenario

class TestForwardK3(TestScenario):
    kernel = "npu_pipeline"
    configs = [
        {"name": "forward_k3", "in_ch": 32, "out_ch": 32,
         "in_width": 4, "in_height": 4, "in_depth": 4,
         "kernel_size": 3, "ifm_stride": 1, "ofm_stride": 1,
         "bias_shift": 8, "is_relu": 1},
    ]
```

---

## 9. 구현 순서 및 현황

### Phase A: Tensor logical-first + Parameter 통일 — ✅ 완료

수정 파일:
- `vten/kernel/tensor.py` — shape=logical, resolved_shape=logical
- `vten/kernel/base.py` — forward(**inputs)→dict, default_params, compute_derived_params(self), layout_*/unlayout_* convention
- `vten/runtime/flattener.py` — KernelInstance.initialize() 순서 변경

구현 노트:
- `logical_data` 별도 필드 대신, `tensor.data`를 primary로 유지 (설계 대비 단순화)
- generate_inputs()에서 `tensor.data = logical_value` 설정 → Stage 3에서 `layout_{name}()` 자동 호출
- KernelInstance.initialize() 순서:
  Resolver → Instance 생성 → base params를 attrs로 설정 → self.compute_derived_params() → derived attrs 설정 → Tensor shape resolve

### Phase B: forward() 통합 — ✅ 완료

수정 파일:
- `vten/kernel/base.py` — forward(**inputs) → dict 시그니처
- `vten/kernel/composite.py` — auto forward chain (multi-round dataflow)
- `vten/runtime/engine.py` — auto-pack Stage 3 pre-step (`_apply_layout`), `_apply_unlayout`
- `vten/runtime/context.py` — run(verify=True) auto-golden 계산

### Phase C: Parameter & Register 통합 — ✅ 완료

수정 파일:
- `vten/spec/models.py` — RegisterSpec 단순화
- `vten/spec/parser.py` — register default 삭제
- `vten/runtime/binder.py` — `resolve_registers()` 통합
- `vten/runtime/resolver.py` — Kernel.default_params 참조 추가

### Phase D: Composition 자동 추론 — ✅ 완료

수정 파일:
- `vten/kernel/composite.py` — `>>` 연산자, TensorRef, auto-expose, auto-prefix, auto-chain forward(), _generate_inputs_for(), _discover_composite()
- `vten/runtime/flattener.py` — auto-expose 기반 flatten 로직, sub-kernel instances

### Phase E: Tests & Examples 마이그레이션 — ✅ 완료

- vten 테스트: 1841/1841 통과
- NPU 3D 커널 9개 마이그레이션 완료 (mac_atu, psum_buffer, weight_loader, bias_loader, fmapIO, act_quant, mac_psum, npu_pipeline + _common utilities)

### Phase F: Spec 문서 업데이트 — 진행 중

---

## 10. 결정된 사항

### 10.1 Composite에서 exposed tensor 접근 → Option C 확정

```python
self.wl.wgt_mem       # sub-kernel 객체를 통해 접근
self.fmap.ifm_mem     # 명시적이고 충돌 없음
```

`__init_subclass__`에서 `_auto_exposed`를 computed하고,
`KernelInstance._resolve_exposed_tensors()`에서 ExposedTensor 프록시를 생성하여
composite 인스턴스의 attribute로 등록.

### 10.2 Weight tiling → layout_wgt_mem()에 통합 확정

```python
class WeightLoaderKernel(Kernel):
    def layout_wgt_mem(self, logical: torch.Tensor) -> torch.Tensor:
        """(out_ch, in_ch, K, K, K) int32 → flat uint8 DDR layout.
        channel padding, per-Ti transpose, kernel_size-dependent packing, depth tiling 포함."""
        # self.in_depth, self.Ti, self.To 등 직접 접근
        ...
```

### 10.3 Composite의 top-level kernel_spec.yaml → optional 확정

NPU composite는 spec 파일 없이 sub-kernel spec 합성으로 동작.
composite spec은 있으면 memory_regions/clock/reset 오버라이드 가능.

### 10.4 logical_data vs data → data를 primary로 유지

설계 시 `logical_data`를 별도 필드로 계획했으나, 구현에서는:
- `tensor.data` = 사용자가 설정하는 logical 데이터 (generate_inputs에서)
- Stage 3에서 `_apply_layout()`이 `layout_{name}()` 호출 → physical bytes 생성
- CompositeKernel.forward() pool seeding에서도 `tensor.data` + `layout_{name}()` 자동 적용

---

## 11. 구현 상세 — 설계 대비 실제 차이

### 11.1 Auto-layout 호출 시점

**설계**: Stage 3 pre-step에서 `tensor.data = layout(tensor.logical_data)` 변환
**구현**: `RuntimeEngine._apply_layout(view, exposed, tensor.data)` — Stage 3 직전에 호출
- `engine.py:1080-1098`: `_apply_layout()` — sub-kernel의 `layout_{name}()` 메서드 호출
- `engine.py:1101-1119`: `_apply_unlayout()` — output tensor의 `unlayout_{name}()` 메서드 호출
- `composite.py:313-330`: CompositeKernel.forward() pool seeding에서도 auto-layout 적용

### 11.2 Auto-golden verification 구현 위치

`vten/runtime/context.py`:
- `_run_auto_verification(compiled, result)` — run(verify=True) 시 호출
- `_compute_auto_golden(tensor_name, compiled)` — 핵심 로직
- `_find_kernel_for_tensor(tensor_name)` — 커널 탐색
- `_run_forward(kernel_inst)` — Composite vs Simple 분기
- `_golden_cache` — `id(kernel_inst)` → forward() 결과 캐시

### 11.3 Auto-chain generate_inputs 구현

`vten/kernel/composite.py`:
- `_generate_inputs_for(target_kernel, seed)` (lines 410-553)
- reverse-BFS로 upstream 의존성만 수집 (target 제외 → 사이클 회피)
- `_same_kernel_class()`: 클래스 이름 + 소스 파일 경로로 동일성 판단
- `_discover_composite()`: sibling 디렉토리 자동 스캔

`vten/kernel/base.py`:
- `generate_inputs()` fallback: `_lookup_composite()` → `_discover_composite()` → `_generate_inputs_for()`

### 11.4 Vectorized behavioral model

NPU 커널의 forward()/serialize/deserialize 성능 최적화:

| 커널 | 메서드 | 최적화 | 성능 |
|------|--------|--------|------|
| mac_atu | `_serialize_psum()` | 36 vectorized numpy ops (21-bit packing) | 84M → 36 ops |
| psum_buffer | `_deserialize_psum()` | vectorized 21-bit unpacking | 3.5M → 36 ops |
| weight_loader | `_ti_parallel_stream_k3()` | broadcast + even/odd split | 80M → broadcast |
| mac_psum | weight packing in `generate_inputs()` | numpy pad + broadcast | 4.2M → broadcast |

핵심 패턴: Python-level 루프를 numpy array 연산으로 대체.
bit packing/unpacking은 K*K=9 반복 × 4 byte 접근 = 36 vectorized ops로 고정.

### 11.5 CompositeKernel forward() multi-round evaluation

`composite.py:287-408`:
1. **Pool seeding**: exposed input tensors → kwargs 또는 instance data, auto-layout 적용
2. **Multi-round loop** (MAX_ROUNDS=20):
   - topo order로 sub-kernel 순회
   - 가용한 입력으로 forward() 호출
   - 출력을 connection을 통해 downstream으로 전파
   - cycle 처리: partial input guard (`if "x" in inputs`)
3. **결과 수집**: exposed output tensors만 반환

fmapIO cycle 처리:
```
Round 1: fmap.forward(ifm_mem=...) → ifm_out, coo_out
Round 2: mac.forward(ifmap=ifm_out, weight=wgt_out) → partial_sum
Round 3: psum.forward(psum_in=partial_sum) → psum_out
Round 4: act.forward(psum_in=psum_out, bias_in=bias_out) → quant_out
Round 5: fmap.forward(ofm_in=quant_out) → ofm_mem  ← 2nd call
```
