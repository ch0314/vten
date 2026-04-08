# vTen E2E Examples & Implementation Roadmap

**Version 0.5.0 — March 2026**
**소스: 서플리먼트 §22, 메인 스펙 §14**

---

## Table of Contents

1. [Passthrough E2E (Minimal Pipeline)](#1-passthrough-e2e)
2. [Conv3D Unit Test](#2-conv3d-unit-test)
3. [NPU Top Composed Full Trace](#3-npu-top-composed-full-trace)
4. [Implementation Phases](#4-implementation-phases)
5. [Open Decisions](#5-open-decisions)

---

## 1. Passthrough E2E (Minimal Pipeline)

최소 파이프라인으로 전체 흐름(DSL → IR → SHM → BFM → DUT → BFM → SHM → verification) 검증.

### 1.1 DUT

AXI4-Stream passthrough — 입력 데이터를 그대로 출력.

```systemverilog
module passthrough #(parameter DATA_W = 256)(
    input  logic clk, input logic rst_n,
    // AXI4-Stream Slave (input) — rtl_port prefix: s_axis
    input  logic [DATA_W-1:0] s_axis_tdata,
    input  logic              s_axis_tvalid,
    output logic              s_axis_tready,
    input  logic              s_axis_tlast,
    // AXI4-Stream Master (output) — rtl_port prefix: m_axis
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

> **포트 이름 규칙:** `rtl_port` 접두사(`s_axis`, `m_axis`)에 AXI4-Stream 표준 접미사
> (`_tdata`, `_tvalid`, `_tready`, `_tlast`)를 붙인다. Codegen의 RTL Port Matching
> (06_codegen_and_cli.md §3.2)이 이 규칙에 따라 DUT 포트와 BFM 와이어를 자동 연결한다.

### 1.2 디렉토리 구조

```
my_project/
├── vten.toml
├── rtl/
│   └── passthrough.sv
└── kernels/
    └── passthrough/
        ├── kernel_spec.yaml
        ├── passthrough_kernel.py
        ├── tests/
        │   └── test_passthrough.py
        └── build/                   # vten build 후 생성
            ├── generated/tb_top.sv
            ├── xsim.dir/
            └── shm/
```

### 1.3 kernel_spec.yaml

```yaml
# kernels/passthrough/kernel_spec.yaml
kernel: passthrough
rtl_top: rtl/passthrough.sv          # PROJECT_ROOT 기준

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

### 1.4 Kernel

```python
# kernels/passthrough/passthrough_kernel.py
class PassthroughKernel(Kernel):
    spec = "kernels/passthrough/kernel_spec.yaml"

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream"
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="output_stream"
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None: rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self):
        return self.data_in.data.clone()
```

### 1.5 Test

```python
# kernels/passthrough/tests/test_passthrough.py
class TestPassthrough(TestScenario):
    kernel = "passthrough"

    def run(self, ctx, cfg):
        k = ctx.instantiate("passthrough", N=1024, **cfg)
        k.generate_inputs(seed=42)

        push1 = ctx.push_tensor(k.data_in)
        pull1 = ctx.pull_tensor(k.data_out, dep=push1)
        ctx.run(verify=True)  # golden 자동 계산 (forward() 호출)
```

### 1.6 Build & Run

```bash
$ cd my_project
$ vten build --kernel passthrough          # Stage 1-5
$ vten run --kernel passthrough --test test_passthrough
```

### 1.7 Expected IR

```
cmd 0: PUSH(iface=0, buf=0, proto=AXI4S, size=1024, role=MASTER)  dep=[]
cmd 1: PULL(iface=1, buf=1, proto=AXI4S, size=1024, role=SLAVE)   dep=[0]
```

### 1.8 Expected SHM

```
Control:     256 B
Commands:    2 × 64B = 128 B
Stats:       2 × 32B = 64 B
Buf Descs:   2 × 24B = 48 B
Data buf 0:  1,024 B (data_in, populated)
Data buf 1:  1,024 B (data_out, empty)
───────────────────────────────────────
Total:       ~1.6 KB
```

---

## 2. Conv3D Unit Test

Memory-mapped 인터페이스 전체 흐름 검증.

### 2.1 Workflow

```
1. configure(kernel)       → N × WRITE_REG cmds (auto_bind)
2. push_tensor(ifm)        → LOAD + PUSH cmds (SHM buf 0 populated, BFM slave, accel reads)
3. push_tensor(weight)     → LOAD + PUSH cmds (SHM buf 1 populated)
4. write_register(start=1) → WRITE_REG cmd
5. pull_tensor(ofm)        → PULL + STORE cmds (BFM slave, accel writes)
6. poll_register(done)     → POLL_REG cmd
7. run(verify=True)        → golden 자동 검증
```

### 2.2 Key Verification Points

- push_tensor 내부의 LOAD 사전 committed → write_register가 dep 충족으로 즉시 dispatch
- configure() → auto_bind 레지스터에 올바른 주소 기록
- PUSH BFM이 DUT read 요청에 올바른 데이터 서빙
- PULL BFM이 DUT write 데이터를 올바르게 SHM에 기록
- POLL_REG가 done 비트 감지 후 PULL의 commit dependency 해제
- STORE 후 역직렬화된 output == golden (run(verify=True)로 자동 검증)

---

## 3. NPU Top Composed Full Trace

서플리먼트 §22의 전체 트레이스를 재배치.

### 3.1 User Code

```python
def run(self, ctx, cfg):
    npu = ctx.instantiate("npu_top", C=64, D=32, H=32, W=32)
    npu.generate_inputs(seed=42)

    ctx.configure(npu)                       # op 0
    push1 = ctx.push_tensor(npu.ifm)            # op 1 → LOAD + PUSH
    push2 = ctx.push_tensor(npu.weight)          # op 2 → LOAD + PUSH
    w0 = ctx.write_register(npu.ctrl,
             {"start": 1}, dep=[push1, push2])   # op 3
    pull1 = ctx.pull_tensor(npu.ofm, dep=w0)     # op 4 → PULL + STORE
    p1 = ctx.poll_register(npu.ctrl,
             "done", dep=w0)                     # op 5
    pull1.add_commit_dependency(p1)
    ctx.run(verify=True)  # golden 자동 계산 (forward() chain)
```

### 3.2 Stage 0 — Flatten

```
Sub-kernels:     dma_ifm, dma_weight, mac, dma_ofm
External ifaces: ddr_port (AXI4), ctrl (AXI4-Lite)
Internal ifaces: mac.axis_ifm (probe), mac.axis_weight, mac.axis_ofm (probe)
Exposed tensors:
  ifm    → dma_ifm.src    → ddr_port  HOST_TO_DEV
  weight → dma_weight.src → ddr_port  HOST_TO_DEV
  ofm    → dma_ofm.dst    → ddr_port  DEV_TO_HOST
Bank offsets:
  dma_ifm=0x000, dma_weight=0x100, mac=0x200, dma_ofm=0x300, global=0x400
```

### 3.3 Stage 1 — Parameters

```
Top resolver: C=64, D=32, H=32, W=32, K=64
  → dma_ifm.SIZE    = 64×32×32×32 = 2,097,152
  → dma_weight.SIZE = 64×64×3×3×3 = 110,592
  → dma_ofm.SIZE    = 64×32×32×32 = 2,097,152
```

### 3.4 Stage 2 — Shapes

```
ifm:    (1, 64, 32, 32, 32) → 2,097,152 elements
weight: (64, 64, 3, 3, 3)   →   110,592 elements
ofm:    (1, 64, 32, 32, 32) → 2,097,152 elements
Connection checks: all pass ✓
```

### 3.5 Stage 3 — Serialization

```
Packing: 8-bit elements, 32/beat, 256-bit bus (32 bytes/beat)
ifm:    65,536 beats × 32B = 2,097,152 bytes (serialized)
weight:  3,456 beats × 32B =   110,592 bytes (serialized)
ofm:    65,536 beats × 32B = 2,097,152 bytes (size only, no data)
```

### 3.6 Stage 4 — Addresses

```
Region: ddr (base=0x0, size=4GB, align=4096)
ifm    → 0x0000_0000  (2,097,152 bytes)
weight → 0x0020_0000  (110,592 bytes, aligned to 4K)
ofm    → 0x0022_0000  (2,097,152 bytes, aligned to 4K)
```

### 3.7 Stage 5 — auto_bind

```
dma_ifm bank (0x000):     0x010: ifm_base_lo=0x00000000
                          0x014: ifm_base_hi=0x00000000
                          0x028: transfer_size=65,536
dma_weight bank (0x100):  0x110: wgt_base_lo=0x00200000
                          0x114: wgt_base_hi=0x00000000
                          0x128: transfer_size=3,456
mac bank (0x200):         0x230: input_ch=64
dma_ofm bank (0x300):     0x310: ofm_base_lo=0x00220000
                          0x314: ofm_base_hi=0x00000000
                          0x328: transfer_size=65,536
```

### 3.8 Stage 6 — IR Commands

```
cmd  0: LOAD     buf=0  size=2,097,152        dep=[]
cmd  1: LOAD     buf=1  size=110,592          dep=[]
cmd  2: WRITE_REG off=0x010 val=0x00000000    dep=[]       # configure
cmd  3: WRITE_REG off=0x014 val=0x00000000    dep=[]
cmd  4: WRITE_REG off=0x028 val=0x00010000    dep=[]
cmd  5: WRITE_REG off=0x110 val=0x00200000    dep=[]
cmd  6: WRITE_REG off=0x114 val=0x00000000    dep=[]
cmd  7: WRITE_REG off=0x128 val=0x00000D80    dep=[]
cmd  8: WRITE_REG off=0x230 val=0x00000040    dep=[]
cmd  9: WRITE_REG off=0x310 val=0x00220000    dep=[]
cmd 10: WRITE_REG off=0x314 val=0x00000000    dep=[]
cmd 11: WRITE_REG off=0x328 val=0x00010000    dep=[]
cmd 12: WRITE_REG off=0x400 val=0x00000001    dep=[0,1]    # start
cmd 13: PUSH     iface=0 buf=0 addr=0x0      dep=[12]     # push ifm
cmd 14: PUSH     iface=0 buf=1 addr=0x200000 dep=[12]     # push weight
cmd 15: PULL     iface=0 buf=2 addr=0x220000 dep=[13,14]  # pull ofm
                                              commit_dep=[16]
cmd 16: POLL_REG off=0x404 mask=1 exp=1       dep=[12]     # poll done
cmd 17: STORE    buf=2                        dep=[15]     # store ofm
```

### 3.9 Stage 6b — BFM Configs

```
BFM "ddr_port" (AXI4, slave, 256-bit):
  [0x0000_0000, 2,097,152B, buf=0]  ← ifm
  [0x0020_0000,   110,592B, buf=1]  ← weight
  [0x0022_0000, 2,097,152B, buf=2]  ← ofm

BFM "ctrl" (AXI4-Lite, master, 32-bit):
  (register-only, no address ranges)
```

### 3.10 Stage 7 — SHM Layout

```
Region           Offset        Size
─────────────────────────────────────────
Control          0             256 B
Commands         256           18 × 64B = 1,152 B
Stats            1,408         18 × 32B = 576 B
Buffer Descs     1,984         3 × 24B = 72 B
Data buf 0       2,112         2,097,152 B  (ifm, populated)
Data buf 1       2,099,264     110,592 B    (weight, populated)
Data buf 2       2,209,920     2,097,152 B  (ofm, empty)
─────────────────────────────────────────
Total            ~4.1 MB
```

---

## 4. Implementation Phases

### 4.1 Phase Breakdown

| Phase | Focus | Deliverables | 참조 스펙 |
|-------|-------|-------------|----------|
| 0 | 스펙 재구성 | 이 문서들 | 계획서 |
| 1 | Python Core | kernel/, dsl/, spec/ 패키지 | 00 + 01 + 03 |
| 2 | Runtime | runtime/ 패키지 | 00 + 02 |
| 3 | SV + C Backend | vten_sv/ 디렉토리 | 00 + 04 + 05 |
| 4 | Integration | codegen/, cli/, templates/ | 00 + 06 |
| 5 | Validation | examples/, tests/ | 00 + 07 |

### 4.2 Phase별 완료 기준

**Phase 1:** Kernel 선언 → Tensor 생성 → generate_inputs() → forward() 동작. CompositeKernel interface_map 검증 통과. kernel_spec.yaml 파싱.

**Phase 2:** Unit/Composite Kernel 모두에 대해 DSL → IR → SHM 이미지 생성 동작. §3의 Full Trace 기대값과 일치.

**Phase 3:** C 라이브러리 컴파일 성공. SV 모듈 xvlog 구문 검사 통과. 개별 BFM 테스트벤치 동작.

**Phase 4:** Staged build pipeline 동작. `vten build --kernel passthrough` → 5-stage 빌드 (project_setup → dpi_c → codegen → compile_order → compile) 성공. `vten run --kernel passthrough --test test_passthrough` → xsim 기동 → SHM 핸드셰이크 → 결과 수집 파이프라인 동작. Passthrough E2E xsim 실행 PASS.

**Phase 5:** Conv3D golden match. NPU Top composed multi-kernel test pass with probe. Multi-kernel 프로젝트 전체 빌드 및 실행.

### 4.3 Minimal End-to-End Target

Phase 4 마일스톤: 가장 단순한 파이프라인으로 전체 흐름을 검증.

1. **DUT**: AXI4-Stream passthrough (data in → same data out)
2. **Kernel**: 단일 push_tensor + pull_tensor. output == input 검증.
3. **Backend**: xsim with DPI-C SHM bridge.
4. **Serialization**: 8비트 원소 패킹, 256비트 버스.
5. **Build**: `vten build --kernel passthrough` (5-stage 빌드 파이프라인)
6. **Run**: `vten run --kernel passthrough --test test_passthrough`

이것으로 전체 파이프라인(DSL → IR → SHM → BFM → DUT → BFM → SHM → verification)을 검증한 후 복잡도를 추가한다.

### 4.4 Multi-Kernel E2E Target

Phase 5 마일스톤: 다중 커널 프로젝트에서 공유 빌드 + 커널별 테스트.

1. **프로젝트**: `my_npu/` 아래 `conv3d`, `dma_ifm`, `npu_top` 커널 3개
2. **공유 빌드**: Vivado project setup + DPI-C (한 번만)
3. **커널별 빌드**: 각 커널 codegen → compile_order → compile (독립적)
4. **CompositeKernel**: `npu_top`이 `conv3d` + `dma_ifm` 합성
5. **전체 빌드**: `vten build` (모든 커널 빌드, Vivado 프로젝트 캐시 활용)

---

## 5. Open Decisions

### Resolved (v0.3-v0.4)

- ~~SHM 동기화 메커니즘~~: Semaphore 기반
- ~~SHM 바이너리 프로토콜~~: 64B 커맨드, 32B 통계, 24B 버퍼 디스크립터
- ~~Command Scheduler 의존성 해결~~: Committed bitmap
- ~~BFM 커맨드 큐잉~~: SV dynamic queue
- ~~AXI4 BFM 동시 커맨드~~: Active Table with address matching
- ~~BARRIER 구현~~: Preprocessed barrier fence lookup
- ~~Sync 모드 인코딩~~: flags[0] SYNC 비트

### Resolved (v0.4.2)

- ~~CONFIGURE OpCode SHM 출현 여부~~: 삭제됨. Runtime이 WRITE_REG로 확장. OpCode re-encoding (BARRIER=8). COMPARE도 삭제됨.
- ~~Backend error code 값 테이블~~: `BackendErrorCode` 클래스 정의 (00_data_models §10.13). 코드 0~9.
- ~~cmd_bfm_map 생성~~: Codegen이 `iface_to_bfm[]` 룩업 테이블 생성 (06_codegen_and_cli §3.3).
- ~~Controller↔Scheduler 인터페이스~~: `feed_valid`/`feed_ready`/`feed_done` handshake 통일. S_DISPATCH → S_LOAD_BATCH + S_FEED 분리.
- ~~Scheduler dependency 배열~~: `bfm_cmd_t`와 분리. `vten_read_command_deps()` DPI-C 함수 추가.
- ~~bus_width ≤ data_width 규칙~~: 03_kernel_spec_schema §9.1에 검증 규칙 추가.

### Remaining Open

- Verilator 지원 시점: 초기 구현 포함 vs 후속 Phase
- Register spec 포맷: 현재 YAML-inline; IP-XACT 또는 Vivado address editor 통합 가능
- Probe mode 성능 영향: 벤치마킹 필요
- Custom packing 복잡도: 혼합 필드 비트 정의의 한계
- Multi-layer sequential execution: 주소 재사용 패턴 및 double-buffering 자동화
