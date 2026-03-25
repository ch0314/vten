# vTen Code Generation & CLI Workflow

**Version 0.5.0 — March 2026**
**참조: `00_data_models.md` (BFMConfig, Command), `04_backend_xsim.md` (Generated Testbench)**
**Status: Phase 4 구현 전 완성 필요**

---

## Table of Contents

1. [Jinja2 Template Architecture](#1-jinja2-template-architecture)
2. [Template Context Schema](#2-template-context-schema)
3. [Code Generator (sv_generator.py)](#3-code-generator)
4. [CLI Commands](#4-cli-commands)
5. [Error Propagation Path](#5-error-propagation-path)
6. [vten.toml Reference](#6-vtentoml-reference)
7. [Multi-Kernel Project Structure](#7-multi-kernel-project-structure)
8. [Staged Build Pipeline](#8-staged-build-pipeline)

---

## 1. Jinja2 Template Architecture

### 1.1 Template File List

```
templates/
├── tb_top.sv.j2                # DUT 인스턴스화, 클럭/리셋, BFM 연결
├── bfm_instantiation.sv.j2     # BFMConfig 기반 BFM 인스턴스 생성 (include)
├── wire_declarations.sv.j2     # DUT-BFM 와이어 선언 (include)
├── project_setup.tcl.j2        # Vivado 프로젝트 생성 (Stage 1)
└── resolve_order.tcl           # 컴파일 순서 해석 (Stage 4)
```

> **Note (v0.5.0)**: `build_xsim.tcl.j2`, `run_xsim.tcl.j2`, `Makefile.j2`는 §8.9에 따라 삭제.
> xvlog/xelab/xsim 호출은 `vten/cli/build.py`에서 직접 subprocess로 실행.

### 1.2 생성 흐름

```
KernelSpec + BFMConfig[]
       │
       ▼
  sv_generator.py
       │
       ├── tb/generated/tb_top.sv
       ├── tb/generated/vten_types.svh
       └── Makefile
```

고정 모듈(변경 없이 사용)은 `vten_sv/`에서 복사:
- `vten_shm_controller.sv`
- `vten_command_scheduler.sv`
- `vten_bfm_axi4s.sv`
- `vten_bfm_axi4.sv`
- `vten_bfm_axilite.sv`
- `vten_bfm_cmd_if.sv`
- `vten_shm_bridge.c` / `.h`

---

## 2. Template Context Schema

### 2.1 tb_top.sv.j2 Context

```python
@dataclass
class TestbenchContext:
    # Project
    project_name: str
    top_module: str                    # DUT 모듈 이름

    # Session
    session_id: str                    # SHM 이름에 사용

    # DUT
    dut_ports: list[DUTPort]           # RTL 포트 목록
    clock_name: str = "clk"
    reset_name: str = "rst_n"
    reset_active_low: bool = True

    # BFMs
    bfms: list[BFMInstance]

    # Simulation
    clock_period_ns: float = 10.0
    timeout_cycles: int = 1000000

@dataclass
class DUTPort:
    name: str
    direction: str       # "input" | "output" | "inout"
    width: int
    connected_to: str    # BFM 와이어 이름 또는 상수

@dataclass
class BFMInstance:
    name: str                    # 인스턴스 이름 (예: "bfm_data_port")
    module_name: str             # SV 모듈 (예: "vten_bfm_axi4")
    protocol: str                # "axi4_stream" | "axi4" | "axi4_lite"
    data_width: int
    role: str                    # "master" | "slave"
    rtl_port_prefix: str         # DUT 포트 접두사 (예: "m_axi_data")
    parameters: dict             # SV 파라미터 오버라이드
    interface_id: int            # Scheduler BFM 인덱스
```

### 2.2 build_xsim.tcl.j2 Context

```python
@dataclass
class BuildContext:
    vivado_path: str
    rtl_sources: list[str]        # 글로브 확장된 RTL 파일 경로
    include_dirs: list[str]
    generated_sv: list[str]       # 생성된 SV 파일
    vten_sv_dir: str              # 고정 SV 라이브러리 경로
    dpi_c_source: str             # vten_shm_bridge.c 경로
    compile_options: list[str]
    timescale: str = "1ns/1ps"
    top_module: str = "tb_top"
```

---

## 3. Code Generator

### 3.1 sv_generator.py

```python
class SVGenerator:
    def __init__(self, kernel_spec: KernelSpec,
                 bfm_configs: list[BFMConfig],
                 project_config: dict):
        self.spec = kernel_spec
        self.bfm_configs = bfm_configs
        self.config = project_config

    def generate(self, output_dir: str):
        ctx = self._build_context()
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates/'))

        # Testbench
        tb = env.get_template('tb_top.sv.j2')
        write(output_dir / 'tb_top.sv', tb.render(ctx.tb))

    def _build_context(self):
        bfm_instances = []
        for i, bfm_cfg in enumerate(self.bfm_configs):
            iface_spec = self.spec.get_interface(bfm_cfg.interface_name)
            bfm_instances.append(BFMInstance(
                name=f"bfm_{bfm_cfg.interface_name}",
                module_name=self._module_for_protocol(bfm_cfg.protocol),
                protocol=bfm_cfg.protocol.value,
                data_width=bfm_cfg.data_width,
                role=bfm_cfg.role,
                rtl_port_prefix=iface_spec.rtl_port,
                parameters={"DATA_W": bfm_cfg.data_width},
                interface_id=i,
            ))
        # ... DUT port matching, wire generation
```

### 3.2 RTL Port Matching

BFM의 `rtl_port_prefix`를 사용하여 DUT 포트와 BFM 와이어를 자동 매칭:

```
rtl_port: "m_axi_data" → DUT ports: m_axi_data_araddr, m_axi_data_arvalid, ...
                        → BFM wires: bfm_data_port_araddr, bfm_data_port_arvalid, ...
```

프로토콜별 표준 AXI 신호 목록을 기반으로 매칭. 매칭 실패 시 경고.

### 3.3 BFM Index Mapping Generation (v0.4.2)

Scheduler의 `iface_to_bfm[]` 룩업 테이블을 codegen이 생성한다.
`interface_id`(IR lowering에서 할당)를 BFM 인스턴스 인덱스로 변환.

```python
def _generate_bfm_index_mapping(self) -> dict:
    """interface_id → BFM 인덱스 매핑 생성.
    
    BFMConfig 리스트 순서가 BFM 인덱스.
    interface_names() 순서가 interface_id.
    
    Returns: {interface_id: bfm_index} — tb_top.sv.j2에 주입
    """
    # interface_name → interface_id
    iface_id_map = {name: idx
                    for idx, name in enumerate(self.spec.interface_names())}

    # interface_name → bfm_index
    bfm_idx_map = {bfm.interface_name: idx
                   for idx, bfm in enumerate(self.bfm_configs)}

    # interface_id → bfm_index
    iface_to_bfm = {}
    for iface_name, iface_id in iface_id_map.items():
        if iface_name in bfm_idx_map:
            iface_to_bfm[iface_id] = bfm_idx_map[iface_name]
        # Internal 인터페이스는 매핑 없음 (Scheduler가 -1 기본값 사용)

    return iface_to_bfm
```

### 3.4 Scheduler 파라미터 자동 계산 (v0.5)

SVGenerator는 BFMConfig 배열과 Command 수에서 Scheduler 파라미터를 계산하여
`tb_top.sv`의 `vten_command_scheduler` 인스턴스에 전달한다.
기본값(MAX_BFMS=8, MAX_IFACES=16, MAX_CMDS=256)은 하한이며, 대규모 설계에서는 자동 상향된다.

```python
def _compute_scheduler_params(self, num_commands: int) -> dict:
    """BFMConfig[]와 Command 수에서 Scheduler 파라미터 계산.

    vten.toml [backend.scheduler] 오버라이드 적용.
    """
    max_bfms = max(8, len(self.bfm_configs))
    max_ifaces = max(16, max(
        (c.interface_id for c in self.bfm_configs), default=0) + 1)
    max_cmds = max(256, num_commands)

    # vten.toml override (자동 계산보다 큰 경우에만)
    sched_cfg = self.config.get("backend", {}).get("scheduler", {})
    max_bfms = max(max_bfms, sched_cfg.get("max_bfms", 0))
    max_ifaces = max(max_ifaces, sched_cfg.get("max_ifaces", 0))
    max_cmds = max(max_cmds, sched_cfg.get("max_cmds", 0))

    return {
        "max_bfms": max_bfms,
        "max_ifaces": max_ifaces,
        "max_cmds": max_cmds,
    }
```

**Jinja2 템플릿 주입 (tb_top.sv.j2):**

```systemverilog
// Codegen 생성: Scheduler 파라미터 + interface_id → BFM 인덱스 매핑
vten_command_scheduler #(
    .MAX_CMDS({{ max_cmds }}),
    .MAX_BFMS({{ max_bfms }}),
    .MAX_IFACES({{ max_ifaces }})
) scheduler ( ... );

initial begin
    // 기본값: -1 (BFM 없음)
    for (int i = 0; i < {{ max_ifaces }}; i++)
        scheduler.iface_to_bfm[i] = -1;
    // 매핑 설정
    {% for iface_id, bfm_idx in iface_to_bfm.items() %}
    scheduler.iface_to_bfm[{{ iface_id }}] = {{ bfm_idx }};
    {% endfor %}
end
```

---

## 4. CLI Commands

### 4.1 vten init

```bash
$ vten init my_npu
$ vten init my_npu --kernel conv3d    # 커널 디렉토리 추가 생성
```

**단계:**
1. 프로젝트 디렉토리 구조 생성 (첫 실행)
2. `vten.toml` 스켈레톤 생성
3. `--kernel` 옵션 시 커널 디렉토리 + 스켈레톤 파일 생성

**출력 (프로젝트 초기화):**
```
my_npu/
├── vten.toml
├── rtl/                           # 공유 RTL 소스
│   └── (사용자가 RTL 파일 배치)
├── ip/                            # Vivado IP 정의 (.xci 파일)
│   └── (사용자가 IP 정의 배치)
├── kernels/                       # 커널별 디렉토리
│   └── (vten init --kernel로 생성)
├── build/                         # 프로젝트 레벨 빌드 출력
│   ├── vivado_proj/               # Stage 1: Vivado 프로젝트 (.xpr)
│   ├── lib/                       # Stage 2: DPI-C 공유 라이브러리
│   └── .cache.json                # 빌드 캐시
└── results/                       # 테스트 결과
```

**출력 (커널 추가: `vten init --kernel conv3d`):**
```
my_npu/kernels/conv3d/
├── kernel_spec.yaml               # 인터페이스 사양
├── conv3d_kernel.py               # Kernel 클래스
├── tests/
│   └── test_conv3d.py             # TestScenario
└── build/                         # Stage 4+: 커널별 빌드 출력
    ├── generated/                 # tb_top.sv 등
    ├── xsim.dir/                  # xelab 출력
    └── shm/                       # SHM 이미지
```

### 4.2 vten spec --detect

```bash
$ vten spec --detect rtl/conv3d_top.sv
```

**단계:**
1. RTL 파일 파싱 (포트 이름/방향/폭 추출)
2. 프로토콜 추론 (이름 패턴 기반)
3. `specs/<module_name>.yaml` 스켈레톤 생성 (TODO 마커 포함)

**입력:** RTL 파일 경로
**출력:** `specs/<module_name>.yaml`

상세는 `03_kernel_spec_schema.md` §15 참조.

### 4.3 vten build

```bash
$ vten build                              # 전체 빌드 (모든 스테이지, 모든 커널)
$ vten build --kernel conv3d              # 특정 커널만 빌드
$ vten build --stage project_setup         # 특정 스테이지만 실행
$ vten build --upto codegen               # 지정 스테이지까지 실행
$ vten build --force                      # 캐시 무시, 전체 재빌드
$ vten build --skip-compile               # codegen만 (xvlog/xelab 생략)
$ vten build --backend xsim              # 백엔드 선택 (기본: xsim)
$ vten build --config C=32,D=4            # 파라미터 오버라이드
```

**스테이지 파이프라인 (§8 참조):**

| Stage | 이름 | 스코프 | 설명 |
|-------|------|--------|------|
| 1 | `project_setup` | 프로젝트 | Vivado 프로젝트 생성 (RTL + vten_sv + IP 등록, IP generate) |
| 2 | `dpi_c` | 프로젝트 | DPI-C 공유 라이브러리 컴파일 (gcc) |
| 3 | `codegen` | 커널별 | Jinja2 → tb_top.sv 생성 |
| 4 | `compile_order` | 커널별 | Vivado `get_compile_order` → .prj 파일 생성 |
| 5 | `compile` | 커널별 | xvlog + xelab → xsim.dir 생성 |

Stage 1-2는 프로젝트 레벨(한번만 실행), Stage 3-5는 커널 단위로 반복.

**입력:** `vten.toml` + `kernels/<name>/kernel_spec.yaml` + `kernels/<name>/*.py`

**출력 — 프로젝트 레벨:**
```
build/
├── vivado_proj/                   # Stage 1: Vivado 프로젝트
│   ├── vten_sim.xpr              # 프로젝트 파일 (RTL + IP + vten_sv 등록)
│   └── vten_sim.ip_user_files/   # IP 생성 결과 (시뮬레이션 소스 포함)
├── lib/                           # Stage 2: DPI-C
│   └── libvten_shm.so
└── .cache.json                    # 빌드 캐시 (해시, 타임스탬프)
```

**출력 — 커널별 (`kernels/<name>/build/`):**
```
kernels/conv3d/build/
├── generated/                     # Stage 3: codegen 출력
│   └── tb_top.sv
├── compile.prj                    # Stage 4: Vivado가 정렬한 컴파일 순서
├── xsim.dir/                      # Stage 5: xvlog + xelab 출력
│   └── tb_top/
└── shm/                           # vten run 시 생성
    └── kernel_task.bin
```

### 4.4 vten run

```bash
$ vten run --kernel conv3d --test test_conv3d [--backend xsim] [--waveform] [--waveform-on-fail]
$ vten run --kernel conv3d --test test_conv3d --gui   # xsim GUI 모드
```

**`--kernel`은 필수.** 커널별 빌드 산출물과 테스트 디렉토리를 특정한다.

**단계:**
1. `kernels/<kernel>/build/` 에서 빌드 산출물 확인 (없으면 에러)
2. `kernels/<kernel>/tests/` 에서 TestScenario 디스커버리
3. TestScenario 실행 → ExecutionContext 기록 → RuntimeEngine.compile()
4. SHM 이미지를 `kernels/<kernel>/build/shm/kernel_task.bin`에 기록
5. SHM 생성 (`shm_open`) 및 이미지 로드
6. 세마포어 생성 (`sem_open`)
7. 시뮬레이터 프로세스 기동 (`xsim`, 커널별 xsim.dir 사용)
8. Backend ready 대기 (`sem_wait(b2h)`)
9. Batch submit (`host_status = CMD_READY`, `sem_post(h2b)`)
10. 완료 대기 (`sem_wait(b2h)`)
11. 결과 읽기 (Data Region + Stats Region)
12. 검증 실행 (`_run_verification`)
13. SHM/세마포어 정리 (`shm_unlink`, `sem_unlink`)

**입력:** `kernels/<kernel>/build/` + `kernels/<kernel>/tests/`
**출력:** `results/<kernel>/` (pass/fail, stats, 선택적 파형)

```
results/
├── conv3d/
│   └── test_conv3d/
│       ├── summary.json          # pass/fail, 타이밍
│       ├── stats.json            # 커맨드별 통계
│       ├── mismatches.json       # probe 불일치 (있으면)
│       └── waveform.wdb          # 파형 (요청 시)
```

### 4.5 vten report

```bash
$ vten report [--format terminal|html|json]
```

**단계:**
1. `results/` 디렉토리 스캔
2. Stats Region 파싱 → `CommandMetrics` 계산
3. 보고서 렌더링

**출력:** 터미널 보고서, HTML 보고서, 또는 JSON (§5.2 of `05_bfm_library.md` 참조)

---

## 5. Error Propagation Path

### 5.1 Backend → Host 에러 전파

```
BFM 에러 발생 (예: 주소 미매칭 DECERR)
    │
    ▼
Scheduler: report_error(cmd_id, error_code)
    │
    ▼
SHM Controller → S_ERROR 상태
    │
    ▼
DPI-C: vten_signal_error(code, msg)
  - Control Region에 기록:
    - error_code (4B, offset 0x40)
    - error_cmd_id (4B, offset 0x44)
    - error_message (64B, offset 0x48)
  - backend_status = ERROR (3)
  - sem_post(b2h)
    │
    ▼
Python Host: sem_wait(b2h) 반환
  - backend_status == ERROR 확인
  - BackendError 예외 발생:
    - code, cmd_id, message 포함
```

### 5.2 Error Code Values

> 정식 정의: `00_data_models.md` §10.13 `BackendErrorCode`. 여기서는 참조용으로 재기술.

| 코드 | 이름 | 설명 |
|------|------|------|
| 0 | OK | 정상 |
| 1 | ADDR_UNMATCH | AXI4 BFM 주소 매칭 실패 (DECERR) |
| 2 | POLL_TIMEOUT | POLL_REG 타임아웃 |
| 3 | BFM_QUEUE_ERROR | BFM 내부 큐 에러 |
| 4 | SCHEDULER_ERROR | 스케줄러 내부 에러 (의존성 데드락 등) |
| 5 | SHM_ACCESS_ERROR | DPI-C SHM 읽기/쓰기 실패 |
| 6 | UNKNOWN_OPCODE | 알 수 없는 opcode |
| 7 | BFM_MAP_ERROR | interface_id → BFM 매핑 실패 |
| 8 | PROBE_MISMATCH | Probe golden 불일치 (경고, 실행 계속) |
| 9 | TIMEOUT | 전역 시뮬레이션 타임아웃 |

### 5.3 error_message 포맷

```
"[BFM:data_port] DECERR at addr=0x00100000, no matching PUSH entry (cmd_id=5)"
"[BFM:ctrl] POLL_REG timeout after 100000 cycles (cmd_id=8, offset=0x04)"
```

형식: `[<source>] <description> (<context>)`

---

## 6. vten.toml Reference

```toml
[project]
name = "my_npu"
version = "0.1.0"

[parameters]                     # 프로젝트 전역 파라미터 (커널별 오버라이드 가능)
C = 64
D = 32
H = 32
W = 32

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"    # FPGA part (Vivado 프로젝트 생성에 필요)
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0                    # 0 = batch mode (10s default)
submit_timeout_s = 300

[backend.scheduler]              # OPTIONAL. 자동 계산 값보다 큰 경우에만 적용
# max_bfms = 48                  # default: max(8, BFM 수)
# max_ifaces = 48                # default: max(16, interface_id 최대값+1)
# max_cmds = 512                 # default: max(256, command 수)

[backend.verilator]
verilator_path = "/usr/bin/verilator"
threads = 4

[rtl]
sources = ["rtl/**/*.sv", "rtl/**/*.v"]   # 공유 RTL (프로젝트 루트 기준)
top_module = "NPU_3D_top"                  # DUT 탑 모듈 (커널별 오버라이드 가능)
include_dirs = ["rtl/include"]

[ip]                             # OPTIONAL. Vivado IP 정의
sources = ["ip/**/*.xci"]        # .xci 파일 경로 (글로브 지원, PROJECT_ROOT 기준)

[test]
default_seed = 42
waveform = false
waveform_on_fail = true

[report]
format = "terminal"              # terminal | html | json
```

### 6.1 커널별 설정

각 커널의 `kernel_spec.yaml`에서 프로젝트 전역 설정을 오버라이드할 수 있다:

```yaml
# kernels/conv3d/kernel_spec.yaml
kernel: conv3d
rtl_top: rtl/conv3d_wrapper.sv     # PROJECT_ROOT 기준 상대 경로

parameters:
  K: 128                            # 커널 고유 파라미터
  STRIDE: 1

interfaces: { ... }
```

커널별 `rtl_top`은 DUT 모듈을 지정하며, `vten.toml`의 `[rtl].top_module`을 오버라이드한다.

---

## 7. Multi-Kernel Project Structure

### 7.1 디렉토리 레이아웃

```
my_npu/                            # PROJECT_ROOT
├── vten.toml                      # 프로젝트 전역 설정
├── rtl/                           # 공유 RTL 소스 (대용량, 이동 불가)
│   ├── include/
│   ├── NPU_3D_top.sv
│   └── *.sv, *.v
├── ip/                            # Vivado IP 정의
│   ├── block_ram_32k.xci
│   └── fifo_sync.xci
├── build/                         # 프로젝트 레벨 빌드 출력
│   ├── vivado_proj/               # Stage 1 출력: Vivado 프로젝트
│   │   ├── vten_sim.xpr
│   │   └── vten_sim.ip_user_files/
│   ├── lib/                       # Stage 2 출력
│   │   └── libvten_shm.so
│   └── .cache.json                # 빌드 캐시
├── kernels/                       # 커널별 디렉토리
│   ├── conv3d/
│   │   ├── kernel_spec.yaml       # 인터페이스 사양
│   │   ├── conv3d_kernel.py       # Kernel 클래스
│   │   ├── tests/                 # 커널별 TestScenario
│   │   │   └── test_conv3d.py
│   │   └── build/                 # Stage 3-5 출력
│   │       ├── generated/         # tb_top.sv (codegen)
│   │       ├── compile.prj        # Vivado 정렬 컴파일 순서
│   │       ├── xsim.dir/          # xvlog + xelab 출력
│   │       └── shm/               # vten run 시 SHM 이미지
│   ├── dma_ifm/                   # 단위 커널
│   │   ├── kernel_spec.yaml
│   │   ├── dma_ifm_kernel.py
│   │   ├── tests/
│   │   └── build/
│   └── npu_top/                   # CompositeKernel (여러 커널 합성)
│       ├── kernel_spec.yaml
│       ├── npu_top_kernel.py
│       ├── tests/
│       └── build/
└── results/                       # 테스트 결과
    ├── conv3d/
    │   └── test_conv3d/
    └── npu_top/
        └── test_full_trace/
```

### 7.2 설계 원칙

1. **Vivado 프로젝트는 한 번만 생성.** Stage 1에서 RTL + vten_sv + IP를 모두 등록한 Vivado 프로젝트를 `build/vivado_proj/`에 생성. IP generate도 이 단계에서 수행. 각 커널은 이 프로젝트를 열어서 `get_compile_order`로 컴파일 순서를 해결.

2. **Compile order는 Vivado에 위임.** 사용자 RTL의 패키지/모듈 의존성 정렬을 직접 하지 않는다. 커널별 tb_top.sv를 프로젝트에 추가한 후 `get_compile_order`를 호출하여 Vivado가 정렬한 .prj 파일을 생성.

3. **커널별 독립 빌드.** 각 커널의 `build/` 디렉토리는 독립적. 한 커널의 빌드가 다른 커널에 영향 없음.

4. **CompositeKernel은 별도 커널.** `npu_top/`처럼 합성 커널도 커널 디렉토리로 존재하며, 자체 `kernel_spec.yaml`과 테스트를 가짐.

5. **경로 해결.** `kernel_spec.yaml`의 `rtl_top`과 모든 파일 경로는 `PROJECT_ROOT` 기준 상대 경로. `VTEN_ROOT`의 `vten_sv/`, `templates/`는 `import vten`으로 해결.

### 7.3 커널 디스커버리

`vten build`와 `vten run`은 `kernels/` 디렉토리를 스캔하여 커널을 자동 발견:

```python
def discover_kernels(project_dir: Path) -> list[str]:
    """kernels/ 디렉토리에서 kernel_spec.yaml이 있는 커널 발견."""
    kernels_dir = project_dir / "kernels"
    if not kernels_dir.exists():
        return []
    return sorted(
        d.name for d in kernels_dir.iterdir()
        if d.is_dir() and (d / "kernel_spec.yaml").exists()
    )
```

`--kernel` 옵션 없이 `vten build`를 실행하면 모든 발견된 커널을 빌드한다.

---

## 8. Staged Build Pipeline

> **Note (v0.5.0):** 빌드 파이프라인은 `BuildPipeline` ABC로 추상화되었다
> (`08_backend_abstraction.md` §8). 아래는 xsim 백엔드의 5-stage 파이프라인이며,
> `--backend` 플래그로 다른 백엔드의 파이프라인을 선택할 수 있다:
>
> | 백엔드 | 클래스 | 스테이지 |
> |--------|--------|---------|
> | xsim | `XsimBuildPipeline` | project_setup → dpi_c → codegen → compile_order → compile |
> | verilator | `VerilatorBuildPipeline` | dpi_c → codegen → verilate → make |
> | xrt | `XrtBuildPipeline` | gen_packaging_tcl → gen_xo_tcl → gen_link_cfg → validate |

### 8.1 Overview (xsim)

```
vten build [--backend xsim]
    │
    ├── Stage 1: Project Setup     [프로젝트, 캐시됨]
    │   rtl/** + vten_sv/* + ip/*.xci → vivado create_project + generate_target
    │   → build/vivado_proj/vten_sim.xpr
    │
    ├── Stage 2: DPI-C Compile     [프로젝트, 캐시됨]
    │   vten_sv/vten_shm_bridge.c → gcc → build/lib/libvten_shm.so
    │
    │  ┌─── 커널별 반복 ─────────────────────────────────────────┐
    │  │                                                             │
    ├──┤ Stage 3: Codegen          [커널별]                          │
    │  │   kernel_spec.yaml → SVGenerator                            │
    │  │   → kernels/<name>/build/generated/tb_top.sv                │
    │  │     (+ axilite_ctrl.sv, wrapper.sv if generate_controller)  │
    │  │                                                             │
    ├──┤ Stage 4: Compile Order    [커널별]                          │
    │  │   tb_top.sv → Vivado get_compile_order                      │
    │  │   → kernels/<name>/build/compile.prj                        │
    │  │                                                             │
    └──┤ Stage 5: Compile          [커널별]                          │
       │   compile.prj → xvlog --prj + xelab                        │
       │   → kernels/<name>/build/xsim.dir/                          │
       └─────────────────────────────────────────────────────────────┘
```

### 8.2 Stage 1: Project Setup

Vivado 프로젝트를 생성하고, 모든 소스(RTL + vten_sv + IP)를 등록한다. IP generate도 이 단계에서 수행.

**호출:**
```bash
vivado -mode batch -source ${VTEN_ROOT}/templates/project_setup.tcl \
    -tclargs <project_dir> <part> <rtl_sources...> <ip_sources...>
```

**TCL 스크립트 (`templates/project_setup.tcl`):**
```tcl
set proj_dir  [lindex $argv 0]
set part      [lindex $argv 1]
# 나머지 argv: RTL 소스, IP 소스 경로 리스트

# 1) 프로젝트 생성
create_project vten_sim $proj_dir -part $part -force
set_property target_simulator XSim [current_project]

# 2) vten_sv 라이브러리 등록
add_files -fileset sim_1 [glob ${VTEN_ROOT}/vten_sv/*.svh]
add_files -fileset sim_1 [glob ${VTEN_ROOT}/vten_sv/*.sv]

# 3) 사용자 RTL 등록 (순서 무관 — Vivado가 정렬)
foreach src $rtl_sources {
    add_files -fileset sim_1 [glob $src]
}

# 4) include 디렉토리 설정
set_property include_dirs $include_dirs [get_filesets sim_1]

# 5) IP 등록 + 시뮬레이션 소스 생성
foreach xci $ip_sources {
    add_files $xci
}
if {[llength [get_ips]] > 0} {
    generate_target simulation [get_ips *]
}

# 6) 프로젝트 저장
save_project
close_project
```

**출력:** `build/vivado_proj/vten_sim.xpr`

**캐시 조건:** RTL 파일 목록 + IP 파일 목록 + vten_sv 해시의 통합 해시. 변경 없으면 스킵.

**`build/.cache.json` 구조:**
```json
{
  "project_setup": {
    "rtl_hash": "sha256:...",
    "vten_sv_hash": "sha256:...",
    "ip_hash": "sha256:...",
    "timestamp": "2026-03-22T10:30:00"
  },
  "dpi_c": {
    "source_hash": "sha256:...",
    "timestamp": "2026-03-22T10:30:01"
  }
}
```

`--force` 옵션은 캐시를 무시하고 모든 스테이지를 재실행한다.

### 8.3 Stage 2: DPI-C Compile

```bash
gcc -shared -fPIC -o build/lib/libvten_shm.so \
    ${VTEN_ROOT}/vten_sv/vten_shm_bridge.c -lrt -lpthread
```

**캐시 조건:** `vten_shm_bridge.c`의 SHA256 해시.

### 8.4 Stage 3: Codegen (커널별)

각 커널의 `kernel_spec.yaml`을 파싱하여 testbench를 생성:

```python
for kernel_name in target_kernels:
    kernel_dir = project / "kernels" / kernel_name
    spec = parse_kernel_spec(kernel_dir / "kernel_spec.yaml")
    bfm_configs = derive_bfm_configs(spec)

    gen = SVGenerator(kernel_spec=spec, bfm_configs=bfm_configs,
                      project_config=config)
    gen.generate(str(kernel_dir / "build" / "generated"),
                 num_commands=num_commands)
```

**출력:**
- `kernels/<name>/build/generated/tb_top.sv` (항상)
- `kernels/<name>/build/generated/<kernel>_axilite_ctrl.sv` (generate_controller: true 시)
- `kernels/<name>/build/generated/<kernel>_wrapper.sv` (generate_controller: true 시)

> 상세: `10_sv_convenience.md` §5-6

### 8.5 Stage 4: Compile Order Resolution (커널별)

Vivado `get_compile_order`를 사용하여 커널별 컴파일 순서를 해결한다.

**핵심:** 각 커널의 tb_top.sv를 top으로 설정하면 Vivado가 의존성 트리를 분석하여 올바른 컴파일 순서(package → interface → leaf module → top)를 반환한다.

**호출:**
```bash
vivado -mode batch -source ${VTEN_ROOT}/templates/resolve_order.tcl \
    -tclargs <xpr_path> <tb_top_sv_path> <output_prj_path>
```

**TCL 스크립트 (`templates/resolve_order.tcl`):**
```tcl
set xpr_path    [lindex $argv 0]
set tb_top_path [lindex $argv 1]
set output_prj  [lindex $argv 2]

# 1) 프로젝트 열기 (Stage 1에서 생성된 프로젝트)
open_project $xpr_path

# 2) 커널별 tb_top.sv 추가 + top 설정
add_files -fileset sim_1 $tb_top_path
set_property top tb_top [get_filesets sim_1]
update_compile_order -fileset sim_1

# 3) Vivado가 정렬한 컴파일 순서 추출
set ordered [get_compile_order -fileset sim_1 -used_in simulation]

# 4) .prj 파일 생성
set fp [open $output_prj w]
foreach f $ordered {
    set ftype [get_property FILE_TYPE $f]
    set lib   [get_property LIBRARY $f]
    set path  [get_property NAME $f]
    if {$ftype eq "SystemVerilog"} {
        puts $fp "sv $lib $path"
    } elseif {$ftype eq "Verilog"} {
        puts $fp "verilog $lib $path"
    } elseif {$ftype eq "Verilog Header"} {
        puts $fp "sv $lib $path"
    }
}
close $fp

# 5) tb_top.sv 제거 (다음 커널을 위해 프로젝트 원복)
remove_files -fileset sim_1 $tb_top_path
save_project
close_project
```

**Python 호출:**
```python
def resolve_compile_order(vivado_path: str, xpr_path: Path,
                          tb_top_sv: Path, output_prj: Path):
    tcl_script = VTEN_ROOT / "templates" / "resolve_order.tcl"
    subprocess.run([
        f"{vivado_path}/bin/vivado",
        "-mode", "batch",
        "-source", str(tcl_script),
        "-tclargs", str(xpr_path), str(tb_top_sv), str(output_prj),
    ], check=True)
```

**생성되는 .prj 예시 (`kernels/conv3d/build/compile.prj`):**
```
sv work /home/user/vten/vten_sv/vten_types.svh
sv work /home/user/vten/vten_sv/vten_dpi_imports.svh
sv work /home/user/vten/vten_sv/vten_bfm_cmd_if.sv
sv work /home/user/my_npu/rtl/include/common_pkg.sv
sv work /home/user/my_npu/rtl/include/npu_types.sv
sv work /home/user/my_npu/build/vivado_proj/vten_sim.ip_user_files/sim/block_ram_32k.sv
sv work /home/user/my_npu/rtl/primitives/fifo_wrapper.sv
sv work /home/user/my_npu/rtl/conv3d_core.sv
sv work /home/user/my_npu/rtl/NPU_3D_top.sv
sv work /home/user/vten/vten_sv/vten_shm_controller.sv
sv work /home/user/vten/vten_sv/vten_command_scheduler.sv
sv work /home/user/vten/vten_sv/vten_bfm_axi4s.sv
sv work /home/user/vten/vten_sv/vten_bfm_axi4.sv
sv work /home/user/vten/vten_sv/vten_bfm_axilite.sv
sv work /home/user/vten/vten_sv/vten_bfm_probe.sv
sv work /home/user/my_npu/kernels/conv3d/build/generated/tb_top.sv
```

### 8.6 Stage 5: Compile (커널별)

Stage 4에서 생성된 .prj 파일로 xvlog 컴파일 후 xelab elaboration:

```bash
# xvlog: .prj 파일 기반 컴파일 (순서 보장됨)
xvlog --sv --prj kernels/conv3d/build/compile.prj \
    --work work=kernels/conv3d/build/xsim.dir/work

# xelab: DPI-C 링크 + elaboration
xelab tb_top \
    --sv_lib build/lib/libvten_shm \
    --timescale 1ns/1ps \
    --debug typical \
    --snapshot kernels/conv3d/build/xsim.dir/tb_top
```

### 8.7 CLI 옵션 상세

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--kernel <name>` | 특정 커널만 빌드 (Stage 3-5) | `--kernel conv3d` |
| `--stage <name>` | 특정 스테이지만 실행 | `--stage project_setup` |
| `--upto <name>` | 지정 스테이지까지 실행 | `--upto compile_order` |
| `--force` | 캐시 무시, 전체 재빌드 | `--force` |
| `--skip-compile` | codegen만 실행 (Stage 4-5 생략) | `--skip-compile` |
| `--backend <name>` | 백엔드 선택 | `--backend xsim` |
| `--config K=V,...` | 파라미터 오버라이드 | `--config C=32,D=4` |

**스테이지 이름:** `project_setup`, `dpi_c`, `codegen`, `compile_order`, `compile`

**의존성:** `--stage compile`은 이전 스테이지(1-4)가 완료된 상태를 전제. 미완료 시 자동으로 선행 스테이지 실행.

### 8.8 SHM 이미지 생성 시점

SHM 이미지(`kernel_task.bin`)는 `vten build` 시점이 아니라 **`vten run` 시점**에 생성된다. 이유:

1. SHM 이미지 내용은 테스트 시나리오(DSL ops)에 따라 달라진다
2. `RuntimeEngine.compile()`은 테스트 실행 직전에 호출된다
3. 파라미터 오버라이드(`--config`)가 SHM 내용에 영향을 준다

### 8.9 템플릿 파일 목록 (v0.5.0)

```
templates/
├── tb_top.sv.j2                # testbench 생성 (Stage 3)
├── bfm_instantiation.sv.j2     # BFM 인스턴스 생성 (include)
├── wire_declarations.sv.j2     # DUT-BFM 와이어 선언 (include)
├── project_setup.tcl           # Vivado 프로젝트 생성 (Stage 1)
└── resolve_order.tcl           # compile order 추출 (Stage 4)
```

기존 `build_xsim.tcl.j2`, `run_xsim.tcl.j2`, `Makefile.j2`는 삭제.
xvlog/xelab/xsim 호출은 Python (`vten/cli/build.py`, `vten/cli/run.py`)에서 직접 subprocess로 실행.
