# vTen Code Generation & CLI Workflow

**Version 0.4.2 — March 2026**
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

---

## 1. Jinja2 Template Architecture

### 1.1 Template File List

```
templates/
├── tb_top.sv.j2                # DUT 인스턴스화, 클럭/리셋, BFM 연결
├── bfm_instantiation.sv.j2     # BFMConfig 기반 BFM 인스턴스 생성 (include)
├── wire_declarations.sv.j2     # DUT-BFM 와이어 선언 (include)
├── build_xsim.tcl.j2           # xvlog/xelab 빌드 스크립트
├── run_xsim.tcl.j2             # xsim 실행 스크립트
└── Makefile.j2                 # 통합 Makefile
```

### 1.2 생성 흐름

```
KernelSpec + BFMConfig[]
       │
       ▼
  sv_generator.py
       │
       ├── tb/generated/tb_top.sv
       ├── tb/generated/vten_types.svh
       ├── scripts/build.tcl
       ├── scripts/run.tcl
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

        # Build script
        build = env.get_template('build_xsim.tcl.j2')
        write(output_dir / 'build.tcl', build.render(ctx.build))

        # Run script
        run = env.get_template('run_xsim.tcl.j2')
        write(output_dir / 'run.tcl', run.render(ctx.run))

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
```

**단계:**
1. 프로젝트 디렉토리 구조 생성
2. `vten.toml` 스켈레톤 생성
3. 예제 커널/테스트 파일 생성

**출력:**
```
my_npu/
├── vten.toml
├── rtl/
│   └── (사용자가 RTL 파일 배치)
├── specs/
│   └── (kernel_spec.yaml 파일)
├── kernels/
│   └── example_kernel.py
├── tests/
│   └── test_example.py
└── build/
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
$ vten build [--backend xsim] [--config C=32,D=4]
```

**단계:**
1. `vten.toml` 파싱
2. `kernel_spec.yaml` 파싱 및 검증
3. Kernel 클래스 로드 (Python import)
4. `RuntimeEngine.compile()` 실행 → `CompiledResult`
5. `SVGenerator.generate()` → SV/TCL 파일 생성
6. DPI-C 공유 라이브러리 컴파일: `gcc -shared -fPIC -o libvten_shm.so vten_shm_bridge.c -lrt -lpthread`
7. 시뮬레이터 컴파일: `xvlog`, `xelab` 호출

**입력:** `vten.toml` + `specs/*.yaml` + `kernels/*.py`
**출력:** `build/` 디렉토리 (컴파일된 시뮬레이터 + SHM 이미지)

```
build/
├── generated/
│   ├── tb_top.sv
│   └── vten_types.svh
├── lib/
│   └── libvten_shm.so
├── scripts/
│   ├── build.tcl
│   └── run.tcl
├── shm/
│   └── kernel_task.bin       # SHM 이미지 (pre-built)
└── xsim.dir/
    └── (xelab 출력)
```

### 4.4 vten run

```bash
$ vten run --test test_conv3d [--backend xsim] [--waveform] [--waveform-on-fail]
```

**단계:**
1. SHM 생성 (`shm_open`) 및 이미지 로드
2. 세마포어 생성 (`sem_open`)
3. 시뮬레이터 프로세스 기동 (`xsim`)
4. Backend ready 대기 (`sem_wait(b2h)`)
5. Batch submit (`host_status = CMD_READY`, `sem_post(h2b)`)
6. 완료 대기 (`sem_wait(b2h)`)
7. 결과 읽기 (Data Region + Stats Region)
8. 검증 실행 (`_run_verification`)
9. SHM/세마포어 정리 (`shm_unlink`, `sem_unlink`)

**입력:** `build/` + 테스트 시나리오
**출력:** `results/` (pass/fail, stats, 선택적 파형)

```
results/
├── test_conv3d/
│   ├── summary.json          # pass/fail, 타이밍
│   ├── stats.json            # 커맨드별 통계
│   ├── mismatches.json       # probe 불일치 (있으면)
│   └── waveform.wdb          # 파형 (요청 시)
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

[parameters]
C = 64
D = 32
H = 32
W = 32

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2024.1"
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
sources = ["rtl/**/*.sv", "rtl/**/*.v"]
top_module = "npu_top"
include_dirs = ["rtl/include"]

[test]
default_seed = 42
waveform = false
waveform_on_fail = true

[report]
format = "terminal"              # terminal | html | json
```
