# CLAUDE.md — vTen Project Instructions

## Project Overview

vTen은 DSA(Domain-Specific Accelerator) 검증을 위한 텐서 중심 프레임워크이다.
Python 프런트엔드(Kernel/DSL)에서 텐서 단위 검증 시나리오를 정의하고,
8-stage 컴파일 파이프라인으로 Execution IR과 SHM 바이너리 이미지를 생성한 후,
DPI-C 브릿지를 통해 xsim 시뮬레이터에서 BFM이 DUT를 구동하여 검증한다.

## Architecture

```
User Code (Kernel + DSL)
    │
    ▼
ExecutionContext.run()
    │  Record pass: Operation 리스트 생성
    ▼
RuntimeEngine.compile()  ← 8-stage pipeline
    │  Stage 0: Composite Flatten
    │  Stage 1: Parameter Resolution
    │  Stage 2: Shape Validation
    │  Stage 3: Tensor Serialization
    │  Stage 4: Address Allocation
    │  Stage 5: auto_bind Resolution
    │  Stage 6: IR Lowering → Command[]
    │  Stage 7: SHM Packing → bytes
    ▼
Backend.submit(shm_image, bfm_configs)
    │  POSIX SHM + Named Semaphore handshake
    ▼
xsim Process
    │  SHM Controller → Command Scheduler → BFMs → DUT
    ▼
Verification (golden vs DUT output)
```

## Spec Files (Authoritative Source)

**스펙 파일은 최고 권위(authority)이다. 설계 결정을 임의로 바꾸지 말 것. 수정이 필요하다면 꼭 물어볼 것**

| File | Content | When to read |
|------|---------|-------------|
| `specs/00_data_models.md` | 모든 데이터 타입 정의 (Single Source of Truth) | **항상** — 모든 Phase에서 참조 |
| `specs/01_kernel_and_dsl.md` | Kernel 클래스, DSL 연산, CompositeKernel 합성 | Phase 1 |
| `specs/02_runtime_engine.md` | 8-stage 컴파일 파이프라인, ExecutionContext API | Phase 2 |
| `specs/03_kernel_spec_schema.md` | kernel_spec.yaml 완전한 스키마 | Phase 1 (parser) |
| `specs/04_backend_xsim.md` | SHM, DPI-C Bridge, Controller, Scheduler | Phase 3 |
| `specs/05_bfm_library.md` | AXI4-Stream / AXI4 / AXI4-Lite BFM | Phase 3 |
| `specs/06_codegen_and_cli.md` | Jinja2 코드 생성, CLI 워크플로우 | Phase 4 |
| `specs/07_e2e_examples.md` | E2E 예제, 구현 Phase 계획 | Phase 5 |

## Project Structure

```
vten/
├── CLAUDE.md                  ← 이 파일
├── pyproject.toml
├── vten.toml                  # 프로젝트 구성 (E2E 테스트용)
├── specs/                     # 스펙 문서 (읽기 전용 참조)
├── vten/                      # Python 패키지
│   ├── __init__.py
│   ├── kernel/                # Phase 1: Kernel, Tensor, CompositeKernel
│   │   ├── __init__.py
│   │   ├── tensor.py
│   │   ├── base.py
│   │   ├── composite.py
│   │   └── register.py
│   ├── dsl/                   # Phase 1: DSL Operations, Dependency
│   │   ├── __init__.py
│   │   ├── operations.py
│   │   └── dependency.py
│   ├── spec/                  # Phase 1: kernel_spec.yaml parser
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── parser.py
│   ├── runtime/               # Phase 2: 8-stage compile pipeline
│   │   ├── __init__.py
│   │   ├── context.py         # ExecutionContext
│   │   ├── engine.py          # RuntimeEngine (orchestrator)
│   │   ├── flattener.py       # Stage 0
│   │   ├── resolver.py        # Stage 1
│   │   ├── serializer.py      # Stage 3
│   │   ├── address.py         # Stage 4
│   │   ├── binder.py          # Stage 5
│   │   ├── ir.py              # Stage 6
│   │   ├── shm.py             # Stage 7
│   │   └── errors.py          # VTenError hierarchy
│   ├── backend/               # Phase 4: Backend adapters
│   │   ├── __init__.py
│   │   ├── base.py            # Backend ABC
│   │   └── xsim.py            # xsim backend
│   ├── codegen/               # Phase 4: Jinja2 code generation
│   │   ├── __init__.py
│   │   ├── sv_generator.py
│   │   └── script_gen.py
│   └── cli/                   # Phase 4: CLI commands
│       ├── __init__.py
│       ├── main.py
│       ├── init_cmd.py
│       ├── build.py
│       ├── run.py
│       └── report.py
├── vten_sv/                   # Phase 3: SystemVerilog library (fixed)
│   ├── vten_types.svh
│   ├── vten_dpi_imports.svh
│   ├── vten_bfm_cmd_if.sv
│   ├── vten_shm_controller.sv
│   ├── vten_command_scheduler.sv
│   ├── vten_bfm_axi4s.sv
│   ├── vten_bfm_axi4.sv
│   ├── vten_bfm_axilite.sv
│   ├── vten_bfm_probe.sv
│   ├── vten_shm_bridge.c
│   └── vten_shm_bridge.h
├── templates/                 # Phase 4: Jinja2 templates
│   ├── tb_top.sv.j2
│   ├── bfm_instantiation.sv.j2
│   ├── wire_declarations.sv.j2
│   ├── build_xsim.tcl.j2
│   ├── run_xsim.tcl.j2
│   └── Makefile.j2
├── tests/                     # pytest 테스트
│   ├── conftest.py
│   ├── test_tensor.py
│   ├── test_kernel.py
│   ├── test_composite.py
│   ├── test_spec_parser.py
│   ├── test_dsl.py
│   ├── test_runtime_resolver.py
│   ├── test_runtime_serializer.py
│   ├── test_runtime_address.py
│   ├── test_runtime_ir.py
│   ├── test_runtime_shm.py
│   └── test_e2e_passthrough.py
└── examples/                  # Phase 5: E2E examples
    ├── passthrough/
    │   ├── rtl/passthrough.sv
    │   ├── specs/passthrough.yaml
    │   ├── kernels/passthrough_kernel.py
    │   └── tests/test_passthrough.py
    └── conv3d/
        ├── rtl/
        ├── specs/
        ├── kernels/
        └── tests/
```

## Workflow

Implementer / Tester 분리 TDD 워크플로우: [`WORKFLOW.md`](WORKFLOW.md) 참조.

## Implementation Phases

```
Phase 1: Python Core       ← specs: 00 + 01 + 03
   kernel/, dsl/, spec/ 패키지
   완료 기준: Kernel 선언 → Tensor → generate_inputs() → forward() 동작

Phase 2: Runtime Engine     ← specs: 00 + 02
   runtime/ 패키지
   완료 기준: DSL → IR → SHM 이미지 생성, Full Trace 기대값 일치

Phase 3: SV + C Backend     ← specs: 00 + 04 + 05
   vten_sv/ 디렉토리
   완료 기준: gcc 컴파일 성공, xvlog 구문 통과, BFM 단위 테스트

Phase 4: Integration        ← specs: 00 + 06
   codegen/, cli/, backend/ 패키지 + templates/
   완료 기준: vten build → vten run 파이프라인 동작 (Passthrough)

Phase 5: Validation         ← specs: 07
   examples/, 통합 테스트
   완료 기준: Passthrough E2E pass, Conv3D golden match
```

## Implementation Rules

1. **스펙 우선.** 모든 데이터 타입, Enum 값, 상수, 바이너리 오프셋은 `00_data_models.md`를 정확히 따른다.
2. **Phase 순차.** Phase N의 테스트가 통과해야 Phase N+1에 진입한다.
3. **테스트 우선.** 각 모듈의 단위 테스트를 먼저 작성하고, 구현으로 테스트를 통과시킨다.
4. **SHM 바이너리 정확성.** 오프셋, 크기, 엔디안을 스펙과 byte-level로 정확히 맞춘다.
5. **설계 결정 임의 변경 금지.** 스펙에 명시된 결정(예: `__init_subclass__` 방식, `copy.copy` shallow copy, Semaphore 동기화)을 다른 방식으로 바꾸지 않는다.
6. **불명확한 부분은 질문.** 스펙에 없는 구현 세부사항이 필요하면 임의로 결정하지 말고 물어본다.

## Coding Conventions

### Python
- Type hints 사용 (Python 3.10+ union syntax `X | Y`)
- `@dataclass` 적극 활용 (00_data_models.md의 모든 모델)
- 파일명: `snake_case.py`
- 에러: `VTenError` 계층 사용 (00_data_models.md §11)
- import 순서: stdlib → third-party (torch, yaml, jinja2) → vten

### SystemVerilog
- `always_ff` / `always_comb` 구분 엄격
- DPI-C 호출은 반드시 sequential 블록에서만 (`always_ff` 또는 `task`)
- 파일명: `vten_` 접두사 (`vten_bfm_axi4s.sv`)
- 모든 모듈에 `\`include "vten_types.svh"` 포함

### C (DPI-C Bridge)
- C99 표준
- POSIX API만 사용 (`shm_open`, `sem_open`, `mmap`, `memcpy`)
- 모든 포인터 접근에 null check
- 에러 시 stderr에 로그, 반환값으로 에러 코드 전달

## Key Data Types (Quick Reference)

```python
# 00_data_models.md에서 정의 — 구현 시 정확히 일치시킬 것

class OpCode(Enum):      # §1.4 — SHM 인코딩 값
    LOAD=1, PUSH=2, PULL=3, STORE=4,
    WRITE_REG=5, READ_REG=6, POLL_REG=7,
    BARRIER=8, COMPARE=9

class Protocol(Enum):    # §1.1
    AXI4S="axi4_stream", AXI4="axi4", AXI4L="axi4_lite"

class Direction(Enum):   # §1.3
    HOST_TO_DEV, DEV_TO_HOST, BIDIRECTIONAL

# SHM Constants — §10.1
MAGIC = 0x5654454E       # "VTEN"
VERSION = 0x00000003     # v0.4.2
CONTROL_SIZE = 256
CMD_SLOT_SIZE = 64
STATS_SLOT_SIZE = 32
BUF_DESC_SIZE = 24
```

## Testing

```bash
# 단위 테스트 실행
pytest tests/ -v

# 특정 Phase 테스트
pytest tests/test_tensor.py tests/test_kernel.py -v   # Phase 1
pytest tests/test_runtime_*.py -v                       # Phase 2

# 커버리지
pytest tests/ --cov=vten --cov-report=term-missing
```

## External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Tensor, dtype, golden reference (conv3d 등) |
| `pyyaml` | ≥6.0 | kernel_spec.yaml 파싱 |
| `jinja2` | ≥3.1 | SV/TCL 코드 생성 (Phase 4) |
| `pytest` | ≥7.0 | 테스트 (dev dependency) |

## Vivado Commands (Phase 3+)

```bash
# SV 구문 검사
xvlog --sv vten_sv/*.sv

# DPI-C 공유 라이브러리 빌드
gcc -shared -fPIC -o build/lib/libvten_shm.so \
    vten_sv/vten_shm_bridge.c -lrt -lpthread

# Elaboration (DPI-C 링크 포함)
xelab tb_top --sv_lib build/lib/libvten_shm -timescale 1ns/1ps

# 시뮬레이션 실행
xsim tb_top --runall
```

## Multi-Directory Setup

vTen은 **라이브러리**이고, 사용자 설계는 별도 **프로젝트** 디렉토리에 존재한다.
RTL 소스가 크기 때문에 vten/ 안으로 옮기지 않는다.

```
/home/user/vten/          ← VTEN_ROOT (pip install -e .)
/home/user/my_npu/        ← PROJECT_ROOT (vten.toml이 있는 곳)
    ├── rtl/              # 대용량 RTL (이동 불가)
    ├── specs/            # kernel_spec.yaml
    ├── kernels/          # Kernel 클래스
    └── tests/            # TestScenario
```

**두 가지 경로 기준점:**
- `VTEN_ROOT`: `import vten`으로 해결. SV 라이브러리, 템플릿 위치.
- `PROJECT_ROOT`: `vten.toml` 위치. RTL, specs, kernels, tests, build 출력.

상세: `specs/path_resolution.md` 참조.

**구현 시 주의:**
- 모든 파일 경로는 `PROJECT_ROOT` 또는 `VTEN_ROOT` 기준 상대 경로로 해결
- 절대 경로를 하드코딩하지 않는다
- `kernel_spec.yaml`의 `rtl_top`은 PROJECT_ROOT 기준
- `vten.toml`의 `[rtl].sources`도 PROJECT_ROOT 기준
