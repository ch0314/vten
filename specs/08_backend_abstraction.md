# Backend Abstraction & Multi-Backend Support

**Version 0.1.0 — March 2026**
**Role: 백엔드 추상화 리팩토링 스펙. 기존 스펙(04, 06) 반영 전 설계 문서.**

---

## Table of Contents

1. [동기 및 목표](#1-동기-및-목표)
2. [백엔드 특성 비교](#2-백엔드-특성-비교)
3. [아키텍처 개요](#3-아키텍처-개요)
4. [핵심 인사이트: DSL → IR → 백엔드 해석](#4-핵심-인사이트-dsl--ir--백엔드-해석)
5. [Backend ABC 재설계](#5-backend-abc-재설계)
6. [XRT Backend 설계](#6-xrt-backend-설계)
7. [Verilator Backend 설계](#7-verilator-backend-설계)
8. [Build Pipeline 추상화](#8-build-pipeline-추상화)
9. [CLI 변경](#9-cli-변경)
10. [vten.toml 확장](#10-vtentoml-확장)
11. [Codegen 확장: XRT 호스트 코드 & Vitis TCL](#11-codegen-확장-xrt-호스트-코드--vitis-tcl)
12. [마이그레이션 계획](#12-마이그레이션-계획)

---

## 1. 동기 및 목표

### 1.1 현재 문제

현재 구현은 Vivado xsim에 강결합되어 있다:

- `cli/run.py`: `XsimBackend` 하드코딩
- `cli/build.py`: 5-stage 파이프라인 중 stage 1, 4, 5가 Vivado 전용
- `cli/init_cmd.py`: `[backend.xsim]` 템플릿 고정
- `codegen/sv_generator.py`: `BuildContext.vivado_path` 직접 참조
- `templates/*.tcl`: Vivado TCL 전용

### 1.2 목표

1. **Backend-agnostic 파이프라인**: DSL → IR까지의 컴파일 경로를 백엔드와 완전 분리
2. **XRT 백엔드** (우선): FPGA 실 배포를 위한 XRT/pyxrt 기반 실행
3. **Verilator 백엔드**: 오픈소스 시뮬레이션 지원
4. **Vitis 자동화**: kernel_spec.yaml 기반 IP/XO 패키징 TCL 자동 생성
5. **기존 xsim 기능 보존**: 하위 호환성

### 1.3 비목표

- xclbin 빌드 전체 자동화 (v++ 플로우는 사용자 책임)
- Vivado IP Integrator GUI 통합
- 실시간 디버그 프로토콜 (ILA/VIO 제어)

---

## 2. 백엔드 특성 비교

| | **xsim** | **verilator** | **xrt** |
|---|---|---|---|
| 대상 | Vivado 시뮬레이션 | 오픈소스 시뮬레이션 | 실 FPGA (Alveo/Versal) |
| 빌드 | Vivado proj → xvlog → xelab | verilator --cc → make | 사용자 xclbin (vTen은 TCL 생성) |
| Testbench | tb_top.sv + BFM | tb_top.sv + BFM | 불필요 |
| DPI-C | xsim 표준 | verilated_dpi.h 래퍼 | 불필요 |
| 데이터 전달 | POSIX SHM + Semaphore | POSIX SHM + Semaphore | XRT Buffer Object (PCIe DMA) |
| 제어 | SHM control region | SHM control region | XRT kernel API (MMIO) |
| 프로세스 | xsim subprocess | compiled binary | XRT 라이브러리 (in-process) |

**핵심 관찰:**
- xsim과 verilator는 런타임 프로토콜(SHM handshake)이 동일, 빌드만 다름
- XRT는 빌드와 런타임 모두 근본적으로 다름
- 그러나 **DSL 오퍼레이션 → IR Command 매핑**은 세 백엔드 모두 공유

---

## 3. 아키텍처 개요

```
User Code (Kernel + DSL)
    │
    ▼
ExecutionContext.run()
    │  Record pass: Operation 리스트 생성
    ▼
RuntimeEngine.compile()  ← Stage 0-6 (백엔드 무관)
    │  Stage 0: Composite Flatten
    │  Stage 1: Parameter Resolution
    │  Stage 2: Shape Validation
    │  Stage 3: Tensor Serialization
    │  Stage 4: Address Allocation
    │  Stage 5: auto_bind Resolution
    │  Stage 6: IR Lowering → Command[]
    │
    ├──[sim]── Stage 7: SHM Packing → bytes ──→ SimBackend (xsim/verilator)
    │          BFM Config synthesis                │  SHM handshake → BFMs → DUT
    │                                              ▼
    │                                         Verification
    │
    └──[hw]── CommandInterpreter ──→ XrtBackend
              IR Command를 XRT API로 직접 해석     │  BO alloc, DMA, MMIO
                                                   ▼
                                              Verification
```

### 3.1 디렉토리 구조 (변경 후)

```
vten/
├── backend/
│   ├── base.py              # Backend ABC (재설계)
│   ├── registry.py          # 🆕 백엔드 팩토리 + 디스커버리
│   ├── sim_base.py          # 🆕 SimBackend — SHM 핸드셰이크 공통
│   ├── xsim.py              # XsimBackend(SimBackend)
│   ├── verilator.py         # 🆕 VerilatorBackend(SimBackend)
│   └── xrt.py               # 🆕 XrtBackend(Backend)
├── build/                   # 🆕 빌드 파이프라인 모듈
│   ├── base.py              # BuildPipeline ABC
│   ├── common.py            # 공통 스테이지 (dpi_c, codegen)
│   ├── xsim_build.py        # Vivado 빌드 (stage 1, 4, 5)
│   ├── verilator_build.py   # 🆕 Verilator 빌드
│   └── xrt_build.py         # 🆕 TCL/스크립트 생성 (IP/XO 패키징)
├── codegen/
│   ├── sv_generator.py      # Testbench SV 생성 (sim 전용)
│   └── xrt_generator.py     # 🆕 XRT 호스트 코드/TCL 생성
├── cli/
│   ├── main.py              # --backend 플래그 추가
│   ├── build.py             # BuildPipeline에 위임 (리팩토링)
│   ├── run.py               # registry에서 백엔드 선택 (리팩토링)
│   └── init_cmd.py          # 백엔드별 템플릿 (리팩토링)
└── runtime/
    ├── engine.py            # Stage 0-6 유지, Stage 7 분리
    └── interpreter.py       # 🆕 IR Command 해석기 (XRT용)
```

---

## 4. 핵심 인사이트: DSL → IR → 백엔드 해석

### 4.1 기존 DSL 오퍼레이션이 XRT에 자연스럽게 매핑

현재 DSL의 OpKind/OpCode는 하드웨어 동작의 추상이다.
시뮬레이션에서는 BFM이 이를 수행하지만, 실 FPGA에서는 XRT API가 동일한 역할을 한다:

| OpCode | SIM (BFM) | XRT (FPGA) |
|--------|-----------|------------|
| `LOAD` | Host → SHM Data Region | Host → XRT BO `write()` |
| `PUSH` | BFM이 AXI4S로 DUT에 데이터 전달 | XRT BO `sync(TO_DEVICE)` + kernel arg 설정 |
| `PULL` | BFM이 AXI4S로 DUT 출력 캡처 | kernel 완료 후 BO `sync(FROM_DEVICE)` |
| `STORE` | SHM Data Region → Host | XRT BO `read()` |
| `WRITE_REG` | BFM이 AXI-Lite 트랜잭션 | `kernel.write_register(offset, value)` |
| `READ_REG` | BFM이 AXI-Lite 읽기 | `kernel.read_register(offset)` |
| `POLL_REG` | BFM 폴링 루프 | `while (read_register() & mask) != expected` |
| `BARRIER` | 스케줄러 글로벌 펜스 | Host-side fence (이전 DMA 완료 대기) |
| `COMPARE` | SHM 내 버퍼 비교 | Host 메모리에서 직접 비교 |

### 4.2 파이프라인 분기점

```
Stage 0-6: 모든 백엔드 공유 (IR Command[] 생성)
           │
           ├─ sim path: Stage 7 (SHM Packing) → SimBackend.submit(shm_image)
           │
           └─ hw path:  CommandInterpreter → XrtBackend.execute(commands, tensors)
```

**Stage 6까지의 IR Command는 백엔드 무관한 중간 표현(IR)**이다.
시뮬레이션은 이를 SHM 바이너리로 팩킹하여 하드웨어 스케줄러에 맡기고,
XRT는 이를 Python에서 순차적으로 해석하여 XRT API를 호출한다.

### 4.3 XRT Command Interpreter

```python
class CommandInterpreter:
    """IR Command 리스트를 XRT API 호출 시퀀스로 변환.

    SIM 경로에서 SHM Packer(Stage 7)가 하는 역할의 HW 대응물.
    차이점: SHM Packer는 바이너리 이미지를 만들고 HW 스케줄러가 실행하지만,
    CommandInterpreter는 Host(Python)에서 직접 실행한다.
    """

    def __init__(self, backend: XrtBackend):
        self._backend = backend
        self._buffers: dict[int, object] = {}  # buffer_id → XRT BO

    def execute(self, commands: list[Command], tensor_data: dict[int, bytes]) -> None:
        """IR 커맨드를 순서대로 실행. Dependency는 host-side에서 관리."""
        for cmd in commands:
            self._wait_deps(cmd)
            match cmd.op:
                case OpCode.LOAD:
                    self._exec_load(cmd, tensor_data)
                case OpCode.PUSH:
                    self._exec_push(cmd)
                case OpCode.PULL:
                    self._exec_pull(cmd)
                case OpCode.STORE:
                    self._exec_store(cmd)
                case OpCode.WRITE_REG:
                    self._exec_write_reg(cmd)
                case OpCode.READ_REG:
                    self._exec_read_reg(cmd)
                case OpCode.POLL_REG:
                    self._exec_poll_reg(cmd)
                case OpCode.BARRIER:
                    self._exec_barrier(cmd)
                case OpCode.COMPARE:
                    self._exec_compare(cmd)
```

---

## 5. Backend ABC 재설계

### 5.1 현재 인터페이스 (SIM 전용)

```python
class Backend(abc.ABC):
    def submit(self, shm_image: bytes, bfm_configs: list) -> None: ...
    def wait(self) -> BackendResult: ...
    def shutdown(self) -> None: ...
    def cleanup(self) -> None: ...
```

### 5.2 새 인터페이스 (통합)

```python
class Backend(abc.ABC):
    """모든 백엔드의 공통 인터페이스.

    CompiledResult를 받아 실행하고, BackendResult를 반환한다.
    SIM 백엔드는 SHM 경로를, HW 백엔드는 IR 해석 경로를 사용한다.
    """

    @abc.abstractmethod
    def execute(self, compiled: CompiledResult) -> BackendResult:
        """컴파일 결과를 실행하고 결과를 반환.

        이 메서드가 submit + wait를 통합한다.
        내부적으로:
          - SIM: SHM 이미지 쓰기 → 시뮬레이터 기동 → 핸드셰이크 → 결과 읽기
          - HW:  IR 커맨드 해석 → XRT API 호출 → 결과 수집
        """
        ...

    @abc.abstractmethod
    def cleanup(self) -> None:
        """리소스 정리. Idempotent."""
        ...

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    # ── 선택적 세부 제어 (SIM 백엔드에서 오버라이드) ──

    def submit(self, compiled: CompiledResult) -> None:
        """비동기 제출 (선택). execute()의 전반부."""
        raise NotImplementedError("Use execute() for synchronous operation")

    def wait(self) -> BackendResult:
        """결과 대기 (선택). execute()의 후반부."""
        raise NotImplementedError("Use execute() for synchronous operation")
```

### 5.3 CompiledResult 확장

현재 `CompiledResult`는 이미 필요한 모든 정보를 갖고 있다:

```python
@dataclass
class CompiledResult:
    commands: list[Command]         # IR 커맨드 (XRT용)
    shm_image: bytes                # SHM 바이너리 (SIM용)
    bfm_configs: list[BFMConfig]    # BFM 설정 (SIM용)
    buffer_ids: dict[str, int]      # 텐서 이름 → 버퍼 ID
    flattened_view: FlattenedKernelView
    probe_reports: list[ProbePoint]
```

XRT 백엔드는 `commands` + `flattened_view` (텐서 데이터 접근)를 사용하고,
SIM 백엔드는 `shm_image` + `bfm_configs`를 사용한다.

**Stage 7 조건부 실행**: XRT 백엔드 사용 시 SHM 패킹을 건너뛸 수 있다.
→ `RuntimeEngine.compile(target: str = "sim")` 파라미터 추가, 또는
→ `CompiledResult.shm_image`를 lazy 생성으로 전환.

### 5.4 BackendResult 확장

```python
@dataclass
class BackendResult:
    status: int
    error_code: int = 0
    error_cmd_id: int = 0
    error_message: str = ""
    stats: list[CmdStats] = field(default_factory=list)

    # 🆕 출력 텐서 데이터 (XRT 용)
    output_buffers: dict[int, bytes] = field(default_factory=dict)
    # buffer_id → raw bytes. SIM에서는 SHM에서 읽고, XRT에서는 BO에서 읽음.

    def read_buffer(self, buffer_id: int) -> bytes:
        """Read output tensor data by buffer ID."""
        return self.output_buffers.get(buffer_id, b"")
```

### 5.5 클래스 계층

```
Backend (ABC)
├── SimBackend (ABC) — SHM 핸드셰이크 공통 로직
│   ├── XsimBackend — Vivado xsim 프로세스 관리
│   └── VerilatorBackend — Verilator 바이너리 관리
└── XrtBackend — XRT/pyxrt 기반 FPGA 실행
```

---

## 6. XRT Backend 설계

### 6.1 개요

XrtBackend는 pyxrt (Xilinx Runtime Python 바인딩)를 사용하여 FPGA와 통신한다.
기존 DSL 오퍼레이션이 XRT API 호출로 직접 매핑된다.

### 6.2 XRT 리소스 모델

```
FPGA Device
├── xclbin (사용자가 빌드)
├── Kernel Object (CU — Compute Unit)
│   ├── Control Registers (AXI4-Lite) ← WRITE_REG, READ_REG, POLL_REG
│   └── Memory Arguments (AXI4 포트) ← kernel arg로 BO 연결
└── Buffer Objects (BO)
    ├── Input BO ← LOAD → PUSH (sync TO_DEVICE)
    └── Output BO ← PULL (sync FROM_DEVICE) → STORE
```

### 6.3 OpCode → XRT API 매핑 상세

#### LOAD (Host → Device Buffer)

```python
def _exec_load(self, cmd: Command, tensor_data: dict[int, bytes]):
    data = tensor_data[cmd.buffer_id]
    bo = xrt.bo(self._device, len(data), xrt.bo.flags.normal, self._mem_group)
    bo.write(data)
    self._buffers[cmd.buffer_id] = bo
```

#### PUSH (Device Buffer → Kernel/DUT)

AXI4-Stream과 AXI4는 FPGA에서 다르게 동작:
- **AXI4 (Memory-Mapped)**: BO를 kernel argument로 설정 + `sync(TO_DEVICE)`
- **AXI4-Stream**: Vitis에서는 보통 AXI4 BO를 DMA를 통해 스트림으로 변환

```python
def _exec_push(self, cmd: Command):
    bo = self._buffers[cmd.buffer_id]
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    # kernel arg 설정은 별도 매핑 테이블 참조
    self._kernel.set_arg(self._arg_index(cmd), bo)
```

#### WRITE_REG / READ_REG / POLL_REG

XRT kernel 오브젝트의 레지스터 직접 접근:

```python
def _exec_write_reg(self, cmd: Command):
    self._kernel.write_register(cmd.reg_offset, cmd.reg_value)

def _exec_read_reg(self, cmd: Command):
    value = self._kernel.read_register(cmd.reg_offset)
    cmd.reg_value = value  # 결과 저장

def _exec_poll_reg(self, cmd: Command):
    timeout = self._config.get("poll_timeout_ms", 10000)
    start = time.monotonic()
    while True:
        val = self._kernel.read_register(cmd.reg_offset)
        if (val & cmd.reg_mask) == cmd.reg_expected:
            break
        if (time.monotonic() - start) * 1000 > timeout:
            raise PollTimeoutError(f"POLL_REG timeout at offset 0x{cmd.reg_offset:X}")
```

#### PULL (Kernel/DUT → Device Buffer) + STORE (Device → Host)

```python
def _exec_pull(self, cmd: Command):
    bo = self._buffers[cmd.buffer_id]
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

def _exec_store(self, cmd: Command):
    bo = self._buffers[cmd.buffer_id]
    data = bo.read(cmd.size)
    self._output_buffers[cmd.buffer_id] = bytes(data)
```

#### BARRIER

```python
def _exec_barrier(self, cmd: Command):
    # Host-side fence: 모든 pending DMA 완료 대기
    self._device.sync()  # 또는 개별 BO sync 확인
```

### 6.4 XrtBackend 클래스

```python
class XrtBackend(Backend):
    """XRT/pyxrt 기반 FPGA 백엔드.

    vten.toml [backend.xrt] 설정:
      xclbin_path: xclbin 파일 경로
      device_index: FPGA 디바이스 인덱스 (기본 0)
      kernel_name: xclbin 내 커널 이름
      memory_bank: 메모리 뱅크 (기본 자동 탐색)
    """

    def __init__(self, project_config: dict):
        xrt_cfg = project_config.get("backend", {}).get("xrt", {})
        self._xclbin_path = xrt_cfg["xclbin_path"]
        self._device_index = xrt_cfg.get("device_index", 0)
        self._kernel_name = xrt_cfg.get("kernel_name", "")
        self._poll_timeout_ms = xrt_cfg.get("poll_timeout_ms", 10000)

        # XRT 리소스 (lazy init)
        self._device = None
        self._xclbin = None
        self._kernel = None
        self._interpreter = None

    def _init_device(self):
        """FPGA 디바이스 초기화. 최초 execute() 호출 시 수행."""
        import pyxrt
        self._device = pyxrt.device(self._device_index)
        self._xclbin = pyxrt.xclbin(self._xclbin_path)
        self._device.load_xclbin(self._xclbin)
        self._kernel = pyxrt.kernel(
            self._device, self._xclbin.get_uuid(), self._kernel_name
        )
        self._interpreter = CommandInterpreter(self)

    def execute(self, compiled: CompiledResult) -> BackendResult:
        if self._device is None:
            self._init_device()

        # IR 커맨드에서 텐서 데이터 추출
        tensor_data = self._extract_tensor_data(compiled)

        # CommandInterpreter로 실행
        self._interpreter.execute(compiled.commands, tensor_data)

        # 결과 수집
        return BackendResult(
            status=0,  # success
            output_buffers=self._interpreter.output_buffers,
        )

    def cleanup(self):
        # XRT 리소스 해제 (BO 등)
        self._interpreter = None
        self._kernel = None
        self._device = None
```

### 6.5 kernel_spec.yaml의 XRT 확장

인터페이스와 XRT kernel argument 사이의 매핑이 필요하다:

```yaml
# kernels/conv3d/kernel_spec.yaml
kernel: conv3d
rtl_top: rtl/conv3d_wrapper.sv

interfaces:
  s_ifm:
    protocol: axi4_stream
    data_width: 256
    tensor: ifm
    # 🆕 XRT 매핑
    xrt:
      arg_index: 0          # kernel.set_arg(0, bo)
      # 또는
      arg_name: "ifm"       # xclbin 메타데이터에서 이름으로 매핑

  m_ofm:
    protocol: axi4_stream
    data_width: 256
    tensor: ofm
    xrt:
      arg_index: 1

  s_axi_control:
    protocol: axi4_lite
    registers: { ... }
    xrt:
      # AXI4-Lite는 별도 매핑 불필요 — XRT kernel.write_register()가 직접 접근
      # register offset은 기존 spec과 동일
```

### 6.6 Interface ↔ XRT Argument 자동 매핑

대부분의 경우 xclbin의 XML 메타데이터에서 자동 매핑 가능:

```python
def _auto_map_arguments(self, xclbin, kernel_spec):
    """xclbin 메타데이터에서 interface → kernel arg 자동 매핑."""
    xclbin_args = self._parse_xclbin_metadata(xclbin)
    # xclbin의 각 arg에 대해 kernel_spec.interfaces와 이름 매칭
    # 매칭 실패 시 kernel_spec.yaml의 xrt.arg_index 사용
```

---

## 7. Verilator Backend 설계

### 7.1 개요

Verilator 백엔드는 xsim과 동일한 SHM 핸드셰이크 프로토콜을 사용한다.
차이점은 빌드 파이프라인과 DPI-C 통합뿐이다.

### 7.2 SimBackend 공통 클래스

xsim과 verilator가 공유하는 SHM 핸드셰이크 로직을 `SimBackend`로 추출:

```python
class SimBackend(Backend):
    """SHM 핸드셰이크 기반 시뮬레이션 백엔드 공통 클래스.

    POSIX SHM + Named Semaphore 프로토콜을 구현한다 (04_backend_xsim.md §3).
    시뮬레이터별 차이(프로세스 기동)는 하위 클래스에서 오버라이드.
    """

    def execute(self, compiled: CompiledResult) -> BackendResult:
        self._submit_shm(compiled.shm_image, compiled.bfm_configs)
        result = self._wait_completion()
        self._shutdown_sim()
        return result

    # SHM 핸드셰이크 (현재 XsimBackend에서 추출)
    def _submit_shm(self, shm_image, bfm_configs): ...
    def _wait_completion(self) -> BackendResult: ...
    def _shutdown_sim(self): ...

    # 시뮬레이터 프로세스 관리 (하위 클래스 오버라이드)
    @abc.abstractmethod
    def _start_simulator(self) -> subprocess.Popen: ...

    @abc.abstractmethod
    def _simulator_args(self) -> list[str]: ...
```

### 7.3 VerilatorBackend

```python
class VerilatorBackend(SimBackend):
    """Verilator 컴파일 바이너리 기반 시뮬레이션."""

    def _start_simulator(self) -> subprocess.Popen:
        binary = self._config["_kernel_build_dir"] + "/Vtb_top"
        cmd = [
            binary,
            f"+SESSION_ID={self._session_id}",
            f"+TIMEOUT_MS={self._timeout_ms}",
        ]
        return subprocess.Popen(cmd, ...)
```

### 7.4 Verilator DPI-C 차이

Verilator는 DPI-C를 C++ 래퍼를 통해 호출한다:

```c
// vten_sv/vten_shm_bridge_verilator.cpp (신규)
#include "verilated_dpi.h"
#include "vten_shm_bridge.h"

// 기존 C 함수를 Verilator DPI로 래핑
extern "C" {
    // DPI export 함수들은 동일한 시그니처
    // 내부적으로 vten_shm_bridge.c의 함수 호출
}
```

기존 `vten_shm_bridge.c`의 핵심 로직은 변경 없이 재사용.

---

## 8. Build Pipeline 추상화

### 8.1 BuildPipeline ABC

```python
class BuildPipeline(abc.ABC):
    """백엔드별 빌드 파이프라인 추상 클래스."""

    def __init__(self, project: Path, config: dict):
        self.project = project
        self.config = config
        self.vten_root = Path(__file__).resolve().parent.parent.parent

    @abc.abstractmethod
    def stages(self) -> list[str]:
        """이 파이프라인의 스테이지 이름 목록."""
        ...

    @abc.abstractmethod
    def run_stage(self, stage: str, kernel_dir: Path | None, **kwargs) -> None:
        """개별 스테이지 실행."""
        ...

    def build(self, kernel_name: str | None = None, **kwargs) -> None:
        """전체 빌드 실행 (orchestrator)."""
        ...
```

### 8.2 백엔드별 빌드 스테이지

#### XsimBuildPipeline (현재 build.py 리팩토링)

| Stage | 이름 | 스코프 | 도구 |
|-------|------|--------|------|
| 1 | `project_setup` | 프로젝트 | Vivado TCL |
| 2 | `dpi_c` | 프로젝트 | gcc |
| 3 | `codegen` | 커널별 | Jinja2 → tb_top.sv |
| 4 | `compile_order` | 커널별 | Vivado get_compile_order |
| 5 | `compile` | 커널별 | xvlog + xelab |

#### VerilatorBuildPipeline

| Stage | 이름 | 스코프 | 도구 |
|-------|------|--------|------|
| 1 | `dpi_c` | 프로젝트 | gcc (동일) |
| 2 | `codegen` | 커널별 | Jinja2 → tb_top.sv (약간 다른 템플릿) |
| 3 | `verilate` | 커널별 | `verilator --cc --exe` |
| 4 | `make` | 커널별 | `make -C obj_dir` |

#### XrtBuildPipeline

| Stage | 이름 | 스코프 | 도구 |
|-------|------|--------|------|
| 1 | `gen_packaging_tcl` | 커널별 | Jinja2 → IP 패키징 TCL |
| 2 | `gen_xo_tcl` | 커널별 | Jinja2 → XO 생성 TCL |
| 3 | `gen_link_cfg` | 프로젝트 | connectivity config 생성 |
| 4 | `validate` | 커널별 | xclbin 메타데이터 검증 (선택) |

**XRT 빌드는 xclbin을 만들지 않는다.** 사용자가 `v++`로 직접 빌드한다.
vTen은 kernel_spec.yaml의 정보로 **빌드에 필요한 TCL/config 파일**을 생성한다.

### 8.3 XRT 빌드 산출물

```
kernels/conv3d/build/
├── packaging/
│   ├── package_ip.tcl          # IP 패키징 Vivado TCL
│   ├── component.xml           # 참조용 (v++ 입력)
│   └── xo_gen.tcl              # XO 생성 TCL
├── link/
│   └── connectivity.cfg        # v++ --link 입력 (SP 매핑)
└── xrt/
    └── xrt.ini                 # XRT 런타임 설정
```

---

## 9. CLI 변경

### 9.1 --backend 플래그

```bash
# 명시적 백엔드 선택
$ vten build --backend xsim          # Vivado xsim (기본)
$ vten build --backend verilator
$ vten build --backend xrt

$ vten run --kernel conv3d --test test_conv3d --backend xrt

# 기본값: vten.toml의 [project].default_backend, 없으면 "xsim"
```

### 9.2 build 서브커맨드

```bash
# xsim/verilator: 기존과 동일
$ vten build --backend xsim --stage codegen

# xrt: 패키징 파일 생성
$ vten build --backend xrt                    # 전체: TCL + config 생성
$ vten build --backend xrt --stage gen_packaging_tcl
```

### 9.3 init 서브커맨드

#### 신규 프로젝트 생성

```bash
$ vten init my_project                        # 기본 (xsim 섹션만 포함)
$ vten init my_project --backend xrt          # xrt 섹션만 포함
$ vten init my_project --backend verilator    # verilator 섹션만 포함
```

`--backend`에 따라 `vten.toml`에 해당 백엔드 섹션만 생성한다.
`default_backend`도 지정된 백엔드로 설정된다.

#### 기존 프로젝트에 백엔드 추가

```bash
$ vten init my_project --add-backend xrt      # 기존 vten.toml에 [backend.xrt] 추가
$ vten init my_project --add-backend verilator
```

기존 `vten.toml`이 존재하면 파일 끝에 해당 백엔드 섹션을 **append**한다.
이미 해당 섹션이 있으면 에러 (덮어쓰기 방지):

```
Error: [backend.xrt] section already exists in vten.toml
```

#### 생성되는 vten.toml 예시

`vten init my_proj --backend xrt`:

```toml
[project]
name = "my_proj"
version = "0.1.0"
default_backend = "xrt"

[parameters]

[backend.xrt]
xclbin_path = "build/kernel.xclbin"
device_index = 0
kernel_name = ""
poll_timeout_ms = 30000

[rtl]
sources = ["rtl/**/*.sv", "rtl/**/*.v"]
include_dirs = ["rtl/include"]

[test]
default_seed = 42
waveform = false
waveform_on_fail = true
```

이후 `vten init my_proj --add-backend xsim` 실행 시 아래가 append:

```toml
[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0
submit_timeout_s = 300
```

#### 백엔드별 디렉토리 구조

`--backend`에 따라 생성되는 초기 디렉토리도 달라진다:

| 백엔드 | 생성 디렉토리 |
|--------|--------------|
| xsim | `build/vivado_proj/`, `build/lib/` |
| verilator | `build/lib/` |
| xrt | `build/` (xclbin 출력용) |
| 공통 | `rtl/`, `kernels/`, `results/` |

`ip/` 디렉토리는 xsim과 xrt에서만 생성 (verilator는 Vivado IP 미지원).

#### 백엔드별 템플릿 레지스트리

```python
# init_cmd.py 내부
_BACKEND_TOML_TEMPLATES: dict[str, str] = {
    "xsim": """\
[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0
submit_timeout_s = 300
""",
    "xrt": """\
[backend.xrt]
xclbin_path = "build/kernel.xclbin"
device_index = 0
kernel_name = ""
poll_timeout_ms = 30000
""",
    "verilator": """\
[backend.verilator]
verilator_path = ""
threads = 4
trace = false
opt_level = 3
""",
}

_BACKEND_DIRS: dict[str, list[str]] = {
    "xsim":      ["build/vivado_proj", "build/lib", "ip"],
    "verilator": ["build/lib"],
    "xrt":       ["build", "ip"],
}
```

### 9.4 Backend Registry

```python
# vten/backend/registry.py

_BACKEND_MAP: dict[str, tuple[str, str]] = {
    "xsim":      ("vten.backend.xsim",      "XsimBackend"),
    "verilator": ("vten.backend.verilator",  "VerilatorBackend"),
    "xrt":       ("vten.backend.xrt",        "XrtBackend"),
}

_BUILD_MAP: dict[str, tuple[str, str]] = {
    "xsim":      ("vten.build.xsim_build",      "XsimBuildPipeline"),
    "verilator": ("vten.build.verilator_build",  "VerilatorBuildPipeline"),
    "xrt":       ("vten.build.xrt_build",        "XrtBuildPipeline"),
}

def get_backend(name: str, config: dict) -> Backend:
    """백엔드 인스턴스 생성 (lazy import)."""
    module_path, class_name = _BACKEND_MAP[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(config)

def get_build_pipeline(name: str, project: Path, config: dict) -> BuildPipeline:
    """빌드 파이프라인 인스턴스 생성 (lazy import)."""
    module_path, class_name = _BUILD_MAP[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(project, config)
```

---

## 10. vten.toml 확장

### 10.1 프로젝트 기본 설정

```toml
[project]
name = "my_npu"
version = "0.1.0"
default_backend = "xrt"     # 🆕 기본 백엔드 (미지정 시 "xsim")

[parameters]
Ti = 32
To = 32
```

### 10.2 XRT 백엔드 설정

```toml
[backend.xrt]
xclbin_path = "build/conv3d.xclbin"     # xclbin 경로 (PROJECT_ROOT 기준)
device_index = 0                         # FPGA 디바이스 인덱스
kernel_name = "conv3d_top"               # xclbin 내 커널 이름 (자동 탐색 가능)
memory_bank = "HBM[0]"                   # 메모리 뱅크 (선택, 자동 탐색 가능)
poll_timeout_ms = 30000                  # POLL_REG 타임아웃
```

### 10.3 Verilator 백엔드 설정

```toml
[backend.verilator]
verilator_path = "/usr/bin/verilator"    # verilator 바이너리 경로
threads = 4                              # 멀티스레드 시뮬레이션
trace = false                            # VCD 파형 덤프
opt_level = 3                            # 최적화 레벨 (-O3)
extra_args = ["--timing"]                # 추가 verilator 인자
```

### 10.4 기존 xsim 설정 (변경 없음)

```toml
[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0
submit_timeout_s = 300
```

---

## 11. Codegen 확장: XRT 호스트 코드 & Vitis TCL

### 11.1 IP 패키징 TCL 생성

kernel_spec.yaml의 인터페이스 정보로 Vivado IP Packager TCL을 자동 생성:

```tcl
# templates/package_ip.tcl.j2
# kernel_spec.yaml로부터 자동 생성

create_project package_ip ./package_ip -force
add_files [glob {{ rtl_sources | join(" ") }}]
update_compile_order -fileset sources_1

ipx::package_project -root_dir ./package_ip -vendor user.org -taxonomy /vten

# 인터페이스 선언 (kernel_spec.yaml에서 유도)
{% for name, iface in interfaces.items() %}
{% if iface.protocol == "axi4_stream" %}
ipx::add_bus_interface {{ name }} [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 \
    [ipx::get_bus_interfaces {{ name }}]
{% elif iface.protocol == "axi4" %}
ipx::add_bus_interface {{ name }} [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:aximm_rtl:1.0 \
    [ipx::get_bus_interfaces {{ name }}]
{% elif iface.protocol == "axi4_lite" %}
# AXI4-Lite slave for control registers
ipx::add_bus_interface {{ name }} [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:aximm_rtl:1.0 \
    [ipx::get_bus_interfaces {{ name }}]
set_property interface_mode slave [ipx::get_bus_interfaces {{ name }}]
{% endif %}
{% endfor %}

# 레지스터 맵 (kernel_spec의 registers 섹션에서 유도)
{% for reg in registers %}
ipx::add_register {{ reg.name }} -offset {{ reg.offset }} \
    [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps ...]]
{% endfor %}

set_property core_revision 1 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::save_core [ipx::current_core]
```

### 11.2 XO 생성 TCL

```tcl
# templates/gen_xo.tcl.j2
# .xo 파일 생성 (v++ --compile 대안)

package_xo -xo_path {{ xo_path }} \
    -kernel_name {{ kernel_name }} \
    -ip_directory ./package_ip \
    -kernel_xml {{ kernel_xml_path }}
```

### 11.3 Connectivity Config 생성

v++ --link에 필요한 SP(Stream Port) 매핑 파일:

```ini
# templates/connectivity.cfg.j2
[connectivity]
# AXI4 메모리 인터페이스 매핑
{% for name, iface in interfaces.items() if iface.protocol == "axi4" %}
sp={{ kernel_name }}_1.{{ name }}:{{ iface.xrt.memory_bank | default("HBM[0]") }}
{% endfor %}

# 스트림 연결 (멀티 커널 시)
{% for conn in stream_connections %}
stream_connect={{ conn.source }}:{{ conn.dest }}
{% endfor %}
```

### 11.4 kernel.xml 생성

xclbin 빌드에 필요한 커널 메타데이터 XML:

```xml
<!-- templates/kernel.xml.j2 -->
<?xml version="1.0" encoding="UTF-8"?>
<root versionMajor="1" versionMinor="0">
  <kernel name="{{ kernel_name }}" language="ip" type="rtl">
    <ports>
      {% for name, iface in interfaces.items() %}
      {% if iface.protocol == "axi4" %}
      <port name="{{ name }}" mode="master" range="0xFFFFFFFF" dataWidth="{{ iface.data_width }}"/>
      {% elif iface.protocol == "axi4_lite" %}
      <port name="{{ name }}" mode="slave" range="0x{{ '%04X' % iface.addr_range }}" dataWidth="{{ iface.data_width }}"/>
      {% elif iface.protocol == "axi4_stream" %}
      <port name="{{ name }}" mode="{{ 'read_only' if iface.role == 'slave' else 'write_only' }}" dataWidth="{{ iface.data_width }}"/>
      {% endif %}
      {% endfor %}
    </ports>
    <args>
      {% for name, iface in interfaces.items() if iface.protocol != "axi4_lite" %}
      <arg name="{{ name }}" addressQualifier="{{ '4' if iface.protocol == 'axi4_stream' else '1' }}"
           port="{{ name }}" size="0x{{ '%X' % iface.buffer_size }}"/>
      {% endfor %}
    </args>
  </kernel>
</root>
```

---

## 12. 마이그레이션 계획

### 12.1 Phase A: Backend 추상화 리팩토링 (기반 작업)

**선행 조건**: 기존 테스트 전부 통과 상태

1. `Backend.execute(CompiledResult)` 인터페이스 도입
2. `SimBackend` 중간 클래스 추출 (xsim.py에서 SHM 로직 분리)
3. `XsimBackend(SimBackend)` 리팩토링
4. `backend/registry.py` 구현
5. `cli/run.py`, `cli/build.py`, `cli/main.py`에 `--backend` 플래그 추가
6. `cli/init_cmd.py` 백엔드별 템플릿 분기

**완료 기준**: 기존 xsim 테스트 전부 통과 + `--backend xsim` 명시 시 동일 동작

### 12.2 Phase B: XRT Backend 구현 (우선)

1. `runtime/interpreter.py` — CommandInterpreter 구현
2. `backend/xrt.py` — XrtBackend 구현
3. `build/xrt_build.py` — TCL/config 생성 파이프라인
4. `codegen/xrt_generator.py` — IP 패키징 TCL, kernel.xml, connectivity.cfg
5. Jinja2 템플릿 추가 (package_ip.tcl.j2, gen_xo.tcl.j2, kernel.xml.j2, connectivity.cfg.j2)
6. `kernel_spec.yaml` 스키마 확장 (xrt 섹션)
7. 단위 테스트 + passthrough 예제 XRT 테스트

**완료 기준**: `vten build --backend xrt` → TCL 생성 성공,
             `vten run --backend xrt` → FPGA 실 실행 (Alveo 보드 필요)

### 12.3 Phase C: Verilator Backend 구현

1. `backend/verilator.py` — VerilatorBackend(SimBackend)
2. `build/verilator_build.py` — Verilator 빌드 파이프라인
3. DPI-C 래퍼 (vten_shm_bridge_verilator.cpp)
4. Verilator용 tb_top.sv 변형 (또는 조건부 생성)
5. 단위 테스트 + passthrough 예제 Verilator 테스트

**완료 기준**: `vten build --backend verilator && vten run --backend verilator` 동작

### 12.4 기존 스펙 반영

리팩토링 완료 후 기존 스펙에 변경 사항 반영:

- `04_backend_xsim.md` → SimBackend / XsimBackend 분리 반영
- `06_codegen_and_cli.md` → --backend 플래그, 빌드 파이프라인 추상화 반영
- `00_data_models.md` → BackendResult.output_buffers 추가, CompiledResult 확장
- `03_kernel_spec_schema.md` → xrt 섹션 스키마 추가

---

## 부록 A: XRT 의존성

```
# pyproject.toml 추가 (optional dependency)
[project.optional-dependencies]
xrt = ["pyxrt"]        # Xilinx Runtime Python 바인딩
verilator = []         # 시스템 패키지, pip 설치 불가
```

`pyxrt`는 Xilinx가 배포하는 XRT 설치 시 포함된다 (`/opt/xilinx/xrt/python/`).
PyPI에 없으므로 `sys.path` 추가 또는 XRT 환경 설정 필요:

```bash
source /opt/xilinx/xrt/setup.sh
```

## 부록 B: 유저 워크플로우 비교

### SIM 워크플로우 (xsim/verilator)

```bash
# 1. 빌드
$ vten build --backend xsim

# 2. 테스트 실행
$ vten run --kernel conv3d --test test_conv3d --backend xsim

# 3. 결과 확인
$ vten report
```

### XRT 워크플로우 (FPGA)

```bash
# 1. IP/XO 패키징 파일 생성 (vTen)
$ vten build --backend xrt

# 2. xclbin 빌드 (사용자, v++ 사용)
$ v++ --compile -t hw --kernel conv3d -o conv3d.xo \
      --package.kernel_dir kernels/conv3d/build/packaging
$ v++ --link -t hw --config kernels/conv3d/build/link/connectivity.cfg \
      -o build/conv3d.xclbin conv3d.xo

# 3. FPGA 실행 (vTen)
$ vten run --kernel conv3d --test test_conv3d --backend xrt

# 4. 결과 확인 (동일)
$ vten report
```

사용자 코드(Kernel 클래스, DSL 시나리오)는 **백엔드에 관계없이 동일**하다.
`generate_inputs()`, `forward()`, `send_tensor()`, `recv_tensor()` 등
모든 DSL 호출이 SIM과 HW에서 동일하게 동작한다.
