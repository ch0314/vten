# XRT HW Emulation 디버깅 로그

**Updated: 2026-04-05**
**Target: NPU_3D 6-kernel composite pipeline on Xilinx U280**

---

## 목적

XRT hw_emu 백엔드를 통한 NPU_3D E2E 검증 과정에서 발견된 이슈와 해결책을 기록한다.
향후 다른 커널이나 플랫폼에서 동일 문제를 빠르게 진단하기 위한 참조 문서.

---

## 1. pybind11 바인딩 이슈 (Phase B)

### 1.1 `bo.sync()` TypeError — int vs C enum

- **증상**: `bo.sync(0)` 호출 시 TypeError
- **원인**: pybind11에서 `xclBOSyncDirection`은 C++ enum이지만, Python 측에서 int를 전달
- **해결**: `vten_xrt.cpp`에서 lambda 래퍼로 `static_cast<xclBOSyncDirection>(int)` 적용.
  sync direction 상수는 `py::int_()` submodule 속성으로 노출
- **파일**: `vten/xrt_binding/vten_xrt.cpp`

### 1.2 pybind11 enum 등록 순서

- **증상**: `ImportError: arg(): could not convert default argument`
- **원인**: `xrt::kernel` 생성자의 default argument `cu_access_mode::shared`는
  해당 enum이 먼저 등록되어야 변환 가능
- **해결**: `py::enum_<cu_access_mode>`를 `py::class_<xrt::kernel>` 앞에 등록
- **파일**: `vten/xrt_binding/vten_xrt.cpp`

### 1.3 `vten_xrt.so` import 경로

- **증상**: 빌드된 `.so`가 `vten/xrt_binding/build/`에 있으면 `import vten_xrt` 실패
- **해결**: `scripts/build_xrt_binding.sh`에서 site-packages에 자동 설치

### 1.4 테스트 segfault — vten_xrt mock

- **증상**: `vten_xrt`가 설치된 환경에서 XRT-없는 테스트가 segfault
- **원인**: `import vten_xrt` 성공 후 `vten_xrt.device(0)` 호출 시 XRT 런타임 없이 crash
- **해결**: `patch.dict("sys.modules", {"pyxrt": None, "vten_xrt": None})`로 둘 다 패치
- **파일**: `tests/test_build_xrt.py`

---

## 2. XRT 빌드/런타임 이슈

### 2.1 "Transaction cannot be routed" — Wrapper 파라미터화

- **증상**: hw_emu에서 AXI 트랜잭션 라우팅 실패
- **원인**: `wrapper.sv`의 DATA_W/ADDR_W가 하드코딩되어 Vivado가 address space를
  올바르게 추론하지 못함
- **해결**: `wrapper.sv.j2`에서 DATA_W/ADDR_W를 kernel_spec 기반 파라미터로 생성
- **커밋**: `64af85c`

### 2.2 `xrt.kernel` / `xrt.ip` exclusive access 충돌

- **증상**: `xrt.kernel` 생성 후 `xrt.ip` 생성 시 CU access 충돌
- **원인**: `xrt.kernel`이 CU exclusive access를 잡고 있어 `xrt.ip`가 같은 CU에 접근 불가
- **해결**: `xrt.kernel`로 `group_id()` 쿼리 후 delete, 그 다음 `xrt.ip` 생성.
  group_ids 캐시하여 재사용
- **파일**: `vten/backend/xrt.py` `_init_device()`

### 2.3 U280 DDR[0] = memory group 32 (not 0)

- **증상**: BO를 group 0에 할당하면 커널이 DDR에서 읽지 못해 출력이 전부 0
- **원인**: U280에서 DDR[0]의 실제 XRT group은 32. group 0은 HBM[0].
  FPGA 플랫폼마다 memory group 매핑이 다름
- **해결**: `xrt::kernel.group_id(arg_index)`로 런타임에 정확한 bank group 쿼리.
  `XrtBackend._build_mem_bank_map()`에서 구현
- **교훈**: BO 할당 시 항상 `kernel.group_id()` 우선 사용.
  `xrt.arg_index`를 kernel_spec에 반드시 포함
- **파일**: `vten/backend/xrt.py`

---

## 3. AP_CTRL_NONE / user_managed 커널 지원

### 3.1 `xrt.kernel.group_id()` 실패

- **증상**: NPU 3D 커널은 `ap_ctrl_hs` 없이 레지스터로만 제어됨.
  `xrt.kernel.group_id()` 호출 시 예외 발생
- **원인**: XRT는 `AP_CTRL_NONE` 커널에 대해 `group_id()` API를 지원하지 않음
- **해결**:
  1. `kernel.xml.j2`: `language="ip"` + `hwControlProtocol="user_managed"` 속성 추가
  2. `xrt.py`: `_parse_xclbin_connectivity()` fallback 구현 —
     xclbinutil로 IP_LAYOUT + CONNECTIVITY JSON 파싱하여 arg→mem_index 매핑
  3. 런타임에 `xrt.ip` 사용 (raw register access, `ap_ctrl_hs` 가정 없음)
- **파일**: `templates/kernel.xml.j2`, `vten/backend/xrt.py`, `vten/codegen/xrt_generator.py`

---

## 4. Codegen 이슈 — 레지스터 포트 폭 불일치

### 4.1 AXI-Lite 레지스터 포트 narrow vs 32-bit

- **증상**: xsim에서 `reg_layer_done` 읽기가 `0xDEADBEEF` 반환 (axilite_ctrl default case)
- **원인**: 생성된 `axilite_ctrl`이 레지스터를 `reg.width` (1-bit, 8-bit 등)으로 선언하는데,
  core RTL은 모두 32-bit 포트 사용. 좁은 wire → 32-bit port 연결 시 xelab VRFC 10-9543 경고 대량 발생
- **해결 (template 수정)**:
  - `axilite_ctrl.sv.j2`: 모든 레지스터 포트를 `[DATA_W-1:0]` (32-bit)로 통일.
    write decode에서 `s_wdata` 전체 사용. read decode에서 zero-padding 제거.
    reset 값도 `'0` 사용
  - `wrapper.sv.j2`: 레지스터 wire를 `[reg.width-1:0]` 대신 `[dw-1:0]` (DATA_W)로 통일
- **파일**: `templates/axilite_ctrl.sv.j2`, `templates/wrapper.sv.j2`
- **주의**: 수정 후 `vten build` 재실행 필요 (xsim/XRT 모두)

---

## 5. HW Emulation 런타임 이슈

### 5.1 BO 초기화 — hw_emu IPC 호환성

- **증상**: hw_emu에서 output BO 데이터가 garbage
- **원인**: 원본 C++ host code는 항상 `bo.map<T*>()` + `memset(0)` 후 `sync(TO_DEVICE)`.
  vTen은 output BO를 초기화하지 않았음. hw_emu IPC 모델에서 uninitialized device memory 접근 시 문제
- **해결**:
  - pre-allocate 단계에서 output BO도 `b"\x00" * size`로 zero-init + `sync(TO_DEVICE)`
  - `map_init()` 호출 추가 (hw_emu IPC memory tracking용)
  - STORE 단계에서 `map_read()` 선호 (원본 host code 패턴 매치)
- **파일**: `vten/runtime/interpreter.py`

### 5.2 `recv_tensor` 실행 순서 → emulator crash

- **증상**: hw_emu ~18분 후 xsim crash
  ```
  [libprotobuf ERROR] Can't parse message of type "xclReadAddrKernelCtrl_response"
  ERROR: [HW-EMU 22] xclRegRW - xclRead failed for CU: 0
  ```
- **원인**: `npu_pipeline_kernel.py`에서 `recv_tensor` (`bo.sync(FROM_DEVICE)`)가
  `poll_register`보다 먼저 실행. DUT가 DDR[0]에 쓰는 중에 host가 DDR[0] read →
  bus contention → emulator 내부 상태 손상
- **해결**: dependency chain 수정
  ```python
  # Before: cfg → vsync → PULL → POLL → POLL (broken)
  # After:  cfg → vsync → POLL → POLL → PULL (correct)
  h_started = ctx.poll_register(..., expected=0, dep=h_vsync)
  h_done    = ctx.poll_register(..., dep=h_started)
  h_ofm     = ctx.recv_tensor(self.ofm_mem, dep=h_done)
  ```
- **파일**: `npu_pipeline_kernel.py`

### 5.3 빌드 캐시 무효화 실패

- **증상**: RTL 변경 후에도 이전 xclbin이 사용됨
- **원인**: `fmapIO.xo`만 삭제했지만 `ip_repo/`, `package_ip_proj/` 캐시가 남아있음
- **해결**: `ip_repo/`, `package_ip_proj/`, `*.xo`, `*.xclbin`, `_x/` 전체 삭제 후 rebuild
- **교훈**: RTL 변경 시 `vten build --clean` 또는 수동으로 전체 캐시 삭제 필요

### 5.4 hw_emu poll timeout 부족

- **증상**: POLL_REG가 기본 timeout (10분) 안에 완료되지 않음
- **원인**: hw_emu는 1분 wall time ≈ 0.04ms sim time. NPU pipeline 완료에 ~1ms sim time 필요 → ~25분 wall time
- **해결**: hw_emu 기본 `poll_timeout_ms`를 600,000 → 3,600,000 (1시간)으로 변경
- **파일**: `vten/backend/xrt.py`

### 5.5 Deferred STORE가 두 번째 POLL 전에 flush됨 — OFM 전부 0

- **증상**: hw_emu E2E 테스트에서 OFM 데이터가 전부 0. POLL은 둘 다 성공, 데이터 흐름도 정상.
- **원인**: `CommandInterpreter._exec_poll_reg()`이 모든 POLL 성공 후 deferred STORE를 flush.
  Two-phase handshake(POLL#1: layer_done=0, POLL#2: layer_done=1)에서 POLL#1이 즉시 매치되면
  STORE가 POLL#1 직후 실행 → DUT가 아직 처리 중이므로 `bo.sync(FROM_DEVICE)`로 읽은 데이터가 전부 0.
- **해결**: `_exec_poll_reg()`에서 deferred STORE flush 제거. `execute()` 마지막의 safety net에서만 flush.
  이렇게 하면 모든 명령(POLL#2 포함) 완료 후에만 STORE가 실행됨.
- **파일**: `vten/runtime/interpreter.py`

---

## 6. 현재 상태 (2026-04-05)

### 검증된 항목
- XRT 빌드 파이프라인 (xo → xclbin): **완료**
- 6-kernel connectivity (32 HBM + 2 DDR + 76 stream): **완료**
- AP_CTRL_NONE / user_managed 지원: **완료**
- 메모리 뱅크 매핑 (group_id fallback): **완료**
- 레지스터 설정값 (host.cpp 대비): **검증 완료**
- BO 할당/초기화: **완료**
- recv_tensor 순서 (PULL after POLL): **완료**
- emulator crash: **해결**
- 파이프라인 데이터 흐름: **전체 동작 확인** (Vitis-EM 통계로 확인)
- Deferred STORE timing: **해결** — POLL#2 후 flush
- **E2E 검증 (TestUnetL0): PASS** — OFM 데이터 일치 확인

### 미해결 이슈

(현재 없음 — `layer_done` 이슈도 해결됨. POLL#2에서 13 polls/21s 후 매치 확인)
