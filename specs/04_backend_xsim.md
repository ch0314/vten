# vTen xsim Backend Infrastructure

**Version 0.5.0 — March 2026**
**참조 모델: `00_data_models.md` (SHM 상수, Command, BFMConfig)**
**소스: 메인 스펙 §11.1-11.15**
**관련: `05_bfm_library.md` (BFM 구현), `08_backend_abstraction.md` (멀티 백엔드)**

> **Note (v0.5.0):** 이 스펙의 SHM 핸드셰이크 프로토콜은 `SimBackend` 공통 클래스로
> 추출되었다 (08_backend_abstraction.md §7.2). XsimBackend와 VerilatorBackend가
> `SimBackend`를 상속한다. 이 스펙에서 `submit_batch()`로 기술된 메서드는
> `Backend.execute(compiled: CompiledResult) -> BackendResult`로 통합되었다.
> 상세: `08_backend_abstraction.md` §5.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [SHM Memory Layout](#2-shm-memory-layout)
3. [Synchronization: Semaphore Handshake](#3-synchronization-semaphore-handshake)
4. [Backend State Machine](#4-backend-state-machine)
5. [xsim GUI Restart & Deadlock Prevention](#5-xsim-gui-restart--deadlock-prevention)
6. [DPI-C Bridge Implementation](#6-dpi-c-bridge-implementation)
7. [Generated Testbench](#7-generated-testbench)
8. [Vivado/Vitis Compatibility](#8-vivadovitis-compatibility)
9. [Scheduler ↔ BFM Interface](#9-scheduler--bfm-interface)
10. [Command Scheduler](#10-command-scheduler)

---

## 1. Architecture

```
Python Host                              xsim Process
┌───────────────────┐                    ┌──────────────────────────┐
│  Runtime           │                    │  Generated TB (SV)       │
│  ┌───────────────┐ │   POSIX SHM       │  ┌──────────────────┐   │
│  │ Exec IR       │─┼──► Control ──────►│  │ SHM Controller   │   │
│  │ Compiler      │ │    Region          │  │ (state machine)  │   │
│  └───────────────┘ │                    │  └───┬──────────────┘   │
│                     │   ┌────────────┐  │      │                  │
│  ┌───────────────┐ │   │ Command    │  │  ┌───▼──────────────┐   │
│  │ SHM Manager   │─┼──►│ Region     │──┼─►│ Command          │   │
│  │ (Python)      │ │   │ (64B slots)│  │  │ Scheduler        │   │
│  └───────────────┘ │   ├────────────┤  │  └───┬──────────────┘   │
│                     │   │ Stats      │  │      │                  │
│                     │   │ Region     │◄─┼──────┤ (write stats)   │
│                     │   ├────────────┤  │  ┌───▼──────────────┐   │
│                     │   │ Buffer     │  │  │ BFM_AXI4S        │   │
│                     │   │ Descriptors│  │  │ BFM_AXI4         │   │
│                     │   ├────────────┤  │  │ BFM_AXI4L        │   │
│                     │   │ Data       │  │  └───┬──────────────┘   │
│                     │   │ Region     │◄─┼─►    │                  │
│                     │   └────────────┘  │  ┌───▼──────────────┐   │
│                     │                    │  │    DUT            │   │
│  ┌───────────────┐ │   Named Semaphores │  └──────────────────┘   │
│  │ sem_post /    │─┼──► /vten_h2b ────►│  sem_wait (DPI-C)       │
│  │ sem_wait      │◄┼─── /vten_b2h ◄───│  sem_post (DPI-C)       │
│  └───────────────┘ │                    │                          │
└───────────────────┘                    └──────────────────────────┘
```

---

## 2. SHM Memory Layout

SHM 레이아웃은 `00_data_models.md` §10에서 상수 및 바이너리 포맷을 정의한다. 여기서는 설계 근거를 보충한다.

**설계 근거:**
- 모든 메타데이터 영역의 고정 오프셋 → 양쪽 프로세스에서 O(1) 접근
- Control Region은 항상 오프셋 0 → 부트스트래핑 간접 참조 불필요
- Data Region은 버퍼별 64바이트 캐시 라인 정렬 → false sharing 회피 및 DPI-C memcpy 처리량 최적화
- 모든 멀티바이트 필드는 little-endian (x86 호스트 및 SV 시뮬레이터의 네이티브)
- Data Region의 텐서 데이터는 인터페이스별 PackingScheme의 `byte_order` 설정을 따름

**일반적인 크기:**

| 워크로드 | 커맨드 | 버퍼 | 데이터 | 총 SHM |
|----------|--------|------|--------|--------|
| Passthrough (E2E 최소) | ~5 | 2 | ~64KB | ~65KB |
| Conv3D 단위 테스트 | ~15 | 4 (+1 golden) | ~500KB | ~510KB |
| NPU top (합성) | ~40 | 8 (+4 golden) | ~4MB | ~4.2MB |
| Full 3D U-Net layer | ~100 | 12 | ~32MB | ~33MB |
| 5-layer pipeline (single Batch) | ~32 | 6 | ~5MB | ~5.3MB |

### 2.1 Multi-Batch Data Region Lifecycle

`ctx.run()`을 여러 번 호출하는 Multi-Batch 실행 시, SHM 영역별 수명 정책:

| 영역 | `ctx.run()` 사이 | 설명 |
|------|------------------|------|
| **Control Region** | 덮어쓰기 | 새 Batch의 `num_commands`, `num_buffers` 등으로 갱신 |
| **Command Region** | 덮어쓰기 | 새 Batch의 Command 슬롯으로 교체 |
| **Stats Region** | stale (BatchResult에 보존) | `ctx.run()` 반환 시 BatchResult에 캡처됨 |
| **Buffer Descriptor Table** | 보존 + 추가 | 기존 descriptor 유지, 새 버퍼만 추가 |
| **Data Region** | **보존** | 이전 Batch의 데이터 유지. Cross-Batch alias의 핵심. |

```
TestScenario.run() 시작
  │
  ├─ Batch 0: 버퍼 할당, LOAD, ctx.run(), 완료 대기
  │   Data Region: [buf0: ifm][buf1: weight][buf2: ofm_by_DUT]
  │
  ├─ ctx.run() 반환
  │   Data Region: 그대로 보존
  │   Command/Stats Region: stale
  │
  ├─ Batch 1: buf2를 alias로 재사용, buf3(새 weight) 할당, ctx.run()
  │   Data Region: [buf0: stale][buf1: stale][buf2: 재사용][buf3: new weight][buf4: new ofm]
  │
  └─ TestScenario.run() 종료 → SHM unmap 및 cleanup
```

**Buffer Validity Tracking (Python Runtime 측):**

```python
class SHMManager:
    def __init__(self):
        self._valid_buffers: set[int] = set()

    def mark_valid(self, buffer_id: int):
        """LOAD 완료 또는 Backend PULL 완료 후 호출."""
        self._valid_buffers.add(buffer_id)

    def is_valid(self, buffer_id: int) -> bool:
        return buffer_id in self._valid_buffers

    def invalidate(self, buffer_id: int):
        """Host가 버퍼 데이터를 수정한 경우 (transform 후)."""
        self._valid_buffers.discard(buffer_id)
```

Data Region이 단조 증가하여 `ctx.release_buffer()` 없이는 이전 버퍼를 회수하지 않는다. `/dev/shm` (tmpfs)은 보통 RAM의 50%이므로 대부분 시나리오에서 충분. `TestScenario.run()` 종료 시 전체 SHM segment가 unmap된다.

**Backend (SV/C) 측 변경 없음:** Backend는 각 Batch를 독립적으로 처리한다. `S_WAIT_HOST → S_LOAD_BATCH → ... → S_COMPLETE → S_WAIT_HOST` 루프가 Batch마다 반복될 뿐이다. Data Region 보존은 Host 측 정책이며 Backend는 관여하지 않는다.

---

## 3. Synchronization: Semaphore Handshake

POSIX named semaphore 사용. Spin-wait 대비 CPU 효율적이며 병렬 테스트 인스턴스에서 코어 경쟁 방지.

**세마포어 쌍:**

```
/vten_{session_id}_h2b    Host → Backend 시그널 (CMD_READY 후 post)
/vten_{session_id}_b2h    Backend → Host 시그널 (DONE/ERROR 후 post)
```

**Handshake Protocol:**

```
Host (Python)                            Backend (DPI-C in xsim)
─────────────                            ─────────────────────────

[1] shm_open + mmap
    sem_open (양쪽 세마포어 생성)
    magic, version, region offsets 기록
    host_status = IDLE
    xsim 프로세스 기동
                                         [2] DPI-C: vten_shm_init()
                                             sem_open (양쪽 세마포어 열기)
                                             stale 세마포어 카운트 drain
                                             magic / version 검증
                                             backend_status = IDLE
                                             sem_post(b2h)  ← "ready"
[3] sem_wait(b2h)  ← ready 대기
    assert backend_status == IDLE

── Kernel Task 루프 ──

[4] Data Region 적재 (텐서 바이트)
    Command 슬롯 기록
    Buffer Descriptors 기록
    num_commands, num_buffers 설정
    host_status = CMD_READY
    sem_post(h2b)
                      ──────────────►
                                         [5] sem_timedwait(h2b, timeout)
                                             host_status 읽기:
                                               CMD_READY → 진행
                                               SHUTDOWN  → 정리 + 종료
                                             backend_status = RUNNING

                                         [6] Scheduler로 커맨드 실행
                                             BFM을 통해 Data Region 읽기/쓰기
                                             Stats Region 기록 (활성화 시)

                                         [7] backend_status = DONE (또는 ERROR)
                                             sem_post(b2h)
                      ◄──────────────
[8] sem_timedwait(b2h, timeout)
    backend_status 읽기:
      DONE  → 결과 + 통계 읽기
      ERROR → 에러 정보 읽기, 예외 발생
    host_status = ACK

── [4]로 루프 또는 SHUTDOWN ──

[9] host_status = SHUTDOWN
    sem_post(h2b)
                      ──────────────►
                                         [10] SHUTDOWN 감지
                                              정리 + $finish
[11] sem_unlink 양쪽 세마포어
     shm_unlink
```

**Named 세마포어 사용 이유:** POSIX SHM이 다른 툴체인으로 컴파일된 프로세스 간 `sem_t` 레이아웃 호환성을 보장하지 않음 (Host=gcc, xsim=internal). Named 세마포어는 커널 관리 핸들 사용.

---

## 4. Backend State Machine

```
              ┌──────────────────────────────────┐
              │                                  │
              ▼                                  │
        ┌──────────┐    init fail           ┌────┴─────┐
reset ──►│  S_INIT  │─────────────────────►│  S_ERROR │
        └────┬─────┘                       └────┬─────┘
             │ init_done                        │
             ▼                                  │ signal_error
        ┌──────────────┐◄───────────────────────┘
   ┌───►│ S_WAIT_HOST  │◄──────────────┐
   │    └──┬───────┬───┘               │
   │       │       │                   │
   │  SHUTDOWN  CMD_READY              │
   │       │       │                   │
   │       ▼       ▼                   │
   │  ┌────────┐ ┌────────────┐        │
   │  │S_SHUT  │ │S_LOAD_BATCH│        │
   │  │DOWN    │ └─────┬──────┘        │
   │  └───┬────┘       │              │
   │      │            ▼              │
   │   $finish   ┌──────────┐         │
   │             │ S_FEED   │         │
   │             └─────┬────┘         │
   │               feed_done          │
   │                   │              │
   │                   ▼              │
   │             ┌───────────┐        │
   │             │ S_EXECUTE ├─error─►│
   │             └─────┬─────┘        │
   │                   │ all_committed│
   │                   ▼              │
   │             ┌───────────┐        │
   │             │ S_DRAIN   │        │
   │             └─────┬─────┘        │
   │                   │ all_drained  │
   │                   ▼              │
   │             ┌───────────┐        │
   └─────────────┤S_COMPLETE ├────────┘
                 └───────────┘
```

| 상태 | 활동 |
|------|------|
| `S_INIT` | DPI-C: `vten_shm_init()`. Magic/version 검증. Stale 세마포어 drain. `b2h` post (ready). |
| `S_WAIT_HOST` | DPI-C: `vten_wait_host_signal_safe(timeout_ms)`. 타임아웃 시: 재시도. CMD_READY: 전이. SHUTDOWN: 전이. |
| `S_LOAD_BATCH` | SHM에서 모든 Command 슬롯을 로컬 캐시로 일괄 읽기 (DPI-C memcpy). `backend_status = RUNNING` 설정. Multi-Batch 시 이전 Batch의 Data Region은 보존된 상태. |
| `S_FEED` | 로컬 캐시의 커맨드를 Scheduler에 `feed_valid`/`feed_ready` 핸드셰이크로 순차 전달. 완료 시 `feed_done` 펄스. |
| `S_EXECUTE` | BFM 완료 시그널 모니터링. 커맨드별 Issue/Commit 추적. Stats Region 기록. 에러 시 S_ERROR. 모든 커맨드 Committed 시 전이. |
| `S_DRAIN` | In-flight BFM 응답 대기 (예: 최종 AXI 쓰기 응답, B-channel 큐 drain). 모든 BFM idle 시 전이. |
| `S_COMPLETE` | `backend_status = DONE`. DPI-C: `sem_post(b2h)`. `S_WAIT_HOST`로 전이. |
| `S_ERROR` | `backend_status = ERROR`. error_code/cmd_id/message 기록. DPI-C: `sem_post(b2h)`. `S_WAIT_HOST`로 전이. |
| `S_SHUTDOWN` | DPI-C: `vten_cleanup()`. `$finish` 호출. |

### SHM Controller SV 구현

> **v0.4.1 변경:** DPI-C 호출이 `always_comb`에 있던 문제를 수정.
> 단일 `always_ff` 기반으로 재작성하여 posedge당 DPI-C 호출 1회 보장.
> `S_DISPATCH` → `S_LOAD_BATCH` + `S_FEED` 분리.
> Controller→Scheduler 인터페이스를 feed handshake로 명확화.

```systemverilog
module vten_shm_controller #(
    parameter string SESSION_ID = "default",
    parameter MAX_CMDS = 256
)(
    input  logic clk,
    input  logic rst_n,
    // → Scheduler: 커맨드 피드 (handshake)
    output logic        feed_valid,
    output bfm_cmd_t    feed_data,
    input  logic        feed_ready,
    output logic        feed_done,     // S_FEED 완료 → Scheduler 실행 시작
    // ← Scheduler: 상태 보고
    input  logic        sched_all_committed,
    input  logic        sched_all_drained,
    // ← Scheduler: 에러 보고
    input  logic        sched_error,
    input  logic [15:0] sched_error_cmd_id,
    input  logic [15:0] sched_error_code
);

    import "DPI-C" function int  vten_shm_init(input string session_id);
    import "DPI-C" function int  vten_wait_host_signal_safe(input int timeout_ms);
    import "DPI-C" function int  vten_read_host_status();
    import "DPI-C" function void vten_set_backend_status(input int status);
    import "DPI-C" function void vten_signal_complete();
    import "DPI-C" function void vten_signal_error(input int code, input string msg);
    import "DPI-C" function void vten_cleanup();
    import "DPI-C" function int  vten_read_num_commands();
    import "DPI-C" function int  vten_read_timeout_ms();
    import "DPI-C" function int  vten_read_command(input int cmd_id,
                                                    output bfm_cmd_t cmd);

    typedef enum logic [3:0] {
        S_INIT, S_WAIT_HOST, S_LOAD_BATCH,
        S_FEED, S_EXECUTE, S_DRAIN,
        S_COMPLETE, S_ERROR, S_SHUTDOWN
    } state_t;

    state_t state;
    int num_commands;
    int feed_idx;
    int timeout_ms;

    // 커맨드 로컬 캐시 (SHM에서 일괄 읽기 후 저장)
    bfm_cmd_t cmd_cache [0:MAX_CMDS-1];

    // ── 단일 always_ff: 상태 전이 + 데이터패스 + DPI-C ──
    // 모든 DPI-C 호출이 posedge당 정확히 1회만 실행됨을 보장.

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_INIT;
            feed_valid <= 0;
            feed_done  <= 0;
            feed_idx   <= 0;
        end else begin
            // 기본값: 매 사이클 deassert
            feed_valid <= 0;
            feed_done  <= 0;

            case (state)
                // ── 초기화: SHM 연결 ──
                S_INIT: begin
                    if (vten_shm_init(SESSION_ID) == 0)
                        state <= S_WAIT_HOST;
                    else
                        state <= S_ERROR;
                end

                // ── 호스트 시그널 대기 (timed wait으로 GUI 반응성 확보) ──
                S_WAIT_HOST: begin
                    int result;
                    result = vten_wait_host_signal_safe(
                        timeout_ms > 0 ? timeout_ms : 10
                    );
                    if (result == 0) begin  // VTEN_OK
                        case (vten_read_host_status())
                            1: state <= S_LOAD_BATCH;   // CMD_READY
                            3: state <= S_SHUTDOWN;      // SHUTDOWN
                            default: ;                   // 재시도
                        endcase
                    end
                    // TIMEOUT → stay in S_WAIT_HOST (GUI 이벤트 처리 허용)
                end

                // ── 배치 일괄 로드: SHM → 로컬 캐시 ──
                S_LOAD_BATCH: begin
                    num_commands <= vten_read_num_commands();
                    timeout_ms   <= vten_read_timeout_ms();
                    vten_set_backend_status(1);  // RUNNING
                    // 모든 커맨드를 한 사이클에 로컬 캐시로 복사
                    for (int i = 0; i < vten_read_num_commands(); i++)
                        vten_read_command(i, cmd_cache[i]);
                    feed_idx <= 0;
                    state    <= S_FEED;
                end

                // ── Scheduler에 커맨드 순차 전달 ──
                S_FEED: begin
                    if (feed_idx >= num_commands) begin
                        feed_done <= 1;    // Scheduler에 "배치 완료" 통보
                        state     <= S_EXECUTE;
                    end else if (feed_ready) begin
                        feed_valid <= 1;
                        feed_data  <= cmd_cache[feed_idx];
                        feed_idx   <= feed_idx + 1;
                    end
                end

                // ── Scheduler 실행 모니터링 ──
                S_EXECUTE: begin
                    if (sched_error)
                        state <= S_ERROR;
                    else if (sched_all_committed)
                        state <= S_DRAIN;
                end

                // ── BFM in-flight 응답 drain ──
                S_DRAIN: begin
                    if (sched_all_drained)
                        state <= S_COMPLETE;
                end

                // ── 완료 → 호스트에 통보 ──
                S_COMPLETE: begin
                    vten_signal_complete();  // backend_status=DONE + sem_post
                    state <= S_WAIT_HOST;
                end

                // ── 에러 → 호스트에 통보 ──
                S_ERROR: begin
                    vten_signal_error(
                        sched_error_code,
                        $sformatf("[Scheduler] error at cmd_id=%0d",
                                  sched_error_cmd_id)
                    );
                    state <= S_WAIT_HOST;
                end

                // ── 종료 ──
                S_SHUTDOWN: begin
                    vten_cleanup();
                    $finish;
                end
            endcase
        end
    end
endmodule
```

**설계 근거:**

- **단일 `always_ff` 패턴:** `always_comb`에서 DPI-C 함수(`vten_shm_init`, `vten_wait_host_signal_safe` 등)를 호출하면 입력 변경 시 delta-cycle마다 재평가되어 OS-level 함수가 반복 호출되는 위험이 있다. 단일 `always_ff`에서 상태 전이와 DPI-C 호출을 모두 수행하면 posedge당 정확히 1회 실행이 보장된다.
- **S_LOAD_BATCH 분리:** SHM에서 커맨드를 읽는 DPI-C 호출(memcpy)과 Scheduler에 feed하는 순수 SV 로직을 분리. S_LOAD_BATCH는 한 사이클에 모든 DPI-C 호출을 수행하고, S_FEED는 DPI-C 없이 handshake만 수행한다.
- **feed_done 펄스:** Scheduler가 배치 크기를 확정하고 전처리(sync chain, barrier fence, committed bitmap 초기화)를 시작하는 트리거.

---

## 5. xsim GUI Restart & Deadlock Prevention

### 5.1 Deadlock Scenarios

| 시나리오 | 원인 | 미대응 시 |
|----------|------|-----------|
| GUI `stop` during `sem_wait` | DPI-C `sem_wait`은 OS 레벨 블록; xsim이 중단 불가 | GUI 무한 정지 |
| GUI `restart` | SV 상태 리셋되나 SHM/세마포어 카운트 유지 | 오래된 데이터, 카운트 불일치 |
| Host crash | `sem_post(h2b)` 도착 안 함 | Backend 영구 블록 |

### 5.2 Timed Wait

모든 `sem_wait`이 DPI-C에서 `sem_timedwait` 사용 (기본 10ms). 타임아웃 시 SV에 제어 반환하여 xsim이 GUI 이벤트 처리 가능. SV 상태 머신은 다음 클럭에서 재시도.

```c
int vten_wait_host_signal_safe(int timeout_ms) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }

    int ret = sem_timedwait(sem_h2b, &ts);
    if (ret == -1) {
        if (errno == ETIMEDOUT) return VTEN_TIMEOUT;
        return VTEN_ERROR;
    }
    return VTEN_OK;
}
```

**트레이드오프:** 10ms 타임아웃은 배치 제출에 최대 10ms 지연 — 수천 사이클의 커널 실행에 비해 무시 가능. 비GUI 배치 모드에서는 `timeout_ms = 0`으로 설정하여 긴 기본값(10초) 사용.

### 5.3 Restart Recovery

xsim 재시작 시 SV가 `S_INIT` 재진입. 초기화 시퀀스가 오래된 상태 처리:

```c
int vten_shm_init(const char* session_id) {
    char shm_name[64];
    snprintf(shm_name, 64, "/vten_%s", session_id);
    int fd = shm_open(shm_name, O_RDWR, 0);
    shm_base = mmap(NULL, total_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

    ControlHeader* ctrl = (ControlHeader*)shm_base;
    if (ctrl->magic != 0x5654454E) return VTEN_ERR_MAGIC;
    if (ctrl->version != PROTOCOL_VERSION) return VTEN_ERR_VERSION;

    // Stale 세마포어 카운트 drain (재시작 시 핵심)
    while (sem_trywait(sem_h2b) == 0) { /* drain */ }
    while (sem_trywait(sem_b2h) == 0) { /* drain */ }

    ctrl->session_seq++;          // Host가 재시작 감지
    ctrl->backend_status = 0;     // IDLE
    sem_post(sem_b2h);            // Backend ready 시그널
    return VTEN_OK;
}
```

### 5.4 Global Timeout Policy

| 컨텍스트 | `sem_timedwait` 타임아웃 | 만료 시 동작 |
|----------|--------------------------|-------------|
| Backend `S_WAIT_HOST` (GUI 모드) | 10ms | 재시도 (GUI 이벤트 허용) |
| Backend `S_WAIT_HOST` (배치 모드) | 10s | 경고 로그와 함께 재시도 |
| Host `execute()` (내부 submit) | 300s (설정 가능) | `TimeoutError` 발생 |
| Host `wait_backend_ready` | 30s | `TimeoutError` 발생 |

---

## 6. DPI-C Bridge Implementation

### 6.1 Function Signatures

```c
// ── Lifecycle ──
int  vten_shm_init(const char* session_id);
void vten_cleanup();

// ── Synchronization ──
int  vten_wait_host_signal_safe(int timeout_ms);
int  vten_read_host_status();
void vten_set_backend_status(int status);
void vten_signal_complete();
void vten_signal_error(int code, const char* msg);

// ── Control Region ──
int  vten_read_num_commands();
int  vten_read_num_buffers();
int  vten_read_timeout_ms();
int  vten_read_flags();

// ── Command Region ──
int  vten_read_command(int cmd_id, /* output: bfm_cmd_t 필드 (dependency 제외) */);
void vten_read_command_deps(int cmd_id,    // (v0.4.2)
                            int* num_dep,
                            int* dep_ids,      // int[4] (fixed-size, NOT svOpenArrayHandle)
                            int* num_cdep,
                            int* cdep_ids);    // int[4]

// ── Data Region — Bulk transfer (byte[] + memcpy) ──
// SV 측 byte arr[] → C 측 svGetArrayPtr() → memcpy. beat당 1회 호출.
void vten_read_data_bulk(int buf_id, int offset, int size, void* dst_handle);
void vten_write_data_bulk(int buf_id, int offset, int size, const void* src_handle);
void vten_read_golden_bulk(int buf_id, int offset, int size, void* dst_handle);
// Scalar byte write — AXI4 BFM partial WSTRB slow path 전용
void vten_write_data_byte(int buf_id, int offset, int value);

// ── Stats Region ──
void vten_write_cmd_stats(int cmd_id, int status,
                          int issue_cycle, int commit_cycle,
                          int first_active, int last_active,
                          int active_cycles, int total_beats, int stall_cycles);
void vten_write_cmd_status(int cmd_id, int status);

// ── Probe ──
void vten_log_mismatch(int cycle, int beat,
                       int expected_hi, int expected_lo,
                       int actual_hi, int actual_lo);
```

**DPI-C 함수 호출자 매핑:**

| 함수 | C 구현 위치 | SV 호출자 | 용도 |
|------|-------------|-----------|------|
| `vten_shm_init` | shm_bridge.c | Controller (S_INIT) | SHM 매핑, 세마포어 연결, magic/version 검증 |
| `vten_cleanup` | shm_bridge.c | Controller (S_SHUTDOWN) | SHM/세마포어 해제 |
| `vten_wait_host_signal_safe` | shm_bridge.c | Controller (S_WAIT_HOST) | `sem_timedwait(h2b)` + stale drain |
| `vten_read_host_status` | shm_bridge.c | Controller | Control Header의 host_status 읽기 |
| `vten_set_backend_status` | shm_bridge.c | Controller | Control Header의 backend_status 쓰기 |
| `vten_signal_complete` | shm_bridge.c | Controller (S_COMPLETE) | `backend_status=DONE` + `sem_post(b2h)` |
| `vten_signal_error` | shm_bridge.c | Controller (S_ERROR) | error_code/cmd_id/message 기록 + `sem_post(b2h)` |
| `vten_read_num_commands` | shm_bridge.c | Controller (S_LOAD_BATCH) | Control Header에서 커맨드 수 읽기 |
| `vten_read_num_buffers` | shm_bridge.c | Controller | 버퍼 수 읽기 |
| `vten_read_timeout_ms` | shm_bridge.c | Controller | GUI 타임아웃 설정 읽기 |
| `vten_read_flags` | shm_bridge.c | Controller | flags (STATS_ENABLED 등) 읽기 |
| `vten_read_command` | shm_bridge.c | Scheduler (preprocess) | 64B 슬롯 → bfm_cmd_t 디코딩 |
| `vten_read_command_deps` | shm_bridge.c | Scheduler (preprocess) | dependency 배열만 별도 추출 |
| `vten_read_data_bulk` | shm_bridge.c | BFM (AXI4S MASTER, AXI4 Read) | Data Region에서 byte[] 벌크 읽기 (memcpy) |
| `vten_write_data_bulk` | shm_bridge.c | BFM (AXI4S SLAVE, AXI4 Write) | Data Region에 byte[] 벌크 쓰기 (memcpy) |
| `vten_read_golden_bulk` | shm_bridge.c | BFM Probe | Probe golden 버퍼에서 byte[] 벌크 읽기 |
| `vten_write_data_byte` | shm_bridge.c | BFM (AXI4 Write, partial WSTRB) | Data Region에 단일 바이트 쓰기 |
| `vten_write_cmd_stats` | shm_bridge.c | BFM (finish_command) | Stats Region에 완전한 통계 기록 |
| `vten_write_cmd_status` | shm_bridge.c | Scheduler (dispatch) | Stats Region의 status 필드만 업데이트 |
| `vten_log_mismatch` | shm_bridge.c | BFM Probe | Probe 불일치 로깅 (stderr) |

**SV Import 선언 (tb_top.sv 또는 별도 include에 배치):**

```systemverilog
// vten_dpi_imports.svh — DPI-C 함수 SV 측 선언
// tb_top.sv에서 include. 모든 BFM/Controller/Scheduler가 참조.

import "DPI-C" function int  vten_shm_init(input string session_id);
import "DPI-C" function void vten_cleanup();

import "DPI-C" function int  vten_wait_host_signal_safe(input int timeout_ms);
import "DPI-C" function int  vten_read_host_status();
import "DPI-C" function void vten_set_backend_status(input int status);
import "DPI-C" function void vten_signal_complete();
import "DPI-C" function void vten_signal_error(input int code, input string msg);

import "DPI-C" function int  vten_read_num_commands();
import "DPI-C" function int  vten_read_num_buffers();
import "DPI-C" function int  vten_read_timeout_ms();
import "DPI-C" function int  vten_read_flags();

import "DPI-C" function int  vten_read_command(
    input int cmd_id,
    output int opcode, output int interface_id,
    output int protocol, output int role,
    output int buffer_id, output int probe, output int flags,
    output int size, output longint phys_addr,
    output int reg_offset, output int reg_value,
    output int reg_mask, output int reg_expected,
    output int golden_buf_id,
    output int num_deps, output int num_commit_deps,
    output int dep_ids [0:3], output int commit_dep_ids [0:3]);

import "DPI-C" function void vten_read_command_deps(
    input int cmd_id,
    output int num_dep, output int dep_ids [0:3],
    output int num_cdep, output int cdep_ids [0:3]);

// ── Data Region — Bulk transfer (byte[] + memcpy) ──
import "DPI-C" function void vten_read_data_bulk(
    input int buf_id, input int offset, input int size,
    inout byte dst []);

import "DPI-C" function void vten_write_data_bulk(
    input int buf_id, input int offset, input int size,
    inout byte src []);

import "DPI-C" function void vten_read_golden_bulk(
    input int buf_id, input int offset, input int size,
    inout byte dst []);

// ── Data Region — Scalar byte write (AXI4 partial WSTRB) ──
import "DPI-C" function void vten_write_data_byte(
    input int buf_id, input int offset, input int value);

import "DPI-C" function void vten_write_cmd_stats(
    input int cmd_id, input int status,
    input int issue_cycle, input int commit_cycle,
    input int first_active, input int last_active,
    input int active_cycles, input int total_beats, input int stall_cycles);

import "DPI-C" function void vten_write_cmd_status(
    input int cmd_id, input int status);

import "DPI-C" function void vten_log_mismatch(
    input int cycle, input int beat,
    input int expected_hi, input int expected_lo,
    input int actual_hi, input int actual_lo);
```

> **Bulk Transfer 설계:** SV 측 `byte arr[]` (signed 8-bit 동적 배열)은
> Verilator와 xsim 모두에서 `svGetArrayPtr()`로 연속 메모리 포인터를 반환한다.
> C 측에서 `svGetArrayPtr()` + `memcpy()`로 beat당 1회 전송한다.
> `bit [7:0] arr[]`는 시뮬레이터마다 stride가 다를 수 있으므로 `byte`를 사용한다.
> BFM은 모듈 레벨에서 `byte beat_buf [0:BYTES_PER_BEAT-1]`를 선언하여 재사용한다.
>
> **Fixed-size array 주의:** `vten_read_command_deps`의 `output int dep_ids [0:3]`는
> fixed-size array이므로 C 측에서 `int*`로 직접 접근한다 (`svOpenArrayHandle` 아님).
> Verilator는 fixed-size array를 raw pointer로 전달한다.

### 6.2 Internal Pointer Management

```c
static void*          shm_base = NULL;
static ControlHeader* ctrl     = NULL;
static void*          cmd_base = NULL;
static void*          stats_base = NULL;
static void*          bufdesc_base = NULL;
static void*          data_base = NULL;
static sem_t*         sem_h2b = NULL;
static sem_t*         sem_b2h = NULL;

static BufferDescriptor buf_cache[MAX_BUFFERS];
static int buf_cache_valid = 0;

// Bulk read: SV byte[] → svGetArrayPtr → memcpy
void vten_read_data_bulk(int buf_id, int offset, int size, void* dst_handle) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();
    BufferDescriptor* desc = &buf_cache[buf_id];
    uint8_t* src = (uint8_t*)data_base + desc->data_offset + offset;
    uint8_t* dst = (uint8_t*)svGetArrayPtr((svOpenArrayHandle)dst_handle);
    if (dst != NULL) {
        memcpy(dst, src, (size_t)size);
    }
}

// Scalar byte write: AXI4 partial WSTRB path
void vten_write_data_byte(int buf_id, int offset, int value) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();
    BufferDescriptor* desc = &buf_cache[buf_id];
    uint8_t* dst = (uint8_t*)data_base + desc->data_offset + offset;
    *dst = (uint8_t)value;
}
```

### 6.3 vten_read_command 언패킹

64바이트 슬롯에서 `bfm_cmd_t` 구조체로 디코딩:

```c
int vten_read_command(int cmd_id,
    /* SV output args: */
    int* opcode, int* interface_id, int* protocol, int* role,
    int* buffer_id, int* probe, int* flags, int* size,
    long long* phys_addr,
    int* reg_offset, int* reg_value, int* reg_mask, int* reg_expected,
    int* golden_buf_id,
    int* num_deps, int* num_commit_deps,
    int dep_ids[4], int commit_dep_ids[4])
{
    uint8_t* slot = (uint8_t*)cmd_base + cmd_id * CMD_SLOT_SIZE;

    *opcode       = *(uint16_t*)(slot + 0x00);
    // cmd_id at 0x02 is implicit (= cmd_id arg)
    *interface_id = *(uint16_t*)(slot + 0x04);
    *protocol     = *(uint8_t*)(slot + 0x06);
    *role         = *(uint8_t*)(slot + 0x07);
    *buffer_id    = *(uint16_t*)(slot + 0x08);
    *probe        = *(uint8_t*)(slot + 0x0A);
    *flags        = *(uint8_t*)(slot + 0x0B);
    *size         = *(uint32_t*)(slot + 0x0C);
    *phys_addr    = *(uint64_t*)(slot + 0x10);
    *reg_offset   = *(uint32_t*)(slot + 0x18);
    *reg_value    = *(uint32_t*)(slot + 0x1C);
    *reg_mask     = *(uint32_t*)(slot + 0x20);
    *reg_expected = *(uint32_t*)(slot + 0x24);
    *golden_buf_id= *(uint16_t*)(slot + 0x28);
    *num_deps     = *(uint8_t*)(slot + 0x2A);
    *num_commit_deps = *(uint8_t*)(slot + 0x2B);

    uint16_t* dep_ptr = (uint16_t*)(slot + 0x2C);
    uint16_t* cdep_ptr = (uint16_t*)(slot + 0x34);
    for (int i = 0; i < 4; i++) {
        dep_ids[i] = dep_ptr[i];
        commit_dep_ids[i] = cdep_ptr[i];
    }
    return 0;
}
```

### 6.4 vten_read_command_deps 언패킹 (v0.4.2)

Dependency 필드만 별도 추출. Scheduler의 `preprocess_batch()`에서 호출.
`bfm_cmd_t`에는 dependency가 없으므로, Scheduler가 별도 배열에 저장한다.

```c
void vten_read_command_deps(int cmd_id,
                            int* num_dep,
                            svOpenArrayHandle dep_ids,
                            int* num_cdep,
                            svOpenArrayHandle cdep_ids) {
    uint8_t* slot = (uint8_t*)cmd_base + cmd_id * CMD_SLOT_SIZE;

    *num_dep  = *(uint8_t*)(slot + 0x2A);
    *num_cdep = *(uint8_t*)(slot + 0x2B);

    uint16_t* deps = (uint16_t*)(slot + 0x2C);
    uint16_t* cdeps = (uint16_t*)(slot + 0x34);

    uint16_t* dep_out  = (uint16_t*)svGetArrayPtr(dep_ids);
    uint16_t* cdep_out = (uint16_t*)svGetArrayPtr(cdep_ids);

    for (int i = 0; i < 4; i++) {
        dep_out[i]  = deps[i];
        cdep_out[i] = cdeps[i];
    }
}
```

---

## 7. Generated Testbench

빌드 단계에서 Jinja2 템플릿으로 완전한 SV 테스트벤치 생성:

- **tb_top.sv**: DUT 인스턴스화, 클럭/리셋 생성, BFM 연결, `vten_shm_controller` 인스턴스화, 시뮬레이션 제어
- **vten_shm_controller.sv**: 상태 머신 (§4)
- **vten_command_scheduler.sv**: IR 커맨드 읽기, 의존성 추적, BFM 디스패치
- **vten_bfm_axi4s.sv**: AXI4-Stream BFM
- **vten_bfm_axi4.sv**: AXI4 BFM (슬레이브)
- **vten_bfm_axilite.sv**: AXI4-Lite BFM
- **vten_types.svh**: 공유 타입 정의

템플릿 아키텍처 및 컨텍스트 스키마는 `06_codegen_and_cli.md` 참조.

---

## 8. Vivado/Vitis Compatibility

```bash
# 생성된 빌드 스크립트 (xsim 타겟)
xvlog --sv rtl/*.sv tb/generated/*.sv
xelab tb_top --sv_lib libvten_shm -timescale 1ns/1ps
xsim tb_top --runall              # 배치 모드
xsim tb_top --gui                 # GUI 모드 (timed-wait로 반응성 확보)
```

DPI-C 브릿지는 공유 라이브러리로 컴파일:

```bash
gcc -shared -fPIC -o libvten_shm.so \
    vten_shm_bridge.c \
    -lrt -lpthread                    # POSIX SHM + 세마포어
```

---

## 9. Scheduler ↔ BFM Interface

### 9.1 Type Definitions

```systemverilog
typedef enum logic [3:0] {
    OP_LOAD=4'd1, OP_PUSH=4'd2, OP_PULL=4'd3, OP_STORE=4'd4,
    OP_WRITE_REG=4'd5, OP_READ_REG=4'd6, OP_POLL_REG=4'd7,
    OP_BARRIER=4'd8, OP_COMPARE=4'd9
} opcode_t;

typedef enum logic [1:0] {
    PROTO_AXI4S=2'd1, PROTO_AXI4=2'd2, PROTO_AXI4L=2'd3
} protocol_t;

typedef struct packed {
    opcode_t        opcode;
    logic [15:0]    cmd_id;
    logic [15:0]    interface_id;
    protocol_t      protocol;
    logic           role;
    logic [15:0]    buffer_id;
    logic           probe;
    logic           sync;
    logic [31:0]    size;
    logic [63:0]    phys_addr;
    logic [31:0]    reg_offset;
    logic [31:0]    reg_value;
    logic [31:0]    reg_mask;
    logic [31:0]    reg_expected;
    logic [15:0]    golden_buf_id;
} bfm_cmd_t;
```

### 9.2 Interface Definition

```systemverilog
interface vten_bfm_cmd_if;
    logic        cmd_valid;
    bfm_cmd_t    cmd_data;
    logic        done_valid;
    logic [15:0] done_cmd_id;
    logic        done_error;
    logic [15:0] done_error_code;
    logic        idle;          // BFM → Scheduler: 모든 큐/pending 비어있음

    modport scheduler (
        output cmd_valid, cmd_data,
        input  done_valid, done_cmd_id, done_error, done_error_code, idle);
    modport bfm (
        input  cmd_valid, cmd_data,
        output done_valid, done_cmd_id, done_error, done_error_code, idle);
endinterface
```

**설계 결정:**
- **`cmd_ready` 없음:** BFM은 ideal slave — 항상 커맨드 수신. SV dynamic queue로 크기 제한 없음.
- **`done_valid`은 1사이클 펄스:** 동일 클럭에서 복수 BFM 완료 보고 가능. Scheduler가 모두 수집.
- **`idle` 신호 (v0.4.1 추가):** BFM의 모든 내부 큐와 pending 상태가 비어있음을 표시. Scheduler가 `all_drained` 산출에 사용. AXI4 BFM의 B-channel 큐가 비어있지 않으면 idle=0.
- **BFM 인스턴스는 인터페이스당 하나:** `kernel_spec.yaml`의 각 인터페이스 → BFM 1개. 같은 BFM에 복수 커맨드 가능.

### 9.3 vten_types.svh — Complete Content

`vten_types.svh`는 모든 SV 모듈이 `include하는 공유 타입 정의 파일이다.
Codegen이 생성하지 않으며, vten_sv/ 디렉토리에 고정 파일로 존재한다.

```systemverilog
`ifndef VTEN_TYPES_SVH
`define VTEN_TYPES_SVH

// ═══════════════════════════════════════════════════════════════
// OpCode (00_data_models.md §1.4)
// ═══════════════════════════════════════════════════════════════
typedef enum logic [3:0] {
    OP_LOAD      = 4'd1,
    OP_PUSH      = 4'd2,
    OP_PULL      = 4'd3,
    OP_STORE     = 4'd4,
    OP_WRITE_REG = 4'd5,
    OP_READ_REG  = 4'd6,
    OP_POLL_REG  = 4'd7,
    OP_BARRIER   = 4'd8,
    OP_COMPARE   = 4'd9
} opcode_t;

// ═══════════════════════════════════════════════════════════════
// Protocol (00_data_models.md §1.1)
// ═══════════════════════════════════════════════════════════════
typedef enum logic [1:0] {
    PROTO_AXI4S = 2'd1,
    PROTO_AXI4  = 2'd2,
    PROTO_AXI4L = 2'd3
} protocol_t;

// ═══════════════════════════════════════════════════════════════
// Role (00_data_models.md §1.2)
// ═══════════════════════════════════════════════════════════════
localparam logic ROLE_MASTER = 1'b0;
localparam logic ROLE_SLAVE  = 1'b1;

// ═══════════════════════════════════════════════════════════════
// Command Status (00_data_models.md §1.7, Stats Region)
// ═══════════════════════════════════════════════════════════════
typedef enum logic [2:0] {
    CMD_PENDING   = 3'd0,
    CMD_ISSUED    = 3'd1,
    CMD_ACTIVE    = 3'd2,
    CMD_COMMITTED = 3'd3,
    CMD_ERROR     = 3'd4
} cmd_status_t;

// ═══════════════════════════════════════════════════════════════
// Backend Status (00_data_models.md §10.2, Control Header)
// ═══════════════════════════════════════════════════════════════
localparam int BACKEND_IDLE    = 0;
localparam int BACKEND_RUNNING = 1;
localparam int BACKEND_DONE    = 2;
localparam int BACKEND_ERROR   = 3;

// Host Status
localparam int HOST_IDLE      = 0;
localparam int HOST_CMD_READY = 1;
localparam int HOST_ACK       = 2;
localparam int HOST_SHUTDOWN  = 3;

// ═══════════════════════════════════════════════════════════════
// Backend Error Codes (00_data_models.md §10.13)
// ═══════════════════════════════════════════════════════════════
localparam int ERR_OK              = 0;
localparam int ERR_ADDR_UNMATCH    = 1;
localparam int ERR_POLL_TIMEOUT    = 2;
localparam int ERR_BFM_QUEUE       = 3;
localparam int ERR_SCHEDULER       = 4;
localparam int ERR_SHM_ACCESS      = 5;
localparam int ERR_UNKNOWN_OPCODE  = 6;
localparam int ERR_BFM_MAP         = 7;
localparam int ERR_PROBE_MISMATCH  = 8;
localparam int ERR_TIMEOUT         = 9;

// ═══════════════════════════════════════════════════════════════
// SHM Constants (00_data_models.md §10.1)
// ═══════════════════════════════════════════════════════════════
localparam int SHM_MAGIC       = 32'h5654_454E;  // "VTEN"
localparam int SHM_VERSION     = 32'h0000_0003;  // v0.4.2
localparam int CONTROL_SIZE    = 256;
localparam int CMD_SLOT_SIZE   = 64;
localparam int STATS_SLOT_SIZE = 32;
localparam int BUF_DESC_SIZE   = 24;

// ═══════════════════════════════════════════════════════════════
// BFM Command Structure (§9.1)
// Scheduler → BFM 디스패치 전용. 의존성 필드 미포함.
// ═══════════════════════════════════════════════════════════════
typedef struct packed {
    opcode_t        opcode;          // 하위 4비트 유효
    logic [15:0]    cmd_id;
    logic [15:0]    interface_id;
    protocol_t      protocol;
    logic           role;            // 0=MASTER, 1=SLAVE
    logic [15:0]    buffer_id;
    logic           probe;
    logic           sync;            // flags[0]
    logic [31:0]    size;            // 바이트 단위 전송 크기
    logic [63:0]    phys_addr;       // 물리 주소 (AXI4 BFM 용)
    logic [31:0]    reg_offset;      // AXI-Lite 오프셋
    logic [31:0]    reg_value;       // 쓰기 값
    logic [31:0]    reg_mask;        // POLL_REG 마스크
    logic [31:0]    reg_expected;    // POLL_REG 기대값
    logic [15:0]    golden_buf_id;   // Probe golden 버퍼 ID
} bfm_cmd_t;

// ═══════════════════════════════════════════════════════════════
// NULL Dependency Sentinel
// ═══════════════════════════════════════════════════════════════
localparam logic [15:0] DEP_NONE = 16'hFFFF;

`endif // VTEN_TYPES_SVH
```

**사용법:**

```systemverilog
// 모든 vten SV 모듈 및 생성된 tb_top.sv에서:
`include "vten_types.svh"
```

**Codegen 책임 (06_codegen_and_cli.md 참조):**
- `vten_types.svh`는 고정 파일 — codegen이 수정하지 않음
- `vten_bfm_cmd_if.sv`도 고정 파일
- codegen이 생성하는 것: `tb_top.sv` (DUT/BFM 인스턴스화, `iface_to_bfm[]` 초기화)

---

## 10. Command Scheduler

### 10.0 파라미터 자동 결정

`MAX_CMDS`, `MAX_BFMS`, `MAX_IFACES`는 코드젠 단계에서 `BFMConfig[]` 배열과 Command 수로부터 자동 결정된다. 기본값은 소규모 설계(인터페이스 ≤8개)를 위한 하한이며, 대규모 설계에서는 코드젠이 자동으로 상향한다.

**자동 계산 규칙:**

```python
max_bfms   = max(8,   len(bfm_configs))
max_ifaces = max(16,  max(cfg.interface_id for cfg in bfm_configs) + 1)
max_cmds   = max(256, num_commands)
```

**수동 오버라이드**: `vten.toml`의 `[backend.scheduler]` 섹션에서 지정 가능. 자동 계산 값보다 큰 경우에만 적용.

```toml
[backend.scheduler]
max_bfms = 48
max_ifaces = 48
max_cmds = 512
```

**코드젠 적용**: SVGenerator가 `tb_top.sv` 생성 시 `vten_command_scheduler` 인스턴스의 파라미터로 전달한다 (§3.3 of `06_codegen_and_cli.md` 참조).

```systemverilog
module vten_command_scheduler #(
    parameter MAX_CMDS = 256,      // 코드젠이 override
    parameter MAX_BFMS = 8,        // 코드젠이 override
    parameter MAX_IFACES = 16      // 코드젠이 override
)(
    input  logic clk,
    input  logic rst_n,
    // ← Controller: 커맨드 피드
    input  logic        feed_valid,
    input  bfm_cmd_t    feed_data,
    output logic        feed_ready,
    input  logic        feed_done,     // 배치 완료 트리거
    // → Controller: 상태 보고
    output logic        all_committed,
    output logic        all_drained,
    output logic        error_flag,
    output logic [15:0] error_cmd_id,
    output logic [15:0] error_code,
    // BFM interfaces
    vten_bfm_cmd_if.scheduler bfm [MAX_BFMS],
    // Cycle count (글로벌)
    input  int          cycle_count,
    // BFM 매핑: interface_id → BFM 인덱스. Codegen(tb_top.sv)이 연결.
    input  int          iface_to_bfm [0:MAX_IFACES-1]
);

    // ── DPI-C imports ──
    import "DPI-C" function void vten_read_command_deps(
        input int cmd_id,
        output int num_dep, output logic [15:0] dep_ids [0:3],
        output int num_cdep, output logic [15:0] cdep_ids [0:3]);

    // ── 커맨드 저장소 ──
    bfm_cmd_t cmd_store [0:MAX_CMDS-1];
    int num_loaded = 0;
    int num_commands = 0;
    logic batch_active = 0;

    // ── 의존성 저장소 (bfm_cmd_t와 분리, §D of amendment) ──
    // BFM에 전달되지 않는 Scheduler 전용 필드.
    logic [1:0]  cmd_num_dep        [0:MAX_CMDS-1];   // Issue dep 수 (0~4)
    logic [15:0] cmd_dep            [0:MAX_CMDS-1][0:3]; // Issue dep cmd_id
    logic [1:0]  cmd_num_commit_dep [0:MAX_CMDS-1];   // Commit dep 수 (0~4)
    logic [15:0] cmd_commit_dep     [0:MAX_CMDS-1][0:3]; // Commit dep cmd_id

    // ── Sync chain & Barrier fence (전처리 결과) ──
    logic [15:0] prev_sync_cmd  [0:MAX_CMDS-1];  // 16'hFFFF = 없음
    logic [15:0] barrier_fence  [0:MAX_CMDS-1];  // 16'hFFFF = 없음

    // ── BFM 매핑 ──
    // interface_id → BFM 인덱스. 포트로 입력받아 tb_top.sv에서 연결.
    // -1 = BFM 없음 (LOAD/STORE/BARRIER 등 내부 처리 커맨드)
    int cmd_bfm_map  [0:MAX_CMDS-1];

    // ── 상태 비트맵 ──
    logic issued    [0:MAX_CMDS-1];
    logic bfm_done  [0:MAX_CMDS-1];
    logic committed [0:MAX_CMDS-1];
    logic ready     [0:MAX_CMDS-1];  // Combinational (Phase 2)
    logic stats_enabled;

    assign feed_ready = !batch_active && (num_loaded < MAX_CMDS);

    // 커맨드 수신
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            num_loaded   <= 0;
            num_commands <= 0;
            batch_active <= 0;
        end else begin
            if (feed_valid && feed_ready) begin
                cmd_store[num_loaded] <= feed_data;
                num_loaded <= num_loaded + 1;
            end
            if (feed_done) begin
                num_commands <= num_loaded;
                batch_active <= 1;
                preprocess_batch(num_loaded);  // num_loaded를 인자로 전달 (NBA 우회)
            end
            if (batch_active && all_drained) begin
                batch_active <= 0;
                num_loaded   <= 0;
                num_commands <= 0;
            end
        end
    end

    // ── 전처리: 배치 시작 시 1회 실행 ──
    // 인자 n: feed_done 시점의 num_loaded를 직접 전달.
    //   → num_commands <= num_loaded 는 NBA이므로 같은 사이클에 읽히지 않음.
    //   → n을 사용해야 루프가 올바른 커맨드 수만큼 실행됨.
    // NBA 사용: always_ff 내 task에서 레지스터를 blocking(=)으로 쓰면 합성
    //   툴 경고가 발생하므로, 모든 레지스터 업데이트를 NBA(<=)로 통일.
    task automatic preprocess_batch(int n);
        // 1. Dependency 로드 (SHM에서 DPI-C로 추출)
        for (int i = 0; i < n; i++) begin
            int nd, ncd;
            logic [15:0] d [0:3], cd [0:3];
            vten_read_command_deps(i, nd, d, ncd, cd);
            cmd_num_dep[i]        <= nd[1:0];
            cmd_num_commit_dep[i] <= ncd[1:0];
            for (int j = 0; j < 4; j++) begin
                cmd_dep[i][j]        <= d[j];
                cmd_commit_dep[i][j] <= cd[j];
            end
        end

        // 2. Bitmap 초기화 (LOAD = pre-committed, 나머지 = 0)
        // feed_done 사이클에는 batch_active=0이므로 dispatch/completion 루프가
        // 실행되지 않아 NBA 충돌 없음.
        for (int i = 0; i < n; i++) begin
            issued[i]    <= 1'b0;
            bfm_done[i]  <= 1'b0;
            committed[i] <= (cmd_store[i].opcode == OP_LOAD) ? 1'b1 : 1'b0;
        end

        // 3. Sync chain & Barrier fence
        build_sync_chain(n);
        build_barrier_fences(n);

        // 4. BFM 매핑
        build_bfm_map(n);
    endtask

    task automatic build_bfm_map(int n);
        for (int i = 0; i < n; i++) begin
            case (cmd_store[i].opcode)
                OP_LOAD, OP_STORE, OP_BARRIER:
                    cmd_bfm_map[i] <= -1;  // BFM 불필요
                default:
                    cmd_bfm_map[i] <= iface_to_bfm[cmd_store[i].interface_id];
            endcase
        end
    endtask
```

> 이하 §10.1-10.8의 Scheduler 로직은
> 위의 `cmd_store[]`, `num_commands`, dependency 배열, `cmd_bfm_map[]`을 참조한다.

### 10.1 Dispatch Policy

- **Async 커맨드 (기본):** issue dependency 충족된 모든 커맨드가 같은 사이클에 디스패치 가능. 사이클당 커맨드 수 인위적 제한 없음.
- **Sync 커맨드 (flags[0]=1):** 다음 커맨드의 암묵적 의존성 — sync 커맨드 commit 후 다음 issue 가능.
- **BFM당 속도 제한:** 사이클당 BFM 하나에 최대 1 커맨드. 같은 BFM 대상 복수 ready 커맨드 시 최소 cmd_id 우선. 1사이클 지연은 수천~수백만 사이클 DUT 실행에 비해 무시 가능.
- **Cross-BFM 병렬:** 다른 BFM은 같은 사이클에 독립 수신.

### 10.2 Committed Bitmap

`committed[MAX_CMDS]` 비트마스크. 커맨드 issue는 모든 의존이 이 비트맵에 마크될 때만 가능.

**초기화:**

| 커맨드 타입 | 초기 committed 상태 |
|-------------|-------------------|
| `LOAD` | 사전 committed (bit = 1). Runtime이 Stats Region에 COMMITTED 기록. |
| 나머지 | Pending (bit = 0). |

**업데이트 소스:**

| 이벤트 | 동작 |
|--------|------|
| BFM `done_valid` 보고 | `bfm_done[cmd_id]` 설정. Commit dep 충족 시 `committed[cmd_id]` 설정. |
| STORE/BARRIER ready | `committed[cmd_id]` 즉시 설정 (BFM 디스패치 없음). |
| Commit dep 충족 | `bfm_done[cmd_id]` → `committed[cmd_id]` 승격. |

### 10.3 Sync Chain & Barrier Preprocessing

배치 시작 시 두 룩업 구조 전처리:

```systemverilog
task automatic build_sync_chain(int n);
    logic [15:0] last_sync = 16'hFFFF;
    for (int i = 0; i < n; i++) begin
        prev_sync_cmd[i] <= last_sync;
        if (cmd_store[i].sync) last_sync = i[15:0];
    end
endtask

task automatic build_barrier_fences(int n);
    logic [15:0] last_barrier = 16'hFFFF;
    for (int i = 0; i < n; i++) begin
        if (cmd_store[i].opcode == OP_BARRIER) begin
            barrier_fence[i] <= last_barrier;
            last_barrier = i[15:0];
        end else barrier_fence[i] <= last_barrier;
    end
endtask
```

### 10.4 Ready Evaluation (Combinational)

```systemverilog
always_comb begin
    ready = '0;
    for (int i = 0; i < num_commands; i++) begin
        if (issued[i] || committed[i]) continue;
        logic deps_met = 1'b1;

        // 1. Issue dependencies
        for (int d = 0; d < cmd_num_dep[i]; d++)
            if (!committed[cmd_dep[i][d]]) deps_met = 1'b0;

        // 2. Sync chain
        if (prev_sync_cmd[i] != 16'hFFFF)
            if (!committed[prev_sync_cmd[i]]) deps_met = 1'b0;

        // 3. Barrier fence
        if (barrier_fence[i] != 16'hFFFF)
            if (!committed[barrier_fence[i]]) deps_met = 1'b0;

        // Commit dep은 readiness에 영향 없음 (BFM done 후 승격에만 사용)
        if (deps_met) ready[i] = 1'b1;
    end
end
```

BARRIER readiness는 특수: 이전 모든 커맨드 committed 필요.

```systemverilog
function automatic logic barrier_ready(int barrier_id);
    for (int j = 0; j < barrier_id; j++)
        if (!committed[j]) return 1'b0;
    return 1'b1;
endfunction
```

### 10.5 Dispatch Logic

```systemverilog
always_ff @(posedge clk) begin
    bfm_used_this_cycle = '0;
    for (int b = 0; b < MAX_BFMS; b++) bfm[b].cmd_valid <= 1'b0;

    for (int i = 0; i < num_commands; i++) begin
        if (!ready[i]) continue;
        case (cmd_store[i].opcode)
            OP_STORE, OP_BARRIER: begin
                // 내부 commit — BFM 디스패치 없음
                issued[i] <= 1'b1;
                bfm_done[i] <= 1'b1;
                committed[i] <= 1'b1;
                if (stats_enabled) vten_write_cmd_status(i, COMMITTED);
            end
            default: begin
                int b = cmd_bfm_map[i];
                if (b < 0) begin
                    // BFM 매핑 불가 — 알 수 없는 opcode 또는 내부 처리 커맨드
                    report_error(i, 6);  // UNKNOWN_OPCODE (BackendErrorCode §10.13 of 00)
                end else if (!bfm_used_this_cycle[b]) begin
                    bfm[b].cmd_valid <= 1'b1;
                    bfm[b].cmd_data  <= cmd_store[i];
                    issued[i] <= 1'b1;
                    bfm_used_this_cycle[b] = 1'b1;
                    if (stats_enabled) vten_write_cmd_status(i, ISSUED);
                end
            end
        endcase
    end
end

// ── 에러 보고 ──
task automatic report_error(int cmd_id, int code);
    if (!error_flag) begin  // 첫 번째 에러만 기록
        error_flag   <= 1'b1;
        error_cmd_id <= cmd_id[15:0];
        error_code   <= code[15:0];
    end
    // Stats에도 기록
    if (stats_enabled)
        vten_write_cmd_stats(cmd_id, 4 /*ERROR_STATUS*/,
            cycle_count, cycle_count, 0, 0, 0, 0, 0);
endtask
```

### 10.6 Completion Collector

**설계 원칙:** `bfm_done` 레지스터는 NBA(`<=`)로 업데이트되므로, 같은 `always_ff` 블록 내 promotion loop는 NBA 적용 전 값(OLD)을 읽는다. `cur_done[]` local blocking variable로 현재 사이클 done을 즉시 반영하여 **same-cycle promotion**을 구현한다.

- `bfm_done[i]`: 이전 사이클까지 누적된 done 상태 (persistent register)
- `cur_done[i]`: 현재 사이클 `done_valid` 수신 여부 (local, blocking)
- promotion 조건: `(bfm_done[i] || cur_done[i]) && !committed[i]`

commit_dep 체인은 `committed[]`가 NBA이므로 **hop당 1사이클 지연**이 불가피하다. 이는 §6 Execution Trace의 "cycle 600 → committed[16]=1, cycle 601 → committed[15]=1" 동작과 정확히 일치한다.

```systemverilog
always_ff @(posedge clk) begin
    // ── 현재 사이클 done 집합 (local blocking var, NBA 우회) ──
    logic cur_done [0:MAX_CMDS-1];
    for (int i = 0; i < MAX_CMDS; i++) cur_done[i] = 1'b0;

    for (int b = 0; b < MAX_BFMS; b++) begin
        if (bfm[b].done_valid) begin
            logic [15:0] cid = bfm[b].done_cmd_id;
            cur_done[cid] = 1'b1;                   // blocking: 즉시 반영
            bfm_done[cid] <= 1'b1;                  // NBA: persistent 업데이트
            if (bfm[b].done_error) report_error(cid, bfm[b].done_error_code);
        end
    end

    // bfm_done → committed 승격 (commit dep 확인)
    // cur_done 포함으로 done_valid 수신 사이클에 즉시 승격 가능
    for (int i = 0; i < num_commands; i++) begin
        if ((bfm_done[i] || cur_done[i]) && !committed[i]) begin
            logic all_commit_deps = 1'b1;
            for (int d = 0; d < cmd_num_commit_dep[i]; d++)
                if (!committed[cmd_commit_dep[i][d]]) all_commit_deps = 1'b0;
            if (all_commit_deps) committed[i] <= 1'b1;
        end
    end
end
```

### 10.7 Termination

```systemverilog
assign all_committed = (committed[num_commands-1:0] == {num_commands{1'b1}});

// v0.4.1: all_drained은 all_committed AND 모든 BFM idle.
// BFM의 idle 신호는 내부 큐(cmd_queue, done_queue, read_pending,
// write_pending, b_queue)가 모두 비어있고 active 전송이 없을 때 assert.
// 이는 AXI B-channel 응답이 아직 pending인 경우를 올바르게 처리한다.
logic all_bfm_idle;
always_comb begin
    all_bfm_idle = 1'b1;
    for (int b = 0; b < MAX_BFMS; b++)
        if (!bfm[b].idle) all_bfm_idle = 1'b0;
end
assign all_drained = all_committed && all_bfm_idle;
```

> **변경 근거 (v0.4.1):** 이전 `all_drained = all_committed`는 S_DRAIN 상태를 무의미하게 만들었다.
> AXI4 프로토콜에서 committed(= transferred_bytes ≥ expected_bytes)와 "모든 AXI 응답 완료"는
> 다를 수 있다. 특히 Write Path에서 데이터 전송 완료(committed) 후에도 B-channel 응답이
> 큐에 남아있을 수 있고, done_queue(v0.4.1 추가)에 미보고 이벤트가 있을 수 있다.
> `all_bfm_idle`은 이 모든 in-flight 상태가 해소되었음을 보장한다.

### 10.8 Cycle Budget Summary

| 단계 | 로직 타입 | 작업 |
|------|-----------|------|
| Phase 1: Completion | Sequential | 모든 BFM `done_valid` 수집. `bfm_done` 업데이트. Commit dep 충족 시 `committed` 승격. |
| Phase 2: Ready eval | Combinational | `committed` 비트맵, sync chain, barrier fence 대비 `ready[i]` 평가. |
| Phase 3: Dispatch | Sequential | Ready 커맨드를 BFM에 디스패치 (BFM당 1개/사이클) 또는 내부 commit (STORE/BARRIER). |
| Phase 4: Stats | Sequential | `STATS_ENABLED` 시 DPI-C로 Stats 엔트리 기록. |
