# vTen BFM Library

**Version 0.5.0 — March 2026**
**참조: `00_data_models.md`, `04_backend_xsim.md` (Scheduler ↔ BFM Interface)**
**소스: 메인 스펙 §11.16-11.19, §12**

---

## Table of Contents

1. [AXI4-Stream BFM](#1-axi4-stream-bfm)
2. [AXI4 Memory-Mapped BFM](#2-axi4-memory-mapped-bfm)
3. [AXI4-Lite BFM](#3-axi4-lite-bfm)
4. [Debug and Probe System](#4-debug-and-probe-system)
5. [Per-Command Statistics](#5-per-command-statistics)
6. [Execution Trace Example](#6-execution-trace-example)

---

## 1. AXI4-Stream BFM

PUSH (master: SHM → DUT) 및 PULL (slave: DUT → SHM) 커맨드 처리. SV dynamic queue에 저장하고 순차 실행 — 단일 스트림 채널은 한 번에 하나의 전송만 수행.

### 1.1 Architecture

```
Scheduler ──cmd_valid──► queue[$] ──pop──► Execution FSM ──► AXI4-Stream signals
                                                │
                                           done_valid ──► Scheduler
```

### 1.2 Implementation

```systemverilog
module vten_bfm_axi4s #(
    parameter DATA_W = 256,
    parameter MODE   = "MASTER"
)(
    input  logic clk, input logic rst_n,
    // AXI4-Stream
    output logic [DATA_W-1:0]   m_tdata,
    output logic                m_tvalid,
    input  logic                m_tready,
    output logic                m_tlast,
    input  logic [DATA_W-1:0]   s_tdata,
    input  logic                s_tvalid,
    output logic                s_tready,
    input  logic                s_tlast,
    // Scheduler interface
    vten_bfm_cmd_if.bfm         cmd_if,
    // Global cycle counter (from tb_top)
    input  int                  cycle_count
);
    localparam BYTES_PER_BEAT = DATA_W / 8;

    bfm_cmd_t cmd_queue[$];
    bfm_cmd_t current_cmd;
    logic      cmd_active;
    int beat_count, expected_beats;
    int issue_cycle, first_active, last_active;
    int active_cycles, stall_cycles, total_beats;

    // v0.4.1: idle 신호 — Scheduler의 all_drained 산출에 사용
    assign cmd_if.idle = !cmd_active && (cmd_queue.size() == 0);

    // Command receive
    always_ff @(posedge clk) begin
        if (cmd_if.cmd_valid) cmd_queue.push_back(cmd_if.cmd_data);
    end

    // Command activation & execution
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_active <= 0; cmd_if.done_valid <= 0;
            m_tvalid <= 0; s_tready <= 0;
        end else begin
            cmd_if.done_valid <= 0;

            if (!cmd_active && cmd_queue.size() > 0) begin
                current_cmd = cmd_queue.pop_front();
                cmd_active <= 1; beat_count <= 0;
                expected_beats <= current_cmd.size / BYTES_PER_BEAT;
                issue_cycle <= cycle_count;
                first_active <= 0; last_active <= 0;
                active_cycles <= 0; stall_cycles <= 0; total_beats <= 0;
            end

            if (cmd_active) begin
                if (MODE == "MASTER") execute_master();
                else                  execute_slave();
            end
        end
    end

    // Bulk transfer buffer: byte[] — contiguous 1-byte stride on all simulators
    byte beat_buf [0:BYTES_PER_BEAT-1];

    // MASTER mode: PUSH (SHM → DUT)
    // NBA-aware: on handshake, drive NEXT beat; on initial, drive beat 0.
    task automatic execute_master();
        if (m_tvalid && m_tready) begin
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;
            if (beat_count == expected_beats - 1) begin
                m_tvalid <= 1'b0;
                beat_count <= beat_count + 1;
                finish_command();
            end else begin
                // Drive next beat — bulk read
                vten_read_data_bulk(current_cmd.buffer_id,
                    (beat_count + 1) * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    m_tdata[i*8 +: 8] <= beat_buf[i];
                m_tlast <= ((beat_count + 1) == expected_beats - 1);
                beat_count <= beat_count + 1;
            end
        end else if (!m_tvalid) begin
            // Initial: drive first beat — bulk read
            vten_read_data_bulk(current_cmd.buffer_id,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
            for (int i = 0; i < BYTES_PER_BEAT; i++)
                m_tdata[i*8 +: 8] <= beat_buf[i];
            m_tvalid <= 1'b1;
            m_tlast  <= (expected_beats == 1);
        end else
            stall_cycles <= stall_cycles + 1;
    endtask

    // SLAVE mode: PULL (DUT → SHM)
    task automatic execute_slave();
        s_tready <= 1'b1;
        if (s_tvalid && s_tready) begin
            // Bulk write: pack tdata into byte buffer, then single memcpy
            for (int i = 0; i < BYTES_PER_BEAT; i++)
                beat_buf[i] = s_tdata[i*8 +: 8];
            vten_write_data_bulk(current_cmd.buffer_id,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
            beat_count <= beat_count + 1;
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;

            // Probe mode: beat-by-beat golden comparison
            if (current_cmd.probe) begin : probe_blk
                logic [DATA_W-1:0] golden;
                byte golden_buf [0:BYTES_PER_BEAT-1];
                vten_read_golden_bulk(current_cmd.golden_buf_id,
                    beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, golden_buf);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    golden[i*8 +: 8] = golden_buf[i];
                if (s_tdata !== golden)
                    vten_log_mismatch(cycle_count, beat_count,
                                      golden[DATA_W-1:DATA_W/2],
                                      golden[DATA_W/2-1:0],
                                      s_tdata[DATA_W-1:DATA_W/2],
                                      s_tdata[DATA_W/2-1:0]);
            end

            if (beat_count == expected_beats - 1) begin
                s_tready <= 1'b0;
                finish_command();
            end
        end else if (!s_tvalid && s_tready)
            stall_cycles <= stall_cycles + 1;
    endtask

    task automatic finish_command();
        cmd_active <= 0;
        cmd_if.done_valid  <= 1'b1;
        cmd_if.done_cmd_id <= current_cmd.cmd_id;
        cmd_if.done_error  <= 1'b0;
        vten_write_cmd_stats(current_cmd.cmd_id,
            COMMITTED, issue_cycle, cycle_count,
            first_active, last_active,
            active_cycles, total_beats, stall_cycles);
    endtask
endmodule
```

---

## 2. AXI4 Memory-Mapped BFM

DUT가 AXI 마스터, BFM이 **슬레이브**. DUT가 read(AR/R) 및 write(AW/W/B) 트랜잭션을 시작하면 BFM이 SHM 버퍼 데이터로 응답.

AXI4-Stream BFM과 달리 **복수 커맨드가 동시에 활성화** 가능. 각 PUSH 커맨드는 DUT read 요청을 서빙하는 주소 범위를 등록하고, PULL은 DUT write를 수신하는 범위를 등록.

### 2.1 Internal Architecture

```
Scheduler ──cmd_valid──► Active Table (queue[$])
                              │
                    ┌─────────┴─────────┐
                    │                   │
              AR channel           AW/W channel
              (DUT reads)          (DUT writes)
                    │                   │
              addr match           addr match
              → PUSH entry         → PULL entry
                    │                   │
              R channel            B channel
              (serve data)         (write response)
                    │                   │
              beat tracking        beat tracking
                    │                   │
              completion ──────► done_valid
```

### 2.2 Module Declaration

```systemverilog
module vten_bfm_axi4 #(
    parameter DATA_W = 256,
    parameter ADDR_W = 64
)(
    input  logic clk, input logic rst_n,
    // AXI4 Slave (DUT is master)
    input  logic [ADDR_W-1:0]   s_araddr,
    input  logic [7:0]          s_arlen,
    input  logic [2:0]          s_arsize,
    input  logic [1:0]          s_arburst,
    input  logic                s_arvalid,
    output logic                s_arready,
    output logic [DATA_W-1:0]   s_rdata,
    output logic [1:0]          s_rresp,
    output logic                s_rlast,
    output logic                s_rvalid,
    input  logic                s_rready,
    input  logic [ADDR_W-1:0]   s_awaddr,
    input  logic [7:0]          s_awlen,
    input  logic [2:0]          s_awsize,
    input  logic [1:0]          s_awburst,
    input  logic                s_awvalid,
    output logic                s_awready,
    input  logic [DATA_W-1:0]   s_wdata,
    input  logic [DATA_W/8-1:0] s_wstrb,
    input  logic                s_wlast,
    input  logic                s_wvalid,
    output logic                s_wready,
    output logic [1:0]          s_bresp,
    output logic                s_bvalid,
    input  logic                s_bready,
    // Scheduler interface
    vten_bfm_cmd_if.bfm         cmd_if,
    // Global cycle counter (from tb_top)
    input  int                  cycle_count
);
```

### 2.3 Active Table

```systemverilog
typedef struct {
    bfm_cmd_t   cmd;
    logic       active;
    int         transferred_bytes;
    int         expected_bytes;
    int         issue_cycle, first_active, last_active;
    int         active_cycles, stall_cycles, total_beats;
} active_entry_t;

active_entry_t active_table[$];

// v0.4.1: Done Queue — 동일 사이클 복수 완료 충돌 방지
// Read Path와 Write Path에서 동시에 check_completion()이 호출되면
// done_valid에 대한 last-write-wins로 하나의 완료 보고가 손실된다.
// Done Queue에 push 후, 매 사이클 하나씩 pop하여 Scheduler에 보고.
typedef struct {
    logic [15:0] cmd_id;
    logic        error;
    logic [15:0] error_code;
} done_event_t;

done_event_t done_queue[$];

// v0.4.1: idle 신호
// 모든 내부 큐와 pending 상태가 비어있을 때 assert.
// Scheduler가 all_drained 산출에 사용.
assign cmd_if.idle = (active_table.size() == 0)
                  && (read_pending.size() == 0)
                  && (write_pending.size() == 0)
                  && (b_queue.size() == 0)
                  && (done_queue.size() == 0)
                  && !r_active && !w_active;
```

> **cycle_count:** AXI4 BFM은 `input int cycle_count` 포트로 글로벌 카운터를 수신한다
> (모듈 선언부에 추가, 이 문서에서는 간결함을 위해 포트 목록 생략).
> 모든 Stats 기록(`issue_cycle`, `first_active` 등)에 이 포트 값을 사용.

### 2.4 Address Matching

```systemverilog
function automatic int find_entry(logic [ADDR_W-1:0] addr, opcode_t op);
    foreach (active_table[i]) begin
        if (!active_table[i].active) continue;
        if (active_table[i].cmd.opcode != op) continue;
        logic [63:0] base = active_table[i].cmd.phys_addr;
        logic [63:0] top  = base + active_table[i].cmd.size;
        if (addr >= base && addr < top) return i;
    end
    return -1;  // DECERR
endfunction
```

- DUT read (AR) → `OP_PUSH` 엔트리 매칭
- DUT write (AW) → `OP_PULL` 엔트리 매칭

> **동시 AR/AW 처리 정책 (v0.4.2):**
>
> AXI4 BFM은 Read Path와 Write Path를 완전히 독립된 `always_ff` 블록으로 구현한다.
> 아래 동작은 모두 스펙으로 보장된다:
>
> | 항목 | 동작 |
> |------|------|
> | AR + AW 동일 사이클 수락 | 허용. `s_arready`와 `s_awready`가 각각 항상 1. |
> | R-burst + W-burst 동시 in-flight | 허용. `r_active`와 `w_active`는 독립 신호이며 별도 `always_ff` 블록에서 관리. |
> | `find_entry()` 동일 사이클 동시 호출 | 안전. AR path는 `OP_PUSH`, AW path는 `OP_PULL`만 검색하므로 서로 다른 엔트리를 참조. `active_table` 읽기는 비파괴적. |
> | `active_table` 엔트리 동시 수정 | 안전. R/W Path는 서로 다른 인덱스(PUSH vs PULL 엔트리)를 수정하므로 충돌 없음. |
> | `done_queue` 동시 push | 안전. v0.4.1에서 `done_queue`를 도입하여 동일 사이클 복수 완료 시 last-write-wins 손실 문제 해결. |
>
> 따라서 AR/AW 동시 처리를 위한 별도의 arbiter나 mutex는 필요하지 않다.

### 2.5 Read Path (DUT reads ← BFM serves)

```systemverilog
// AR channel: ideal slave, always accept
always_ff @(posedge clk) begin
    s_arready <= 1'b1;
    if (s_arvalid && s_arready) begin
        int idx = find_entry(s_araddr, OP_PUSH);
        if (idx >= 0) begin
            read_pending.push_back('{
                entry_idx: idx,
                addr: s_araddr,
                len: s_arlen,
                size: s_arsize,
                burst: s_arburst,
                id: s_arid
            });
        end else begin
            // Address not matched: DECERR
            read_pending.push_back('{entry_idx: -1, ...});
        end
    end
end

// R channel: serve data from SHM
always_ff @(posedge clk) begin
    if (read_pending.size() > 0 && !r_active) begin
        current_read = read_pending.pop_front();
        r_active <= 1; r_beat <= 0;
    end

    if (r_active) begin
        if (current_read.entry_idx < 0) begin
            // DECERR response
            s_rvalid <= 1; s_rresp <= 2'b11; s_rlast <= 1;
            if (s_rready) r_active <= 0;
        end else begin
            active_entry_t* entry = &active_table[current_read.entry_idx];
            int offset = (current_read.addr - entry.cmd.phys_addr)
                         + r_beat * (1 << current_read.size);

            int transfer_size = 1 << current_read.size;
            byte r_beat_buf [0:BYTES_PER_BEAT-1];
            vten_read_data_bulk(entry.cmd.buffer_id, offset,
                               transfer_size, r_beat_buf);
            for (int i = 0; i < transfer_size; i++)
                s_rdata[i*8 +: 8] <= r_beat_buf[i];
            s_rvalid <= 1;
            s_rid    <= current_read.id;
            s_rresp  <= 2'b00;
            s_rlast  <= (r_beat == current_read.len);

            if (s_rvalid && s_rready) begin
                r_beat <= r_beat + 1;
                entry.active_cycles++;
                entry.total_beats++;
                if (entry.first_active == 0) entry.first_active = cycle_count;
                entry.last_active = cycle_count;

                if (r_beat == current_read.len) begin
                    s_rvalid <= 0;
                    r_active <= 0;
                    entry.transferred_bytes += (current_read.len + 1)
                                               * (1 << current_read.size);
                    check_completion(current_read.entry_idx);
                end
            end
        end
    end
end
```

### 2.6 Write Path (DUT writes → BFM captures)

```systemverilog
// AW channel: ideal slave
always_ff @(posedge clk) begin
    s_awready <= 1'b1;
    if (s_awvalid && s_awready) begin
        int idx = find_entry(s_awaddr, OP_PULL);
        write_pending.push_back('{
            entry_idx: idx, addr: s_awaddr,
            len: s_awlen, size: s_awsize,
            burst: s_awburst, id: s_awid
        });
    end
end

// W channel: capture data to SHM
always_ff @(posedge clk) begin
    if (write_pending.size() > 0 && !w_active) begin
        current_write = write_pending.pop_front();
        w_active <= 1; w_beat <= 0;
    end

    s_wready <= w_active;

    if (w_active && s_wvalid && s_wready) begin
        if (current_write.entry_idx >= 0) begin
            active_entry_t* entry = &active_table[current_write.entry_idx];
            int offset = (current_write.addr - entry.cmd.phys_addr)
                         + w_beat * (1 << current_write.size);

            // WSTRB 처리: fast path (all-ones) vs slow path (partial)
            int transfer_size = 1 << current_write.size;
            if (s_wstrb == {BYTES_PER_BEAT{1'b1}}) begin
                // Fast path: all bytes valid → bulk write
                byte w_beat_buf [0:BYTES_PER_BEAT-1];
                for (int b = 0; b < transfer_size; b++)
                    w_beat_buf[b] = s_wdata[b*8 +: 8];
                vten_write_data_bulk(
                    entry.cmd.buffer_id, offset, transfer_size, w_beat_buf);
            end else begin
                // Slow path: partial WSTRB → scalar byte-by-byte
                for (int b = 0; b < transfer_size; b++) begin
                    if (s_wstrb[b])
                        vten_write_data_byte(
                            entry.cmd.buffer_id, offset + b,
                            s_wdata[b*8 +: 8]);
                end
            end

            entry.active_cycles++;
            entry.total_beats++;
            if (entry.first_active == 0) entry.first_active = cycle_count;
            entry.last_active = cycle_count;

            // Probe mode: bulk golden read
            if (entry.cmd.probe) begin
                logic [DATA_W-1:0] golden;
                byte golden_buf [0:BYTES_PER_BEAT-1];
                int golden_offset = (entry.total_beats - 1) * BYTES_PER_BEAT;
                vten_read_golden_bulk(entry.cmd.golden_buf_id,
                    golden_offset, BYTES_PER_BEAT, golden_buf);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    golden[i*8 +: 8] = golden_buf[i];
                if (s_wdata !== golden)
                    vten_log_mismatch(cycle_count, entry.total_beats - 1,
                                      golden[DATA_W-1:DATA_W/2],
                                      golden[DATA_W/2-1:0],
                                      s_wdata[DATA_W-1:DATA_W/2],
                                      s_wdata[DATA_W/2-1:0]);
            end
        end

        w_beat <= w_beat + 1;
        if (s_wlast) begin
            w_active <= 0;
            if (current_write.entry_idx >= 0) begin
                active_entry_t* entry = &active_table[current_write.entry_idx];
                entry.transferred_bytes += (current_write.len + 1)
                                           * (1 << current_write.size);
                check_completion(current_write.entry_idx);
            end
            // B channel response
            b_queue.push_back('{id: current_write.id,
                resp: (current_write.entry_idx >= 0) ? 2'b00 : 2'b11});
        end
    end
end

// B channel: write response
always_ff @(posedge clk) begin
    if (b_queue.size() > 0) begin
        s_bvalid <= 1; s_bid <= b_queue[0].id; s_bresp <= b_queue[0].resp;
        if (s_bready) b_queue.pop_front();
    end else s_bvalid <= 0;
end
```

### 2.7 Completion Tracking

```systemverilog
// v0.4.1: check_completion은 done_queue에 push.
// 직접 done_valid를 세팅하지 않아 동일 사이클 복수 완료 시 충돌 방지.
task automatic check_completion(int idx);
    active_entry_t* entry = &active_table[idx];
    if (entry.transferred_bytes >= entry.expected_bytes) begin
        entry.active = 0;
        done_queue.push_back('{
            cmd_id:     entry.cmd.cmd_id,
            error:      0,
            error_code: 0
        });
        vten_write_cmd_stats(entry.cmd.cmd_id,
            COMMITTED, entry.issue_cycle, cycle_count,
            entry.first_active, entry.last_active,
            entry.active_cycles, entry.total_beats, entry.stall_cycles);
    end
endtask

// v0.4.1: Done Queue drain — 매 사이클 하나씩 Scheduler에 보고
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cmd_if.done_valid <= 0;
    end else begin
        cmd_if.done_valid <= 0;  // 기본 deassert
        if (done_queue.size() > 0) begin
            done_event_t ev = done_queue.pop_front();
            cmd_if.done_valid      <= 1;
            cmd_if.done_cmd_id     <= ev.cmd_id;
            cmd_if.done_error      <= ev.error;
            cmd_if.done_error_code <= ev.error_code;
        end
    end
end
```

> **변경 근거 (v0.4.1):** Read Path의 `check_completion`과 Write Path의 `check_completion`이
> 같은 사이클에 호출되면, SV last-write-wins로 `cmd_if.done_valid` 할당이 충돌한다.
> Done Queue를 통해 복수 완료 이벤트를 순차 보고한다. 1사이클 추가 지연은
> 수천 사이클 DUT 실행에 비해 무시 가능.

### 2.8 DECERR Error Reporting (v0.4.2)

`find_entry()` 주소 매칭 실패 시 DUT에 DECERR 응답을 반환하고, Scheduler에도 에러를 보고한다.
`BackendErrorCode.ADDR_UNMATCH(1)` 사용 (`00_data_models.md` §10.13).

```systemverilog
// Read path에서 DECERR 응답 완료 후:
if (current_read.entry_idx < 0 && s_rready) begin
    r_active <= 0;
    done_queue.push_back('{
        cmd_id:     16'hFFFF,  // 매칭 실패 — 특정 커맨드 불명
        error:      1'b1,
        error_code: 16'd1      // ADDR_UNMATCH
    });
end

// Write path에서 DECERR 응답 완료 후:
// b_queue에 DECERR 응답이 이미 push된 상태에서 추가로:
if (current_write.entry_idx < 0 && s_wlast) begin
    done_queue.push_back('{
        cmd_id:     16'hFFFF,
        error:      1'b1,
        error_code: 16'd1      // ADDR_UNMATCH
    });
end
```

> **`cmd_id = 0xFFFF`:** DUT의 잘못된 주소 접근은 특정 커맨드에 귀속할 수 없는 경우가 있다.
> Scheduler는 `cmd_id == 0xFFFF`를 "unattributed error"로 처리하여 배치를 ERROR로 종료한다.

---

## 3. AXI4-Lite BFM

레지스터 read/write/poll 처리. BFM이 **마스터** — AXI-Lite 트랜잭션을 구동.

```systemverilog
module vten_bfm_axilite #(parameter ADDR_W = 32, DATA_W = 32)(
    input logic clk, input logic rst_n,
    // AXI4-Lite Master
    output logic [ADDR_W-1:0]  m_awaddr,
    output logic               m_awvalid,
    input  logic               m_awready,
    output logic [DATA_W-1:0]  m_wdata,
    output logic [DATA_W/8-1:0] m_wstrb,
    output logic               m_wvalid,
    input  logic               m_wready,
    input  logic [1:0]         m_bresp,
    input  logic               m_bvalid,
    output logic               m_bready,
    output logic [ADDR_W-1:0]  m_araddr,
    output logic               m_arvalid,
    input  logic               m_arready,
    input  logic [DATA_W-1:0]  m_rdata,
    input  logic [1:0]         m_rresp,
    input  logic               m_rvalid,
    output logic               m_rready,
    // Scheduler interface
    vten_bfm_cmd_if.bfm        cmd_if,
    // Global cycle counter (from tb_top)
    input  int                 cycle_count
);
    bfm_cmd_t cmd_queue[$];
    bfm_cmd_t current_cmd;
    logic cmd_active;
    int poll_count, poll_interval, poll_timeout;

    // v0.4.1: idle 신호
    assign cmd_if.idle = !cmd_active && (cmd_queue.size() == 0);

    always_ff @(posedge clk)
        if (cmd_if.cmd_valid) cmd_queue.push_back(cmd_if.cmd_data);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin /* reset */ end
        else begin
            cmd_if.done_valid <= 0;
            if (!cmd_active && cmd_queue.size() > 0) begin
                current_cmd = cmd_queue.pop_front();
                cmd_active <= 1;
                poll_count <= 0;
            end
            if (cmd_active) begin
                case (current_cmd.opcode)
                    OP_WRITE_REG: do_write();
                    OP_READ_REG:  do_read();
                    OP_POLL_REG:  do_poll();
                endcase
            end
        end
    end

    task automatic do_write();
        // AW + W 동시 구동
        m_awaddr  <= current_cmd.reg_offset;
        m_awvalid <= 1;
        m_wdata   <= current_cmd.reg_value;
        m_wstrb   <= '1;
        m_wvalid  <= 1;
        m_bready  <= 1;

        if (m_awready) m_awvalid <= 0;
        if (m_wready)  m_wvalid  <= 0;
        if (m_bvalid && m_bready) begin
            m_bready <= 0;
            finish_cmd(m_bresp != 2'b00);
        end
    endtask

    task automatic do_read();
        m_araddr  <= current_cmd.reg_offset;
        m_arvalid <= 1;
        m_rready  <= 1;

        if (m_arready) m_arvalid <= 0;
        if (m_rvalid && m_rready) begin
            m_rready <= 0;
            // 읽은 값을 Stats에 기록 (또는 SHM의 reg_value에 다시 기록)
            vten_write_cmd_stats(current_cmd.cmd_id, ...);
            finish_cmd(m_rresp != 2'b00);
        end
    endtask

    task automatic do_poll();
        // 설정 가능한 간격으로 반복 읽기
        if (poll_count % POLL_INTERVAL == 0) begin
            m_araddr  <= current_cmd.reg_offset;
            m_arvalid <= 1;
            m_rready  <= 1;
        end

        if (m_rvalid && m_rready) begin
            m_rready <= 0;
            if ((m_rdata & current_cmd.reg_mask) == current_cmd.reg_expected) begin
                finish_cmd(0);
            end else begin
                poll_count <= poll_count + 1;
                if (poll_count >= POLL_TIMEOUT) begin
                    finish_cmd(1);  // 타임아웃 에러
                end
            end
        end
    endtask

    task automatic finish_cmd(logic error);
        cmd_active <= 0;
        cmd_if.done_valid    <= 1;
        cmd_if.done_cmd_id   <= current_cmd.cmd_id;
        cmd_if.done_error    <= error;
        // v0.4.2: BackendErrorCode 사용 (00_data_models.md §10.13)
        // poll 타임아웃 = POLL_TIMEOUT(2), 기타 에러 = BFM_QUEUE_ERROR(3)
        if (error && current_cmd.opcode == OP_POLL_REG)
            cmd_if.done_error_code <= 16'd2;  // POLL_TIMEOUT
        else if (error)
            cmd_if.done_error_code <= 16'd3;  // BFM_QUEUE_ERROR
        else
            cmd_if.done_error_code <= 16'd0;  // OK
    endtask
endmodule
```

**POLL_REG의 블로킹:** POLL 자체는 블로킹이 아니다 — 블로킹은 의존성 그래프가 결정한다. BFM은 설정 가능한 간격과 타임아웃으로 폴링.

---

## 4. Debug and Probe System

### 4.1 Probe 모드 분류

Probe는 두 가지 독립적인 모드로 동작한다.

**출력 Probe** — `pull_tensor(probe=True)`

DUT 출력 인터페이스에서 BFM이 수신 비트를 golden 텐서와 직접 비교한다.

```python
pull1 = ctx.pull_tensor(kernel.ofm, dep=push1, probe=True)
```

Runtime이 처리하는 절차:
1. golden 텐서를 동일 PackingScheme으로 직렬화 → SHM에 별도 버퍼로 저장
2. probe BFM에 해당 `buffer_id`를 plusarg로 전달
3. BFM이 AXI4-Stream 핸드쉐이크마다 SHM golden 데이터와 비트별 비교
4. 불일치 시 stderr 로그 + `mismatches.jsonl` 기록, `probe_error` 신호 어서트

**내부 Probe** — `Internal(probe=True)` in CompositeKernel

서브커널 간 내부 배선에 패시브 모니터 BFM을 삽입하여 중간값을 검증한다.

```python
class NPUTopKernel(CompositeKernel):
    mac = MACKernel.bind(
        interface_map={
            "axis_ifm": Internal(probe=True),
            "axis_ofm": Internal(probe=True),
        }
    )
```

테스트 시나리오에서 내부 probe의 golden 데이터를 등록:

```python
# TestScenario.run() 내부
ctx.set_internal_probe_golden("mac", "axis_ifm", golden_tensor)
```

Engine이 내부 probe golden 텐서를 SHM에 직렬화하고, probe BFM이 plusarg를 통해 `buffer_id`를 조회하여 비교를 수행한다.

---

### 4.2 Probe BFM (`vten_bfm_axi4s_probe.sv`)

AXI4-Stream 인터페이스를 패시브하게 모니터링한다. `tready`를 항상 1로 구동하므로 DUT 데이터 흐름에 역압(backpressure)을 가하지 않는다.

```systemverilog
module vten_bfm_axi4s_probe #(
    parameter int DATA_W      = 256,
    parameter int PROBE_INDEX = 0
)(
    input  logic clk,
    input  logic [DATA_W-1:0] tdata,
    input  logic              tvalid,
    output logic              tready,
    input  logic              tlast,
    output logic              probe_error    // 불일치 발생 시 어서트
);
```

**동작 순서:**

1. 시뮬레이션 시작 시 `$value$plusargs("PROBE_GOLDEN_<PROBE_INDEX>=%d", buffer_id)`로 SHM buffer_id 획득
2. `tvalid && tready` 핸드쉐이크마다 `vten_read_golden_bulk()`로 SHM에서 해당 beat의 golden 데이터 읽기
3. `tdata !== golden`이면:
   - `probe_error <= 1'b1` 어서트
   - `vten_log_mismatch()` 호출 → stderr + `mismatches.jsonl` 기록
   - `vten_read_flags() & 8` (FLAG_PAUSE_ON_MISMATCH, `--gui` 옵션)이면 `$stop` 실행
4. `probe_error`가 어서트된 이후에는 `!probe_error` 가드로 추가 비교를 중단

`tready`는 1로 고정된다.

---

### 4.2b Probe 에러 전파 경로

```
probe_error (각 probe BFM)
    │
    ▼ OR 게이트 (tb_top)
probe_error_any
    │
    ▼ Controller probe_error 입력
Controller S_ERROR (error_code = 8, ERR_PROBE_MISMATCH)
    │
    ▼ SHM BACKEND_ERROR 기록
Python ProbeMismatchError 발생
```

Controller는 scheduler 에러(BFM 타임아웃 등)와 probe 에러(code=8)를 별도 코드로 구분한다.

---

### 4.3 Mismatch 리포트

C 브릿지 `vten_log_mismatch()`는 두 곳에 기록한다:

1. **stderr** (즉시 출력):
   ```
   [PROBE MISMATCH] cmd=0 cycle=46 beat=0 expected=0x000...abc actual=0x000...def
   ```

2. **`$VTEN_MISMATCH_DIR/mismatches.jsonl`** (JSON Lines 형식):
   ```json
   {"cmd_id": 0, "cycle": 46, "beat": 0, "expected_hi": 0, "expected_lo": 2748779069, "actual_hi": 0, "actual_lo": 3735928559}
   ```

Python 런타임은 시뮬레이션 완료 후 `mismatches.jsonl`을 파싱하여 `ProbeMismatchError(beat_index, mismatches, cmd_id)`를 발생시킨다. 테스트 결과 디렉토리에 `mismatches.json`이 저장된다.

---

### 4.4 내부 Probe Golden 등록 API

```python
# vten/runtime/context.py
ctx.set_internal_probe_golden(
    sub_kernel_name: str,   # CompositeKernel 내 서브커널 이름 ("mac")
    tensor_name:     str,   # 해당 서브커널의 텐서 이름 ("axis_ifm")
    golden:          Tensor # 기대값 텐서
)
```

Engine Stage 3 (Tensor Serialization)에서 내부 probe golden 텐서를 일반 입출력 텐서와 동일한 방식으로 SHM에 직렬화한다. 생성된 `buffer_id`는 codegen 단계에서 `PROBE_GOLDEN_<INDEX>` plusarg 형태로 xsim 시뮬레이션 명령줄에 삽입된다.

---

### 4.5 Waveform Dump

```bash
$ vten run --test test_conv3d --waveform            # 항상 덤프
$ vten run --test test_conv3d --waveform-on-fail    # 실패 시에만 덤프
$ vten run --test test_conv3d --gui                 # 대화형 GUI 모드
$ vten run --test test_conv3d --gui --waveform      # GUI + waveform 신호 자동 등록
```

`templates/waveform.tcl.j2`를 렌더링한 `waveform.tcl`이 생성되며, DUT / BFM / probe / scheduler 신호를 자동 등록한다. 배치 모드에서는 xsim `--tclbatch` 옵션으로 주입된다.

---

### 4.6 GUI 모드 (`--gui`) 와 `$stop` 재시작 핸들링

`--gui` 옵션이 지정되면 SHM flags의 bit 3(`FLAG_PAUSE_ON_MISMATCH = 0x08`)이 세트된다.

**SHM 플래그 정의:**

```python
FLAG_STATS_ENABLED     = 0x01  # 항상 세트
FLAG_PROGRESS_ENABLED  = 0x02
FLAG_WAVEFORM_DUMP     = 0x04
FLAG_PAUSE_ON_MISMATCH = 0x08  # --gui 옵션 시 세트
```

**GUI 모드 동작 흐름:**

1. Probe BFM이 불일치 검출 시 `vten_read_flags() & 8` 확인 → `$stop` 실행
2. xsim이 일시 정지하면 사용자가 waveform을 확인
3. Python은 24시간 타임아웃으로 대기하며 재시작 사이클을 처리:
   - **에러 발생:** 에러 로그 기록 → SHM 리셋 → CMD_READY 재전송
   - **재시작:** `vten_shm_init()` b2h 신호 감지 → CMD_READY 재전송
   - **xsim 종료:** 누적 결과와 함께 graceful exit
4. KeyboardInterrupt 발생 시 graceful shutdown

---

## 5. Per-Command Statistics

### 5.1 Stats Collection in BFMs

```systemverilog
always_ff @(posedge clk) begin
    if (cmd_active) begin
        if (tvalid && tready) begin
            stats.active_cycles <= stats.active_cycles + 1;
            stats.total_beats   <= stats.total_beats + 1;
            if (stats.first_active_cycle == 0)
                stats.first_active_cycle <= cycle_count;
            stats.last_active_cycle <= cycle_count;
        end else if (tvalid && !tready)
            stats.stall_cycles <= stats.stall_cycles + 1;
        else if (!tvalid && tready)
            stats.stall_cycles <= stats.stall_cycles + 1;
    end
end
```

### 5.2 Report Output Example

```
═══════════════════════════════════════════════════════════════════
  vTen Execution Report — conv3d_top (C=64, D=32, H=32, W=32)
═══════════════════════════════════════════════════════════════════

  Command Summary
  ─────────────────────────────────────────────────────────────────
  ID  Op          Interface    Latency   Active   Util    Status
   0  LOAD        -              -         -       -      OK
   1  LOAD        -              -         -       -      OK
   2  WRITE_REG   ctrl           2 cyc     -       -      OK
   ...
   5  PUSH        data_port    1,204 cyc   1,024   85.0%  OK
   6  PUSH        data_port      512 cyc     480   93.8%  OK
   7  PULL        data_port    2,048 cyc   1,920   93.8%  OK
   8  POLL_REG    ctrl         2,060 cyc     -       -    OK

  Timeline (cycle scale)
  ─────────────────────────────────────────────────────────────────
  Cycle:    0        500      1000     1500     2000     2500
  PUSH.ifm  |████████████████████░░░░|
  PUSH.wgt           |██████████░|
  PULL.ofm                       |█████████████████████████|
  POLL.done                      |........................█|

  Total: 2,512 cycles  |  Verification: PASS
═══════════════════════════════════════════════════════════════════
```

---

## 6. Execution Trace Example

Scheduler가 처리하는 커맨드 배치의 실행 흐름 예시:

```
cycle  0: Batch start. committed = {0,1} (LOADs pre-committed)
cycle  1: cmd 2-11 (WRITE_REGs) all ready (no deps). Dispatch cmd 2 to ctrl BFM.
cycle  2: cmd 2 completes. Dispatch cmd 3.
...
cycle 11: cmd 11 completes. All WRITE_REGs done.
cycle 12: cmd 12 (start) ready: dep [0,1] committed. Dispatch to ctrl BFM.
cycle 14: cmd 12 completes.
cycle 15: cmd 13 (PUSH ifm) ready: dep [12]. Dispatch to data_port BFM.
          cmd 14 (PUSH wgt) ready: dep [12]. Same BFM → deferred.
          cmd 16 (POLL done) ready: dep [12]. Dispatch to ctrl BFM.
cycle 16: cmd 14 dispatched to data_port BFM.
...
cycle 500: cmd 15 (PULL ofm) BFM done. But commit_dep=[16] (POLL not yet committed).
           bfm_done[15]=1, committed[15]=0 (held).
cycle 600: cmd 16 (POLL) detects done=1. committed[16]=1.
cycle 601: cmd 15 promoted: committed[15]=1 (commit_dep [16] now satisfied).
cycle 602: cmd 17 (STORE) ready: dep [15]. Committed immediately (no BFM).
cycle 602: all_committed = 1. → S_DRAIN → S_COMPLETE.
```

**핵심 관찰:**
1. LOAD 사전 committed → 후속 커맨드가 LOAD 완료 대기 없이 즉시 dispatch 가능
2. 같은 BFM(ctrl) 대상 WRITE_REG는 사이클당 1개씩 순차 dispatch
3. Cross-BFM 병렬: cmd 13 (AXI4)과 cmd 16 (AXI-Lite)이 동시 dispatch
4. Commit dependency: PULL의 BFM 전송은 사이클 ~500에서 완료되나, POLL 완료(~600) 전까지 committed 승격 보류
5. STORE는 의존(PULL committed) 충족 즉시 committed (BFM 없음)
