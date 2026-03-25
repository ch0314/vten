# SV Convenience Features: Interface, AXI-Lite Controller, Wrapper Generation

**Version 0.1.0 — March 2026**
**Status: 설계 문서 (구현 전)**

---

## Table of Contents

1. [동기](#1-동기)
2. [기능 요약](#2-기능-요약)
3. [Before / After: stream_dma 예제](#3-before--after-stream_dma-예제)
4. [SV Interface 라이브러리](#4-sv-interface-라이브러리)
5. [AXI-Lite Controller 자동 생성](#5-axi-lite-controller-자동-생성)
6. [Wrapper 자동 생성](#6-wrapper-자동-생성)
7. [kernel_spec.yaml 확장](#7-kernel_specyaml-확장)
8. [NPU_3D 적용 예시](#8-npu_3d-적용-예시)
9. [생성 파일 구조 & 빌드 통합](#9-생성-파일-구조--빌드-통합)
10. [구현 단계](#10-구현-단계)

---

## 1. 동기

DUT 설계자는 커널마다 동일한 보일러플레이트를 반복 작성한다:

| 보일러플레이트 | 줄 수 (stream_dma 기준) | 정보 출처 |
|---------------|------------------------|----------|
| AXI-Lite slave Write FSM | ~56줄 | `kernel_spec.yaml` registers |
| AXI-Lite slave Read FSM | ~27줄 | `kernel_spec.yaml` registers |
| AXI4/AXI4-Stream 개별 신호 나열 | ~55줄 | AXI 표준 |

총 **~138줄의 보일러플레이트** vs **~105줄의 실제 로직**. 비율이 절반을 넘는다.

`kernel_spec.yaml`에 이미 레지스터 맵, 프로토콜, 데이터 폭이 모두 선언되어 있으므로 이 정보로 보일러플레이트를 자동 생성할 수 있다.

---

## 2. 기능 요약

```
┌─────────────────────────────────────────────────────────────────┐
│  사용자가 작성하는 것                                              │
│                                                                 │
│  stream_dma_core.sv                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ module stream_dma_core (                                │    │
│  │     input  logic clk, rst_n,                            │    │
│  │     input  logic [31:0] reg_dst_addr_lo, ...            │    │
│  │     output logic        reg_done,                       │    │
│  │     vten_axis_if.slave   s_axis,    ← SV interface    │    │
│  │     vten_aximm_if.master   m_axi      ← SV interface    │    │
│  │ );                                                      │    │
│  │     // 순수 로직만 (~105줄)                               │    │
│  │ endmodule                                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  kernel_spec.yaml (registers + generate_controller: true)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                         vten build
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  vTen이 자동 생성하는 것                                          │
│                                                                 │
│  stream_dma_axilite_ctrl.sv  ← AXI-Lite slave 디코딩 로직       │
│  stream_dma_wrapper.sv       ← controller + core 연결           │
│                                 외부 포트는 flat (BFM 호환)       │
│  tb_top.sv                   ← 기존과 동일 (변경 없음)            │
└─────────────────────────────────────────────────────────────────┘
```

**세 가지 기능:**

1. **SV Interface 라이브러리** (`vten_sv/`): `vten_axis_if`, `vten_aximm_if`, `vten_axilite_if` — 파라미터화된 AXI interface 정의. 사용자 core 모듈의 포트를 깔끔하게 선언.
2. **AXI-Lite Controller 자동 생성**: `kernel_spec.yaml`의 `registers`에서 Write/Read FSM + address decode를 생성.
3. **Wrapper 자동 생성**: controller + core를 연결. 외부 포트는 flat signal (BFM, FPGA 모두 호환).

---

## 3. Before / After: stream_dma 예제

### 3.1 BEFORE — stream_dma.sv (293줄, 전부 수동)

```systemverilog
module stream_dma #(parameter DATA_W = 256, ADDR_W = 64)(
    input  logic clk, rst_n,

    // ── AXI-Lite Slave: 17개 개별 신호 (20줄) ──
    input  logic [15:0]       s_axilite_awaddr,
    input  logic              s_axilite_awvalid,
    output logic              s_axilite_awready,
    input  logic [31:0]       s_axilite_wdata,
    input  logic [3:0]        s_axilite_wstrb,
    input  logic              s_axilite_wvalid,
    output logic              s_axilite_wready,
    output logic [1:0]        s_axilite_bresp,
    output logic              s_axilite_bvalid,
    input  logic              s_axilite_bready,
    input  logic [15:0]       s_axilite_araddr,
    input  logic              s_axilite_arvalid,
    output logic              s_axilite_arready,
    output logic [31:0]       s_axilite_rdata,
    output logic [1:0]        s_axilite_rresp,
    output logic              s_axilite_rvalid,
    input  logic              s_axilite_rready,

    // ── AXI4-Stream Slave: 4개 신호 ──
    input  logic [DATA_W-1:0] s_axis_tdata,
    input  logic              s_axis_tvalid,
    output logic              s_axis_tready,
    input  logic              s_axis_tlast,

    // ── AXI4 Master: 27개 신호 (30줄) ──
    output logic [ADDR_W-1:0] m_axi_awaddr,
    output logic [7:0]        m_axi_awlen,
    // ... 25줄 더 ...
);
    // ── 레지스터 선언 (5줄) ──
    logic [31:0] reg_dst_addr_lo, reg_dst_addr_hi, reg_length;
    logic reg_start, reg_done;

    // ── AXI-Lite Write FSM (56줄) ── ← 보일러플레이트
    always_ff @(posedge clk or negedge rst_n) begin
        // AW handshake, W handshake, address decode case, B response
    end

    // ── AXI-Lite Read FSM (27줄) ── ← 보일러플레이트
    always_ff @(posedge clk or negedge rst_n) begin
        // AR handshake, address decode case, R response
    end

    // ── DMA Engine (105줄) ── ← 실제 로직
    // ...
endmodule
```

### 3.2 AFTER — stream_dma_core.sv (사용자 작성, ~120줄)

```systemverilog
module stream_dma_core #(
    parameter DATA_W = 256,
    parameter ADDR_W = 64
)(
    input  logic clk,
    input  logic rst_n,

    // ── Registers (단순 신호, vTen controller가 연결) ──
    input  logic [31:0] reg_dst_addr_lo,
    input  logic [31:0] reg_dst_addr_hi,
    input  logic [31:0] reg_length,
    input  logic        reg_start,        // pulse (controller가 1-cycle만 high)
    output logic        reg_done,         // read-only status

    // ── AXI4-Stream Slave (SV interface로 깔끔하게) ──
    vten_axis_if.slave  s_axis,

    // ── AXI4 Master (SV interface로 깔끔하게) ──
    vten_aximm_if.master  m_axi
);
    localparam int BYTES_PER_BEAT = DATA_W / 8;

    // AXI4 Read — tie off (unused)
    assign m_axi.araddr  = '0;
    assign m_axi.arlen   = '0;
    assign m_axi.arsize  = '0;
    assign m_axi.arburst = '0;
    assign m_axi.arvalid = 1'b0;
    assign m_axi.rready  = 1'b0;

    // ── DMA Engine (순수 로직만) ──
    typedef enum logic [2:0] {
        DMA_IDLE, DMA_ACCEPT_STREAM, DMA_WRITE_AW,
        DMA_WRITE_W, DMA_WAIT_B, DMA_DONE
    } dma_state_t;

    dma_state_t dma_state;
    logic [ADDR_W-1:0] dma_dst_addr;
    int dma_beats_left, dma_beat_count;
    logic [DATA_W-1:0] dma_data_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_state      <= DMA_IDLE;
            s_axis.tready  <= 1'b0;           // interface 멤버 접근
            m_axi.awvalid  <= 1'b0;
            m_axi.wvalid   <= 1'b0;
            m_axi.bready   <= 1'b0;
            reg_done       <= 1'b0;
            dma_beat_count <= 0;
        end else begin
            case (dma_state)
                DMA_IDLE: begin
                    s_axis.tready <= 1'b0;
                    if (reg_start) begin       // pulse — 1 cycle만 high
                        dma_dst_addr   <= {reg_dst_addr_hi, reg_dst_addr_lo};
                        dma_beats_left <= reg_length;
                        dma_beat_count <= 0;
                        reg_done       <= 1'b0;
                        if (reg_length == 0)
                            dma_state <= DMA_DONE;
                        else begin
                            dma_state     <= DMA_ACCEPT_STREAM;
                            s_axis.tready <= 1'b1;
                        end
                    end
                end

                DMA_ACCEPT_STREAM: begin
                    if (s_axis.tvalid && s_axis.tready) begin
                        dma_data_latch <= s_axis.tdata;
                        s_axis.tready  <= 1'b0;
                        dma_state      <= DMA_WRITE_AW;
                    end
                end

                DMA_WRITE_AW: begin
                    m_axi.awaddr  <= dma_dst_addr + dma_beat_count * BYTES_PER_BEAT;
                    m_axi.awlen   <= 8'd0;
                    m_axi.awsize  <= $clog2(BYTES_PER_BEAT);
                    m_axi.awburst <= 2'b01;
                    m_axi.awvalid <= 1'b1;
                    m_axi.wdata   <= dma_data_latch;
                    m_axi.wstrb   <= {BYTES_PER_BEAT{1'b1}};
                    m_axi.wlast   <= 1'b1;
                    m_axi.wvalid  <= 1'b1;
                    dma_state     <= DMA_WRITE_W;
                end

                DMA_WRITE_W: begin
                    if (m_axi.awready && m_axi.awvalid)
                        m_axi.awvalid <= 1'b0;
                    if (m_axi.wready && m_axi.wvalid)
                        m_axi.wvalid <= 1'b0;
                    if ((!m_axi.awvalid || m_axi.awready) &&
                        (!m_axi.wvalid  || m_axi.wready)) begin
                        m_axi.awvalid <= 1'b0;
                        m_axi.wvalid  <= 1'b0;
                        m_axi.bready  <= 1'b1;
                        dma_state     <= DMA_WAIT_B;
                    end
                end

                DMA_WAIT_B: begin
                    if (m_axi.bvalid && m_axi.bready) begin
                        m_axi.bready   <= 1'b0;
                        dma_beat_count <= dma_beat_count + 1;
                        dma_beats_left <= dma_beats_left - 1;
                        if (dma_beats_left == 1)
                            dma_state <= DMA_DONE;
                        else begin
                            s_axis.tready <= 1'b1;
                            dma_state     <= DMA_ACCEPT_STREAM;
                        end
                    end
                end

                DMA_DONE: begin
                    reg_done  <= 1'b1;
                    dma_state <= DMA_IDLE;
                end
            endcase
        end
    end
endmodule
```

**차이점 요약:**

| | Before | After |
|---|--------|-------|
| 포트 선언 | 48개 개별 신호 (62줄) | 5 레지스터 + 2 interface (12줄) |
| AXI-Lite FSM | 83줄 수동 작성 | 없음 (자동 생성) |
| DMA 로직 | 105줄 | 105줄 (동일, `.` 접근으로 가독성 향상) |
| **총** | **293줄** | **~120줄** |

---

## 4. SV Interface 라이브러리

`vten_sv/`에 고정 라이브러리로 제공. 빌드 시 자동 컴파일에 포함.

### 4.1 vten_axis_if.sv

```systemverilog
interface vten_axis_if #(
    parameter int DATA_W = 256
);
    logic [DATA_W-1:0] tdata;
    logic              tvalid;
    logic              tready;
    logic              tlast;

    modport master (
        output tdata, tvalid, tlast,
        input  tready
    );
    modport slave (
        input  tdata, tvalid, tlast,
        output tready
    );
endinterface
```

### 4.2 vten_aximm_if.sv

```systemverilog
interface vten_aximm_if #(
    parameter int DATA_W = 256,
    parameter int ADDR_W = 64
);
    // AW channel
    logic [ADDR_W-1:0]   awaddr;
    logic [7:0]          awlen;
    logic [2:0]          awsize;
    logic [1:0]          awburst;
    logic                awvalid;
    logic                awready;
    // W channel
    logic [DATA_W-1:0]   wdata;
    logic [DATA_W/8-1:0] wstrb;
    logic                wlast;
    logic                wvalid;
    logic                wready;
    // B channel
    logic [1:0]          bresp;
    logic                bvalid;
    logic                bready;
    // AR channel
    logic [ADDR_W-1:0]   araddr;
    logic [7:0]          arlen;
    logic [2:0]          arsize;
    logic [1:0]          arburst;
    logic                arvalid;
    logic                arready;
    // R channel
    logic [DATA_W-1:0]   rdata;
    logic [1:0]          rresp;
    logic                rlast;
    logic                rvalid;
    logic                rready;

    modport master (
        output awaddr, awlen, awsize, awburst, awvalid, input awready,
        output wdata, wstrb, wlast, wvalid, input wready,
        input  bresp, bvalid, output bready,
        output araddr, arlen, arsize, arburst, arvalid, input arready,
        input  rdata, rresp, rlast, rvalid, output rready
    );
    modport slave (
        input  awaddr, awlen, awsize, awburst, awvalid, output awready,
        input  wdata, wstrb, wlast, wvalid, output wready,
        output bresp, bvalid, input bready,
        input  araddr, arlen, arsize, arburst, arvalid, output arready,
        output rdata, rresp, rlast, rvalid, input rready
    );
endinterface
```

### 4.3 vten_axilite_if.sv

```systemverilog
interface vten_axilite_if #(
    parameter int ADDR_W = 32,
    parameter int DATA_W = 32
);
    // Write address
    logic [ADDR_W-1:0]   awaddr;
    logic                awvalid;
    logic                awready;
    // Write data
    logic [DATA_W-1:0]   wdata;
    logic [DATA_W/8-1:0] wstrb;
    logic                wvalid;
    logic                wready;
    // Write response
    logic [1:0]          bresp;
    logic                bvalid;
    logic                bready;
    // Read address
    logic [ADDR_W-1:0]   araddr;
    logic                arvalid;
    logic                arready;
    // Read data
    logic [DATA_W-1:0]   rdata;
    logic [1:0]          rresp;
    logic                rvalid;
    logic                rready;

    modport master (
        output awaddr, awvalid, input awready,
        output wdata, wstrb, wvalid, input wready,
        input  bresp, bvalid, output bready,
        output araddr, arvalid, input arready,
        input  rdata, rresp, rvalid, output rready
    );
    modport slave (
        input  awaddr, awvalid, output awready,
        input  wdata, wstrb, wvalid, output wready,
        output bresp, bvalid, input bready,
        input  araddr, arvalid, output arready,
        output rdata, rresp, rvalid, input rready
    );
endinterface
```

### 4.4 BFM과의 관계

**BFM은 수정하지 않는다.** BFM은 flat port를 유지하고, wrapper가 interface ↔ flat 변환을 담당:

```
                    SV interface          flat signals
  core module  ◄──────────────────► wrapper ◄──────────► BFM
  (사용자 작성)    vten_axis_if        │              (기존 유지)
                                       │
                                  자동 생성:
                                  assign s_axis_tdata = s_axis_if.tdata;
                                  assign s_axis_if.tready = s_axis_tready;
                                  ...
```

---

## 5. AXI-Lite Controller 자동 생성

### 5.1 생성 조건

`kernel_spec.yaml`의 AXI-Lite 인터페이스에 `generate_controller: true`가 설정된 경우.

### 5.2 생성 모듈 구조

```
<kernel>_axilite_ctrl.sv
├── Module ports
│   ├── clock, reset
│   ├── AXI-Lite slave (flat signals)
│   ├── Register outputs (access: rw → output, pulse → output)
│   └── Register inputs  (access: ro → input)
├── Write FSM
│   ├── AW handshake
│   ├── W handshake
│   ├── Address decode (case statement from register offsets)
│   ├── Pulse register handling (auto-clear next cycle)
│   └── B response
└── Read FSM
    ├── AR handshake
    ├── Address decode (case statement)
    └── R response
```

### 5.3 Register Access Types

| access | Controller 동작 | Core 포트 방향 | 설명 |
|--------|----------------|---------------|------|
| `rw` | Write: latch, Read: readback | `output` to core | 호스트가 쓰고 읽음 |
| `rw` + `pulse: true` | Write: 1-cycle high then clear, Read: 0 | `output` to core | 트리거 레지스터 |
| `ro` | Write: 무시, Read: core 값 | `input` from core | DUT 상태 (done, busy 등) |
| `wo` | Write: latch, Read: 0 | `output` to core | 쓰기 전용 |
| `w1c` | Write-1-to-clear, Read: core 값 | `input` from core, `output` clear pulse | 인터럽트 클리어 패턴 |

### 5.4 생성 예시 (stream_dma)

Jinja2 템플릿 `axilite_ctrl.sv.j2`로부터 생성:

```systemverilog
// Auto-generated by vTen — DO NOT EDIT
module stream_dma_axilite_ctrl #(
    parameter int ADDR_W = 16,
    parameter int DATA_W = 32
)(
    input  logic clk,
    input  logic rst_n,

    // AXI-Lite slave port
    input  logic [ADDR_W-1:0]   s_awaddr,
    input  logic                s_awvalid,
    output logic                s_awready,
    input  logic [DATA_W-1:0]   s_wdata,
    input  logic [DATA_W/8-1:0] s_wstrb,
    input  logic                s_wvalid,
    output logic                s_wready,
    output logic [1:0]          s_bresp,
    output logic                s_bvalid,
    input  logic                s_bready,
    input  logic [ADDR_W-1:0]   s_araddr,
    input  logic                s_arvalid,
    output logic                s_arready,
    output logic [DATA_W-1:0]   s_rdata,
    output logic [1:0]          s_rresp,
    output logic                s_rvalid,
    input  logic                s_rready,

    // Register outputs (rw/pulse → core)
    output logic [31:0] reg_dst_addr_lo,
    output logic [31:0] reg_dst_addr_hi,
    output logic [31:0] reg_length,
    output logic        reg_start,           // pulse

    // Register inputs (ro ← core)
    input  logic        reg_done
);

    // ── Write Path ──
    logic        aw_done, w_done;
    logic [ADDR_W-1:0] aw_addr_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_awready     <= 1'b0;
            s_wready      <= 1'b0;
            s_bvalid      <= 1'b0;
            s_bresp       <= 2'b00;
            aw_done       <= 1'b0;
            w_done        <= 1'b0;
            // Register resets
            reg_dst_addr_lo <= 32'h0;      // reset_value: 0x0
            reg_dst_addr_hi <= 32'h0;
            reg_length      <= 32'h0;
            reg_start       <= 1'b0;
        end else begin
            // Pulse registers: auto-clear every cycle
            reg_start <= 1'b0;

            // AW handshake
            if (s_awvalid && !aw_done && !s_bvalid) begin
                s_awready     <= 1'b1;
                aw_addr_latch <= s_awaddr;
                aw_done       <= 1'b1;
            end else
                s_awready <= 1'b0;

            // W handshake
            if (s_wvalid && !w_done && !s_bvalid) begin
                s_wready <= 1'b1;
                w_done   <= 1'b1;
            end else
                s_wready <= 1'b0;

            // Write decode
            if (aw_done && w_done && !s_bvalid) begin
                case (aw_addr_latch)
                    ADDR_W'(16'h0000): reg_dst_addr_lo <= s_wdata[31:0];
                    ADDR_W'(16'h0004): reg_dst_addr_hi <= s_wdata[31:0];
                    ADDR_W'(16'h0008): reg_length      <= s_wdata[31:0];
                    ADDR_W'(16'h000C): reg_start       <= s_wdata[0];
                    // 0x0010: status — ro, write ignored
                    default: ;
                endcase
                s_bvalid <= 1'b1;
                s_bresp  <= 2'b00;
                aw_done  <= 1'b0;
                w_done   <= 1'b0;
            end

            if (s_bvalid && s_bready)
                s_bvalid <= 1'b0;
        end
    end

    // ── Read Path ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_arready <= 1'b0;
            s_rvalid  <= 1'b0;
            s_rdata   <= '0;
            s_rresp   <= 2'b00;
        end else begin
            if (s_arvalid && !s_rvalid) begin
                s_arready <= 1'b1;
                s_rvalid  <= 1'b1;
                s_rresp   <= 2'b00;
                case (s_araddr)
                    ADDR_W'(16'h0000): s_rdata <= reg_dst_addr_lo;
                    ADDR_W'(16'h0004): s_rdata <= reg_dst_addr_hi;
                    ADDR_W'(16'h0008): s_rdata <= reg_length;
                    ADDR_W'(16'h000C): s_rdata <= 32'd0;         // pulse: reads as 0
                    ADDR_W'(16'h0010): s_rdata <= {31'd0, reg_done};
                    default:           s_rdata <= 32'hDEAD_BEEF;
                endcase
            end else
                s_arready <= 1'b0;

            if (s_rvalid && s_rready)
                s_rvalid <= 1'b0;
        end
    end
endmodule
```

---

## 6. Wrapper 자동 생성

### 6.1 역할

Wrapper는 **세 가지 연결**을 수행한다:

1. **AXI-Lite**: 외부 flat port → controller → register 신호 → core
2. **AXI4-Stream**: 외부 flat port ↔ SV interface 인스턴스 ↔ core
3. **AXI4**: 외부 flat port ↔ SV interface 인스턴스 ↔ core

### 6.2 이름 규칙

| | 모듈 이름 | 작성자 |
|---|----------|--------|
| Core | `<kernel>_core` | 사용자 |
| Controller | `<kernel>_axilite_ctrl` | vTen 생성 |
| Wrapper | `<kernel>` (= 원래 DUT 이름) | vTen 생성 |

Wrapper 모듈명이 원래 DUT명과 동일 → **tb_top.sv 변경 불필요**.

### 6.3 생성 예시 (stream_dma)

```systemverilog
// Auto-generated by vTen — DO NOT EDIT
module stream_dma #(
    parameter DATA_W = 256,
    parameter ADDR_W = 64
)(
    input  logic clk,
    input  logic rst_n,

    // ── AXI-Lite Slave (flat, BFM 호환) ──
    input  logic [15:0]          s_axilite_awaddr,
    input  logic                 s_axilite_awvalid,
    output logic                 s_axilite_awready,
    input  logic [31:0]          s_axilite_wdata,
    input  logic [3:0]           s_axilite_wstrb,
    input  logic                 s_axilite_wvalid,
    output logic                 s_axilite_wready,
    output logic [1:0]           s_axilite_bresp,
    output logic                 s_axilite_bvalid,
    input  logic                 s_axilite_bready,
    input  logic [15:0]          s_axilite_araddr,
    input  logic                 s_axilite_arvalid,
    output logic                 s_axilite_arready,
    output logic [31:0]          s_axilite_rdata,
    output logic [1:0]           s_axilite_rresp,
    output logic                 s_axilite_rvalid,
    input  logic                 s_axilite_rready,

    // ── AXI4-Stream Slave (flat, BFM 호환) ──
    input  logic [DATA_W-1:0]    s_axis_tdata,
    input  logic                 s_axis_tvalid,
    output logic                 s_axis_tready,
    input  logic                 s_axis_tlast,

    // ── AXI4 Master (flat, BFM 호환) ──
    output logic [ADDR_W-1:0]    m_axi_awaddr,
    output logic [7:0]           m_axi_awlen,
    output logic [2:0]           m_axi_awsize,
    output logic [1:0]           m_axi_awburst,
    output logic                 m_axi_awvalid,
    input  logic                 m_axi_awready,
    output logic [DATA_W-1:0]    m_axi_wdata,
    output logic [DATA_W/8-1:0]  m_axi_wstrb,
    output logic                 m_axi_wlast,
    output logic                 m_axi_wvalid,
    input  logic                 m_axi_wready,
    input  logic [1:0]           m_axi_bresp,
    input  logic                 m_axi_bvalid,
    output logic                 m_axi_bready,
    output logic [ADDR_W-1:0]    m_axi_araddr,
    output logic [7:0]           m_axi_arlen,
    output logic [2:0]           m_axi_arsize,
    output logic [1:0]           m_axi_arburst,
    output logic                 m_axi_arvalid,
    input  logic                 m_axi_arready,
    input  logic [DATA_W-1:0]    m_axi_rdata,
    input  logic [1:0]           m_axi_rresp,
    input  logic                 m_axi_rlast,
    input  logic                 m_axi_rvalid,
    output logic                 m_axi_rready
);

    // ════════════════════════════════════════════════
    // Register wires (controller ↔ core)
    // ════════════════════════════════════════════════
    logic [31:0] reg_dst_addr_lo;
    logic [31:0] reg_dst_addr_hi;
    logic [31:0] reg_length;
    logic        reg_start;
    logic        reg_done;

    // ════════════════════════════════════════════════
    // SV Interface instances
    // ════════════════════════════════════════════════
    vten_axis_if #(.DATA_W(DATA_W)) s_axis_if();
    vten_aximm_if  #(.DATA_W(DATA_W), .ADDR_W(ADDR_W)) m_axi_if();

    // ════════════════════════════════════════════════
    // AXI4-Stream: flat ↔ interface
    // ════════════════════════════════════════════════
    assign s_axis_if.tdata  = s_axis_tdata;
    assign s_axis_if.tvalid = s_axis_tvalid;
    assign s_axis_tready    = s_axis_if.tready;
    assign s_axis_if.tlast  = s_axis_tlast;

    // ════════════════════════════════════════════════
    // AXI4 Master: interface ↔ flat
    // ════════════════════════════════════════════════
    // AW
    assign m_axi_awaddr  = m_axi_if.awaddr;
    assign m_axi_awlen   = m_axi_if.awlen;
    assign m_axi_awsize  = m_axi_if.awsize;
    assign m_axi_awburst = m_axi_if.awburst;
    assign m_axi_awvalid = m_axi_if.awvalid;
    assign m_axi_if.awready = m_axi_awready;
    // W
    assign m_axi_wdata   = m_axi_if.wdata;
    assign m_axi_wstrb   = m_axi_if.wstrb;
    assign m_axi_wlast   = m_axi_if.wlast;
    assign m_axi_wvalid  = m_axi_if.wvalid;
    assign m_axi_if.wready = m_axi_wready;
    // B
    assign m_axi_if.bresp  = m_axi_bresp;
    assign m_axi_if.bvalid = m_axi_bvalid;
    assign m_axi_bready    = m_axi_if.bready;
    // AR
    assign m_axi_araddr  = m_axi_if.araddr;
    assign m_axi_arlen   = m_axi_if.arlen;
    assign m_axi_arsize  = m_axi_if.arsize;
    assign m_axi_arburst = m_axi_if.arburst;
    assign m_axi_arvalid = m_axi_if.arvalid;
    assign m_axi_if.arready = m_axi_arready;
    // R
    assign m_axi_if.rdata  = m_axi_rdata;
    assign m_axi_if.rresp  = m_axi_rresp;
    assign m_axi_if.rlast  = m_axi_rlast;
    assign m_axi_if.rvalid = m_axi_rvalid;
    assign m_axi_rready    = m_axi_if.rready;

    // ════════════════════════════════════════════════
    // AXI-Lite Controller
    // ════════════════════════════════════════════════
    stream_dma_axilite_ctrl #(
        .ADDR_W(16), .DATA_W(32)
    ) u_axilite_ctrl (
        .clk(clk), .rst_n(rst_n),
        .s_awaddr  (s_axilite_awaddr),
        .s_awvalid (s_axilite_awvalid),
        .s_awready (s_axilite_awready),
        .s_wdata   (s_axilite_wdata),
        .s_wstrb   (s_axilite_wstrb),
        .s_wvalid  (s_axilite_wvalid),
        .s_wready  (s_axilite_wready),
        .s_bresp   (s_axilite_bresp),
        .s_bvalid  (s_axilite_bvalid),
        .s_bready  (s_axilite_bready),
        .s_araddr  (s_axilite_araddr),
        .s_arvalid (s_axilite_arvalid),
        .s_arready (s_axilite_arready),
        .s_rdata   (s_axilite_rdata),
        .s_rresp   (s_axilite_rresp),
        .s_rvalid  (s_axilite_rvalid),
        .s_rready  (s_axilite_rready),
        // Registers
        .reg_dst_addr_lo (reg_dst_addr_lo),
        .reg_dst_addr_hi (reg_dst_addr_hi),
        .reg_length      (reg_length),
        .reg_start       (reg_start),
        .reg_done        (reg_done)
    );

    // ════════════════════════════════════════════════
    // Core
    // ════════════════════════════════════════════════
    stream_dma_core #(
        .DATA_W(DATA_W), .ADDR_W(ADDR_W)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        // Registers
        .reg_dst_addr_lo (reg_dst_addr_lo),
        .reg_dst_addr_hi (reg_dst_addr_hi),
        .reg_length      (reg_length),
        .reg_start       (reg_start),
        .reg_done        (reg_done),
        // AXI4-Stream (via interface)
        .s_axis          (s_axis_if),
        // AXI4 (via interface)
        .m_axi           (m_axi_if)
    );

endmodule
```

### 6.4 호환성

| 대상 | 동작 |
|------|------|
| **xsim (시뮬레이션)** | tb_top.sv → wrapper (flat) → controller + core. BFM은 flat port로 wrapper에 연결. |
| **Vivado Synthesis** | wrapper는 순수 structural, controller는 `always_ff` + `case` — 완전 합성 가능. |
| **Vitis IP Packaging** | wrapper의 flat port가 AXI 표준 준수 → IP Integrator에서 자동 인식. |

---

## 7. kernel_spec.yaml 확장

### 7.1 새 필드

기존 스키마에 **4개 optional 필드** 추가:

```yaml
interfaces:
  ctrl:
    rtl_port: s_axilite
    protocol: axi4_lite
    addr_width: 16
    data_width: 32                   # NEW for axilite (기본 32)
    generate_controller: true        # NEW: 자동 생성 트리거 (기본 false)
    registers:
      - name: dst_addr_lo
        offset: 0x00
        access: rw                   # NEW: rw|ro|wo|w1c (기본 rw)
        reset_value: 0               # NEW: 리셋 초기값 (기본 0)
        auto_bind: { tensor: data_out, value: address, bits: "31:0" }
      - name: ctrl
        offset: 0x0C
        access: rw
        pulse: true                  # NEW: 1-cycle pulse (기본 false)
        fields: { start: "0:0" }
      - name: status
        offset: 0x10
        access: ro
        fields: { done: "0:0" }
```

| 새 필드 | 위치 | 타입 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `generate_controller` | InterfaceSpec (axi4_lite) | bool | `false` | 자동 생성 활성화 |
| `access` | RegisterSpec | string | `"rw"` | 접근 유형 |
| `pulse` | RegisterSpec | bool | `false` | `access: rw` 시 1-cycle pulse |
| `reset_value` | RegisterSpec | int | `0` | 리셋 시 초기값 |

### 7.2 하위 호환성

모든 새 필드에 기본값이 있으므로 기존 `kernel_spec.yaml`은 변경 없이 동작.
`generate_controller: false` (기본값)이면 기존 흐름 그대로 — wrapper/controller 미생성.

### 7.3 stream_dma 전체 kernel_spec.yaml

```yaml
kernel: stream_dma
rtl_top: rtl/stream_dma_core.sv

parameters:
  DATA_W: 256
  ADDR_W: 64

memory_regions:
  ddr:
    base: 0x0000_0000
    size: 0x1_0000_0000
    alignment: 4096

interfaces:
  ctrl:
    rtl_port: s_axilite
    protocol: axi4_lite
    addr_width: 16
    data_width: 32
    generate_controller: true
    registers:
      - name: dst_addr_lo
        offset: 0x00
        access: rw
        auto_bind: { tensor: data_out, value: address, bits: "31:0" }
      - name: dst_addr_hi
        offset: 0x04
        access: rw
        auto_bind: { tensor: data_out, value: address, bits: "63:32" }
      - name: length
        offset: 0x08
        access: rw
        auto_bind: { tensor: data_out, value: size_beats }
      - name: ctrl
        offset: 0x0C
        access: rw
        pulse: true
        fields: { start: "0:0" }
      - name: status
        offset: 0x10
        access: ro
        fields: { done: "0:0" }

  input_stream:
    rtl_port: s_axis
    protocol: axi4_stream
    tensor: data_in
    packing:
      element_width: 8
      elements_per_beat: 32

  dma_port:
    rtl_port: m_axi
    protocol: axi4
    data_width: 256
    addr_width: 64
    memory_region: ddr
    tensor: data_out
    packing:
      element_width: 8
      elements_per_beat: 32
```

---

## 8. NPU_3D 적용 예시

### 8.1 kernel_spec.yaml (발췌)

```yaml
kernel: npu_3d
rtl_top: rtl/NPU_3D_core.sv

interfaces:
  ctrl:
    rtl_port: s_axi_control
    protocol: axi4_lite
    addr_width: 16
    data_width: 32
    generate_controller: true
    registers:
      # ── Control ──
      - name: vsync
        offset: 0x100
        access: rw
        pulse: true
        fields: { vsync: "0:0" }
      - name: layer_done
        offset: 0x104
        access: ro
        fields: { done: "0:0" }

      # ── IFM ──
      - name: ifm_base_lo
        offset: 0x110
        access: rw
        auto_bind: { tensor: ifm, value: address, bits: "31:0" }
      - name: ifm_base_hi
        offset: 0x114
        access: rw
        auto_bind: { tensor: ifm, value: address, bits: "63:32" }

      # ── OFM ──
      - name: ofm_base_lo
        offset: 0x120
        access: rw
        auto_bind: { tensor: ofm, value: address, bits: "31:0" }
      - name: ofm_base_hi
        offset: 0x124
        access: rw
        auto_bind: { tensor: ofm, value: address, bits: "63:32" }

      # ── Layer config ──
      - name: in_ch
        offset: 0x200
        access: rw
        auto_bind: { param: "${IN_CH}" }
      - name: out_ch
        offset: 0x204
        access: rw
        auto_bind: { param: "${OUT_CH}" }
      # ... 수십 개 더 ...

  ddr0:
    rtl_port: m_axi_ddr0
    protocol: axi4
    data_width: 256
    addr_width: 64
    memory_region: ddr
    tensors: [ifm, ofm]
    packing: { element_width: 8, elements_per_beat: 32 }

  ddr1:
    rtl_port: m_axi_ddr1
    protocol: axi4
    data_width: 256
    addr_width: 64
    memory_region: ddr
    tensor: weight
    packing: { element_width: 8, elements_per_beat: 32 }
```

### 8.2 NPU_3D_core.sv (사용자 작성)

```systemverilog
module NPU_3D_core (
    input  logic ap_clk,
    input  logic ap_aresetn,

    // Registers — 단순 신호만
    input  logic        reg_vsync,       // pulse
    output logic        reg_layer_done,
    input  logic [31:0] reg_ifm_base_lo,
    input  logic [31:0] reg_ifm_base_hi,
    input  logic [31:0] reg_ofm_base_lo,
    input  logic [31:0] reg_ofm_base_hi,
    input  logic [31:0] reg_in_ch,
    input  logic [31:0] reg_out_ch,
    // ... 더 많은 레지스터 ...

    // DDR ports — SV interface
    vten_aximm_if.master  m_axi_ddr0,
    vten_aximm_if.master  m_axi_ddr1
);
    // NPU 핵심 로직만 — AXI-Lite 보일러플레이트 제거
endmodule
```

기존 NPU_3D_top.sv에서 AXI-Lite slave 로직만 수백 줄이었을 것. 이 전부가 제거됨.

---

## 9. 생성 파일 구조 & 빌드 통합

### 9.1 파일 생성 위치

```
kernels/stream_dma/build/generated/
├── tb_top.sv                          # 기존 (testbench)
├── stream_dma_axilite_ctrl.sv         # NEW
└── stream_dma_wrapper.sv              # NEW (모듈명: stream_dma)
```

### 9.2 빌드 파이프라인 변경

기존 5-stage pipeline에서 **Stage 3 (codegen)**만 확장:

```
Stage 1: project_setup  → Vivado 프로젝트 생성
Stage 2: dpi_c          → SHM 브릿지 빌드
Stage 3: codegen         → tb_top.sv 생성
                         → axilite_ctrl.sv 생성 (NEW, if generate_controller)
                         → wrapper.sv 생성 (NEW, if generate_controller)
Stage 4: compile_order   → resolve_order.tcl (generated/ 포함)
Stage 5: compile         → xvlog + xelab
```

### 9.3 컴파일 순서

vten_sv/ interface 파일이 core보다 먼저 컴파일되어야 함:

```
1. vten_axis_if.sv, vten_aximm_if.sv, vten_axilite_if.sv   (interface 정의)
2. <kernel>_axilite_ctrl.sv                                   (controller)
3. <kernel>_core.sv                                           (사용자 core — interface 사용)
4. <kernel>_wrapper.sv                                        (wrapper — 모두 인스턴스화)
5. tb_top.sv                                                  (testbench — wrapper 인스턴스화)
```

`resolve_order.tcl`이 Vivado의 의존성 분석을 사용하므로 자동 해결되지만,
`project_setup.tcl.j2`에서 vten_sv/ interface 파일을 소스로 추가해야 함.

---

## 10. 구현 단계

### Phase A: SV Interface 라이브러리

1. `vten_sv/vten_axis_if.sv` 작성
2. `vten_sv/vten_aximm_if.sv` 작성
3. `vten_sv/vten_axilite_if.sv` 작성
4. xvlog 구문 검사 통과 확인

### Phase B: Data Model & Parser

1. `RegisterSpec`에 `access`, `pulse`, `reset_value` 필드 추가
2. `InterfaceSpec`에 `generate_controller`, `data_width` (axilite용) 필드 추가
3. Parser에서 새 필드 파싱 + 기본값 처리
4. 검증: `access` 값 유효성, `pulse: true`는 `access: rw`에만 허용
5. 단위 테스트

### Phase C: AXI-Lite Controller 템플릿

1. `templates/axilite_ctrl.sv.j2` 작성
2. Jinja2 context 스키마 정의
3. stream_dma 기준 생성 결과 검증 (xvlog 통과)
4. pulse, ro, rw, w1c access 타입별 테스트

### Phase D: Wrapper 템플릿

1. `templates/wrapper.sv.j2` 작성
2. Interface ↔ flat 변환 로직 생성
3. Controller + Core 인스턴스 연결
4. 파라미터 포워딩
5. stream_dma 기준 생성 결과 검증

### Phase E: Codegen 통합

1. `sv_generator.py`에 `_generate_axilite_controller()`, `_generate_wrapper()` 추가
2. `generate()` 메서드에서 `generate_controller: true` 인터페이스 감지 → 추가 파일 생성
3. `BuildContext`에 생성 파일 목록 추가

### Phase F: 빌드 파이프라인

1. `project_setup.tcl.j2` 수정: vten_sv/ interface 파일 소스 추가
2. `cli/build.py` Stage 3에서 controller/wrapper 생성 호출
3. 컴파일 순서에 interface 파일 포함 확인
4. E2E 테스트: stream_dma_core + 자동 생성으로 빌드 → 시뮬레이션 통과

### Phase G: stream_dma 예제 리팩토링

1. `stream_dma.sv` → `stream_dma_core.sv`로 리팩토링 (AXI-Lite 제거, SV interface 사용)
2. `kernel_spec.yaml` 업데이트 (`generate_controller: true`)
3. 기존 E2E 테스트가 그대로 통과하는지 확인
