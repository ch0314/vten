// vten_types.svh — Shared type definitions for vTen SV library
// Reference: docs/architecture.md

`ifndef VTEN_TYPES_SVH
`define VTEN_TYPES_SVH

// ============================================================================
// SHM Constants (§11.1, §11.2)
// ============================================================================
localparam int SHM_MAGIC        = 32'h5654_454E;  // "VTEN" (little-endian)
localparam int SHM_VERSION      = 32'h0000_0003;  // v0.4 protocol
localparam int CONTROL_SIZE     = 256;
localparam int CMD_SLOT_SIZE    = 64;
localparam int STATS_SLOT_SIZE  = 32;
localparam int BUF_DESC_SIZE    = 24;
localparam int CACHE_LINE       = 64;

// ============================================================================
// OpCode (§1.4)
// ============================================================================
typedef enum logic [3:0] {
    OP_LOAD      = 4'd1,
    OP_PUSH      = 4'd2,
    OP_PULL      = 4'd3,
    OP_STORE     = 4'd4,
    OP_WRITE_REG = 4'd5,
    OP_READ_REG  = 4'd6,
    OP_POLL_REG  = 4'd7,
    OP_BARRIER   = 4'd8
} opcode_t;

// ============================================================================
// Protocol (§1.1)
// ============================================================================
typedef enum logic [7:0] {
    PROTO_AXI4S = 8'd1,
    PROTO_AXI4  = 8'd2,
    PROTO_AXI4L = 8'd3
} protocol_t;

// ============================================================================
// Role (§1.2)
// ============================================================================
localparam logic ROLE_MASTER = 1'b0;
localparam logic ROLE_SLAVE  = 1'b1;

// ============================================================================
// CommandStatus (§1.7)
// ============================================================================
typedef enum logic [7:0] {
    CMD_PENDING   = 8'd0,
    CMD_ISSUED    = 8'd1,
    CMD_ACTIVE    = 8'd2,
    CMD_COMMITTED = 8'd3,
    CMD_ERROR     = 8'd4
} cmd_status_t;

// Alias for convenient use
localparam cmd_status_t COMMITTED = CMD_COMMITTED;

// ============================================================================
// host_status values (§11.4)
// ============================================================================
localparam int HOST_IDLE      = 0;
localparam int HOST_CMD_READY = 1;
localparam int HOST_ACK       = 2;
localparam int HOST_SHUTDOWN  = 3;

// ============================================================================
// backend_status values (§11.5)
// ============================================================================
localparam int BACKEND_IDLE    = 0;
localparam int BACKEND_RUNNING = 1;
localparam int BACKEND_DONE    = 2;
localparam int BACKEND_ERROR   = 3;

// ============================================================================
// Backend Error Codes (§11.13)
// ============================================================================
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

// ============================================================================
// Control flags bit positions (§11.6)
// ============================================================================
localparam int FLAG_STATS_ENABLED    = 0;
localparam int FLAG_PROGRESS_ENABLED = 1;
localparam int FLAG_WAVEFORM_DUMP    = 2;
localparam int FLAG_WAVEFORM_ON_FAIL = 3;

// ============================================================================
// Dependency sentinel
// ============================================================================
localparam logic [15:0] DEP_NONE = 16'hFFFF;

// ============================================================================
// bfm_cmd_t — Command dispatched from Scheduler to BFM (§11.14)
// Dependencies are NOT included — Scheduler-only concern.
// ============================================================================
typedef struct packed {
    opcode_t        opcode;         // lower 4 bits of SHM opcode
    logic [15:0]    cmd_id;
    logic [15:0]    interface_id;
    protocol_t      protocol;
    logic           role;           // 0=MASTER, 1=SLAVE
    logic [15:0]    buffer_id;
    logic           probe;
    logic           sync;           // flags[0]
    logic [31:0]    size;           // transfer size (bytes)
    logic [63:0]    phys_addr;
    logic [31:0]    reg_offset;
    logic [31:0]    reg_value;
    logic [31:0]    reg_mask;
    logic [31:0]    reg_expected;
    logic [15:0]    golden_buf_id;
} bfm_cmd_t;

`endif // VTEN_TYPES_SVH
