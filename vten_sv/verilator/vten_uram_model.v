// vten_uram_model.v — Behavioral URAM model for Verilator simulation
//
// Drop-in replacement for Xilinx URAM288 IP (uram_rd_back wrapper).
// Port-compatible with Vivado-generated `uram` module.
//
// Behavior:
//   - Port A: write when EN_A && RDB_WR_A, read when EN_A && !RDB_WR_A
//   - Port B: write when EN_B && RDB_WR_B, read when EN_B && !RDB_WR_B
//   - 72-bit data width, 4096-entry depth (ADDR[22:0], lower 12 bits used)
//   - RDACCESS_A/B: 1-cycle delayed read-valid
//   - BWE: byte-write-enable (9 bytes × 8 = 72 bits, parity-interleaved)
//   - Cascade/ECC/Sleep: tied off (not modeled)

`timescale 1 ps / 1 ps

module uram (
    // Cascade outputs (unused — directly driven to 0)
    output [22:0]  CAS_OUT_ADDR_A,
    output [22:0]  CAS_OUT_ADDR_B,
    output [8:0]   CAS_OUT_BWE_A,
    output [8:0]   CAS_OUT_BWE_B,
    output         CAS_OUT_DBITERR_A,
    output         CAS_OUT_DBITERR_B,
    output [71:0]  CAS_OUT_DIN_A,
    output [71:0]  CAS_OUT_DIN_B,
    output [71:0]  CAS_OUT_DOUT_A,
    output [71:0]  CAS_OUT_DOUT_B,
    output         CAS_OUT_EN_A,
    output         CAS_OUT_EN_B,
    output         CAS_OUT_RDACCESS_A,
    output         CAS_OUT_RDACCESS_B,
    output         CAS_OUT_RDB_WR_A,
    output         CAS_OUT_RDB_WR_B,
    output         CAS_OUT_SBITERR_A,
    output         CAS_OUT_SBITERR_B,

    // Data outputs
    output         DBITERR_A,
    output         DBITERR_B,
    output [71:0]  DOUT_A,
    output [71:0]  DOUT_B,
    output         RDACCESS_A,
    output         RDACCESS_B,
    output         SBITERR_A,
    output         SBITERR_B,

    // Port A
    input  [22:0]  ADDR_A,
    input  [8:0]   BWE_A,
    input  [71:0]  DIN_A,
    input          EN_A,
    input          RDB_WR_A,      // 1=write, 0=read
    input          RST_A,

    // Port B
    input  [22:0]  ADDR_B,
    input  [8:0]   BWE_B,
    input  [71:0]  DIN_B,
    input          EN_B,
    input          RDB_WR_B,      // 1=write, 0=read
    input          RST_B,

    // Cascade inputs (ignored)
    input  [22:0]  CAS_IN_ADDR_A,
    input  [22:0]  CAS_IN_ADDR_B,
    input  [8:0]   CAS_IN_BWE_A,
    input  [8:0]   CAS_IN_BWE_B,
    input          CAS_IN_DBITERR_A,
    input          CAS_IN_DBITERR_B,
    input  [71:0]  CAS_IN_DIN_A,
    input  [71:0]  CAS_IN_DIN_B,
    input  [71:0]  CAS_IN_DOUT_A,
    input  [71:0]  CAS_IN_DOUT_B,
    input          CAS_IN_EN_A,
    input          CAS_IN_EN_B,
    input          CAS_IN_RDACCESS_A,
    input          CAS_IN_RDACCESS_B,
    input          CAS_IN_RDB_WR_A,
    input          CAS_IN_RDB_WR_B,
    input          CAS_IN_SBITERR_A,
    input          CAS_IN_SBITERR_B,

    // Clock & misc
    input          CLK,
    input          CCLK,
    input          SLEEP,
    input  [21:0]  URAM_LOCATION,
    input  [31:0]  DI,
    input          VLD,
    input          INJECT_DBITERR_A,
    input          INJECT_DBITERR_B,
    input          INJECT_SBITERR_A,
    input          INJECT_SBITERR_B,
    input          OREG_CE_A,
    input          OREG_CE_B,
    input          OREG_ECC_CE_A,
    input          OREG_ECC_CE_B,

    // Config outputs (unused)
    output         CFGMODE,
    output         CFGBUSY
);

    // ── Parameters ──
    localparam DEPTH = 4096;
    localparam ADDR_BITS = 12;

    // ── Memory array ──
    reg [71:0] mem [0:DEPTH-1];

    // ── Output registers ──
    reg [71:0] dout_a_r;
    reg [71:0] dout_b_r;
    reg        rdaccess_a_r;
    reg        rdaccess_b_r;

    // ── Port A ──
    always @(posedge CLK) begin
        if (RST_A) begin
            dout_a_r     <= 72'd0;
            rdaccess_a_r <= 1'b0;
        end else if (EN_A) begin
            if (RDB_WR_A) begin
                // Write with byte-write-enable
                if (BWE_A[0]) mem[ADDR_A[ADDR_BITS-1:0]][ 7: 0] <= DIN_A[ 7: 0];
                if (BWE_A[1]) mem[ADDR_A[ADDR_BITS-1:0]][15: 8] <= DIN_A[15: 8];
                if (BWE_A[2]) mem[ADDR_A[ADDR_BITS-1:0]][23:16] <= DIN_A[23:16];
                if (BWE_A[3]) mem[ADDR_A[ADDR_BITS-1:0]][31:24] <= DIN_A[31:24];
                if (BWE_A[4]) mem[ADDR_A[ADDR_BITS-1:0]][39:32] <= DIN_A[39:32];
                if (BWE_A[5]) mem[ADDR_A[ADDR_BITS-1:0]][47:40] <= DIN_A[47:40];
                if (BWE_A[6]) mem[ADDR_A[ADDR_BITS-1:0]][55:48] <= DIN_A[55:48];
                if (BWE_A[7]) mem[ADDR_A[ADDR_BITS-1:0]][63:56] <= DIN_A[63:56];
                if (BWE_A[8]) mem[ADDR_A[ADDR_BITS-1:0]][71:64] <= DIN_A[71:64];
                rdaccess_a_r <= 1'b0;
            end else begin
                // Read
                dout_a_r     <= mem[ADDR_A[ADDR_BITS-1:0]];
                rdaccess_a_r <= 1'b1;
            end
        end else begin
            rdaccess_a_r <= 1'b0;
        end
    end

    // ── Port B ──
    always @(posedge CLK) begin
        if (RST_B) begin
            dout_b_r     <= 72'd0;
            rdaccess_b_r <= 1'b0;
        end else if (EN_B) begin
            if (RDB_WR_B) begin
                // Write with byte-write-enable
                if (BWE_B[0]) mem[ADDR_B[ADDR_BITS-1:0]][ 7: 0] <= DIN_B[ 7: 0];
                if (BWE_B[1]) mem[ADDR_B[ADDR_BITS-1:0]][15: 8] <= DIN_B[15: 8];
                if (BWE_B[2]) mem[ADDR_B[ADDR_BITS-1:0]][23:16] <= DIN_B[23:16];
                if (BWE_B[3]) mem[ADDR_B[ADDR_BITS-1:0]][31:24] <= DIN_B[31:24];
                if (BWE_B[4]) mem[ADDR_B[ADDR_BITS-1:0]][39:32] <= DIN_B[39:32];
                if (BWE_B[5]) mem[ADDR_B[ADDR_BITS-1:0]][47:40] <= DIN_B[47:40];
                if (BWE_B[6]) mem[ADDR_B[ADDR_BITS-1:0]][55:48] <= DIN_B[55:48];
                if (BWE_B[7]) mem[ADDR_B[ADDR_BITS-1:0]][63:56] <= DIN_B[63:56];
                if (BWE_B[8]) mem[ADDR_B[ADDR_BITS-1:0]][71:64] <= DIN_B[71:64];
                rdaccess_b_r <= 1'b0;
            end else begin
                // Read
                dout_b_r     <= mem[ADDR_B[ADDR_BITS-1:0]];
                rdaccess_b_r <= 1'b1;
            end
        end else begin
            rdaccess_b_r <= 1'b0;
        end
    end

    // ── Output assignments ──
    assign DOUT_A     = dout_a_r;
    assign DOUT_B     = dout_b_r;
    assign RDACCESS_A = rdaccess_a_r;
    assign RDACCESS_B = rdaccess_b_r;

    // ── Unused outputs ──
    assign DBITERR_A  = 1'b0;
    assign DBITERR_B  = 1'b0;
    assign SBITERR_A  = 1'b0;
    assign SBITERR_B  = 1'b0;
    assign CFGMODE    = 1'b0;
    assign CFGBUSY    = 1'b0;

    // Cascade outputs tied off
    assign CAS_OUT_ADDR_A      = 23'd0;
    assign CAS_OUT_ADDR_B      = 23'd0;
    assign CAS_OUT_BWE_A       = 9'd0;
    assign CAS_OUT_BWE_B       = 9'd0;
    assign CAS_OUT_DBITERR_A   = 1'b0;
    assign CAS_OUT_DBITERR_B   = 1'b0;
    assign CAS_OUT_DIN_A       = 72'd0;
    assign CAS_OUT_DIN_B       = 72'd0;
    assign CAS_OUT_DOUT_A      = 72'd0;
    assign CAS_OUT_DOUT_B      = 72'd0;
    assign CAS_OUT_EN_A        = 1'b0;
    assign CAS_OUT_EN_B        = 1'b0;
    assign CAS_OUT_RDACCESS_A  = 1'b0;
    assign CAS_OUT_RDACCESS_B  = 1'b0;
    assign CAS_OUT_RDB_WR_A    = 1'b0;
    assign CAS_OUT_RDB_WR_B    = 1'b0;
    assign CAS_OUT_SBITERR_A   = 1'b0;
    assign CAS_OUT_SBITERR_B   = 1'b0;

endmodule
