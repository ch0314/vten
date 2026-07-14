// tb_bfm_axi4s.sv — Verilator testbench wrapper for vten_bfm_axi4s.
// Flattens the vten_bfm_cmd_if interface into regular ports.

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module tb_bfm_axi4s #(
    parameter int DATA_W = 256,
    parameter string MODE = "MASTER"
)(
    input  logic clk,
    input  logic rst_n,
    input  int   cycle_count,

    // ── AXI4-Stream Master (output from BFM) ──
    output logic [DATA_W-1:0] m_tdata,
    output logic              m_tvalid,
    output logic              m_tlast,
    input  logic              m_tready,

    // ── AXI4-Stream Slave (input to BFM) ──
    input  logic [DATA_W-1:0] s_tdata,
    input  logic              s_tvalid,
    input  logic              s_tlast,
    output logic              s_tready,

    // ── Flattened cmd_if (Scheduler → BFM) ──
    input  logic        cmd_valid,
    input  bfm_cmd_t    cmd_data,
    output logic        done_valid,
    output logic [15:0] done_cmd_id,
    output logic        done_error,
    output logic [15:0] done_error_code,
    output logic        idle
);

    // Internal interface
    vten_bfm_cmd_if cmd_if();

    // Connect flat ports to interface
    assign cmd_if.cmd_valid = cmd_valid;
    assign cmd_if.cmd_data  = cmd_data;
    assign done_valid     = cmd_if.done_valid;
    assign done_cmd_id    = cmd_if.done_cmd_id;
    assign done_error     = cmd_if.done_error;
    assign done_error_code = cmd_if.done_error_code;
    assign idle           = cmd_if.idle;

    // Instantiate BFM
    vten_bfm_axi4s #(
        .DATA_W(DATA_W),
        .MODE(MODE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cycle_count(cycle_count),
        .m_tdata(m_tdata),
        .m_tvalid(m_tvalid),
        .m_tlast(m_tlast),
        .m_tready(m_tready),
        .s_tdata(s_tdata),
        .s_tvalid(s_tvalid),
        .s_tlast(s_tlast),
        .s_tready(s_tready),
        .cmd_if(cmd_if)
    );

endmodule
