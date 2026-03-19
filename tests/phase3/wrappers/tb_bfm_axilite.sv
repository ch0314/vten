// tb_bfm_axilite.sv — Verilator testbench wrapper for vten_bfm_axilite.

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module tb_bfm_axilite #(
    parameter int ADDR_W = 32,
    parameter int DATA_W = 32,
    parameter int POLL_INTERVAL = 1,
    parameter int POLL_TIMEOUT = 100000
)(
    input  logic clk,
    input  logic rst_n,
    input  int   cycle_count,

    // ── AXI4-Lite Master Write ──
    output logic [ADDR_W-1:0]   m_awaddr,
    output logic                m_awvalid,
    input  logic                m_awready,
    output logic [DATA_W-1:0]   m_wdata,
    output logic [DATA_W/8-1:0] m_wstrb,
    output logic                m_wvalid,
    input  logic                m_wready,
    input  logic [1:0]          m_bresp,
    input  logic                m_bvalid,
    output logic                m_bready,

    // ── AXI4-Lite Master Read ──
    output logic [ADDR_W-1:0]   m_araddr,
    output logic                m_arvalid,
    input  logic                m_arready,
    input  logic [DATA_W-1:0]   m_rdata,
    input  logic [1:0]          m_rresp,
    input  logic                m_rvalid,
    output logic                m_rready,

    // ── Flattened cmd_if ──
    input  logic        cmd_valid,
    input  bfm_cmd_t    cmd_data,
    output logic        done_valid,
    output logic [15:0] done_cmd_id,
    output logic        done_error,
    output logic [15:0] done_error_code,
    output logic        idle
);

    vten_bfm_cmd_if cmd_if();
    assign cmd_if.cmd_valid = cmd_valid;
    assign cmd_if.cmd_data  = cmd_data;
    assign done_valid      = cmd_if.done_valid;
    assign done_cmd_id     = cmd_if.done_cmd_id;
    assign done_error      = cmd_if.done_error;
    assign done_error_code = cmd_if.done_error_code;
    assign idle            = cmd_if.idle;

    vten_bfm_axilite #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .POLL_INTERVAL(POLL_INTERVAL),
        .POLL_TIMEOUT(POLL_TIMEOUT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cycle_count(cycle_count),
        .m_awaddr(m_awaddr), .m_awvalid(m_awvalid), .m_awready(m_awready),
        .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wvalid(m_wvalid), .m_wready(m_wready),
        .m_bresp(m_bresp), .m_bvalid(m_bvalid), .m_bready(m_bready),
        .m_araddr(m_araddr), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rvalid(m_rvalid), .m_rready(m_rready),
        .cmd_if(cmd_if)
    );

endmodule
