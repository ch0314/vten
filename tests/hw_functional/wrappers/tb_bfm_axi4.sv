// tb_bfm_axi4.sv — Verilator testbench wrapper for vten_bfm_axi4.

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module tb_bfm_axi4 #(
    parameter int DATA_W = 256,
    parameter int ADDR_W = 64
)(
    input  logic clk,
    input  logic rst_n,
    input  int   cycle_count,

    // ── AXI4 Slave Read Address ──
    input  logic [ADDR_W-1:0] s_araddr,
    input  logic [7:0]        s_arlen,
    input  logic [2:0]        s_arsize,
    input  logic [1:0]        s_arburst,
    input  logic              s_arvalid,
    output logic              s_arready,

    // ── AXI4 Slave Read Data ──
    output logic [DATA_W-1:0] s_rdata,
    output logic [1:0]        s_rresp,
    output logic              s_rlast,
    output logic              s_rvalid,
    input  logic              s_rready,

    // ── AXI4 Slave Write Address ──
    input  logic [ADDR_W-1:0] s_awaddr,
    input  logic [7:0]        s_awlen,
    input  logic [2:0]        s_awsize,
    input  logic [1:0]        s_awburst,
    input  logic              s_awvalid,
    output logic              s_awready,

    // ── AXI4 Slave Write Data ──
    input  logic [DATA_W-1:0]   s_wdata,
    input  logic [DATA_W/8-1:0] s_wstrb,
    input  logic                s_wlast,
    input  logic                s_wvalid,
    output logic                s_wready,

    // ── AXI4 Slave Write Response ──
    output logic [1:0] s_bresp,
    output logic       s_bvalid,
    input  logic       s_bready,

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

    vten_bfm_axi4 #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cycle_count(cycle_count),
        .s_araddr(s_araddr), .s_arlen(s_arlen), .s_arsize(s_arsize),
        .s_arburst(s_arburst), .s_arvalid(s_arvalid), .s_arready(s_arready),
        .s_rdata(s_rdata), .s_rresp(s_rresp), .s_rlast(s_rlast),
        .s_rvalid(s_rvalid), .s_rready(s_rready),
        .s_awaddr(s_awaddr), .s_awlen(s_awlen), .s_awsize(s_awsize),
        .s_awburst(s_awburst), .s_awvalid(s_awvalid), .s_awready(s_awready),
        .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_wlast(s_wlast),
        .s_wvalid(s_wvalid), .s_wready(s_wready),
        .s_bresp(s_bresp), .s_bvalid(s_bvalid), .s_bready(s_bready),
        .cmd_if(cmd_if)
    );

endmodule
