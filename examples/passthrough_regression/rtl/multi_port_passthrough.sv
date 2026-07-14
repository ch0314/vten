// multi_port_passthrough.sv — 2-port split passthrough DUT
// Two independent AXI4-Stream channels: din_0→dout_0, din_1→dout_1.
// Used to verify split interface data distribution and reassembly.

module multi_port_passthrough #(
    parameter DATA_W = 256
)(
    input  logic clk,
    input  logic rst_n,

    // ── Split input port 0 ──
    input  logic [DATA_W-1:0] din_0_tdata,
    input  logic              din_0_tvalid,
    output logic              din_0_tready,
    input  logic              din_0_tlast,

    // ── Split input port 1 ──
    input  logic [DATA_W-1:0] din_1_tdata,
    input  logic              din_1_tvalid,
    output logic              din_1_tready,
    input  logic              din_1_tlast,

    // ── Split output port 0 ──
    output logic [DATA_W-1:0] dout_0_tdata,
    output logic              dout_0_tvalid,
    input  logic              dout_0_tready,
    output logic              dout_0_tlast,

    // ── Split output port 1 ──
    output logic [DATA_W-1:0] dout_1_tdata,
    output logic              dout_1_tvalid,
    input  logic              dout_1_tready,
    output logic              dout_1_tlast
);

    // Port 0 passthrough
    assign dout_0_tdata  = din_0_tdata;
    assign dout_0_tvalid = din_0_tvalid;
    assign din_0_tready  = dout_0_tready;
    assign dout_0_tlast  = din_0_tlast;

    // Port 1 passthrough
    assign dout_1_tdata  = din_1_tdata;
    assign dout_1_tvalid = din_1_tvalid;
    assign din_1_tready  = dout_1_tready;
    assign dout_1_tlast  = din_1_tlast;

endmodule
