// stream_array_2d.sv — 2x2 AXI4-Stream Passthrough
// Tests 2D array interface: dimensions [2,2] → flat ports din_0_0..din_1_1
// Each of the 4 channels independently passes data from slave to master.

module stream_array_2d #(
    parameter DATA_W = 256
)(
    input  logic clk,
    input  logic rst_n,

    // ── din[0][0] ──
    input  logic [DATA_W-1:0] s_axis_din_0_0_tdata,
    input  logic              s_axis_din_0_0_tvalid,
    output logic              s_axis_din_0_0_tready,
    input  logic              s_axis_din_0_0_tlast,

    output logic [DATA_W-1:0] m_axis_dout_0_0_tdata,
    output logic              m_axis_dout_0_0_tvalid,
    input  logic              m_axis_dout_0_0_tready,
    output logic              m_axis_dout_0_0_tlast,

    // ── din[0][1] ──
    input  logic [DATA_W-1:0] s_axis_din_0_1_tdata,
    input  logic              s_axis_din_0_1_tvalid,
    output logic              s_axis_din_0_1_tready,
    input  logic              s_axis_din_0_1_tlast,

    output logic [DATA_W-1:0] m_axis_dout_0_1_tdata,
    output logic              m_axis_dout_0_1_tvalid,
    input  logic              m_axis_dout_0_1_tready,
    output logic              m_axis_dout_0_1_tlast,

    // ── din[1][0] ──
    input  logic [DATA_W-1:0] s_axis_din_1_0_tdata,
    input  logic              s_axis_din_1_0_tvalid,
    output logic              s_axis_din_1_0_tready,
    input  logic              s_axis_din_1_0_tlast,

    output logic [DATA_W-1:0] m_axis_dout_1_0_tdata,
    output logic              m_axis_dout_1_0_tvalid,
    input  logic              m_axis_dout_1_0_tready,
    output logic              m_axis_dout_1_0_tlast,

    // ── din[1][1] ──
    input  logic [DATA_W-1:0] s_axis_din_1_1_tdata,
    input  logic              s_axis_din_1_1_tvalid,
    output logic              s_axis_din_1_1_tready,
    input  logic              s_axis_din_1_1_tlast,

    output logic [DATA_W-1:0] m_axis_dout_1_1_tdata,
    output logic              m_axis_dout_1_1_tvalid,
    input  logic              m_axis_dout_1_1_tready,
    output logic              m_axis_dout_1_1_tlast
);

    // [0][0]: wire passthrough
    assign m_axis_dout_0_0_tdata  = s_axis_din_0_0_tdata;
    assign m_axis_dout_0_0_tvalid = s_axis_din_0_0_tvalid;
    assign s_axis_din_0_0_tready  = m_axis_dout_0_0_tready;
    assign m_axis_dout_0_0_tlast  = s_axis_din_0_0_tlast;

    // [0][1]: wire passthrough
    assign m_axis_dout_0_1_tdata  = s_axis_din_0_1_tdata;
    assign m_axis_dout_0_1_tvalid = s_axis_din_0_1_tvalid;
    assign s_axis_din_0_1_tready  = m_axis_dout_0_1_tready;
    assign m_axis_dout_0_1_tlast  = s_axis_din_0_1_tlast;

    // [1][0]: wire passthrough
    assign m_axis_dout_1_0_tdata  = s_axis_din_1_0_tdata;
    assign m_axis_dout_1_0_tvalid = s_axis_din_1_0_tvalid;
    assign s_axis_din_1_0_tready  = m_axis_dout_1_0_tready;
    assign m_axis_dout_1_0_tlast  = s_axis_din_1_0_tlast;

    // [1][1]: wire passthrough
    assign m_axis_dout_1_1_tdata  = s_axis_din_1_1_tdata;
    assign m_axis_dout_1_1_tvalid = s_axis_din_1_1_tvalid;
    assign s_axis_din_1_1_tready  = m_axis_dout_1_1_tready;
    assign m_axis_dout_1_1_tlast  = s_axis_din_1_1_tlast;

endmodule
