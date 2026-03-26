// stream_array_pt.sv — 4-channel AXI4-Stream Passthrough
// Each channel independently passes data from slave to master.
// Tests array interface tensor distribution (block split across 4 streams).

module stream_array_pt #(
    parameter DATA_W = 256
)(
    input  logic clk,
    input  logic rst_n,

    // ── Channel 0 ──
    input  logic [DATA_W-1:0] s_axis_din_0_tdata,
    input  logic              s_axis_din_0_tvalid,
    output logic              s_axis_din_0_tready,
    input  logic              s_axis_din_0_tlast,

    output logic [DATA_W-1:0] m_axis_dout_0_tdata,
    output logic              m_axis_dout_0_tvalid,
    input  logic              m_axis_dout_0_tready,
    output logic              m_axis_dout_0_tlast,

    // ── Channel 1 ──
    input  logic [DATA_W-1:0] s_axis_din_1_tdata,
    input  logic              s_axis_din_1_tvalid,
    output logic              s_axis_din_1_tready,
    input  logic              s_axis_din_1_tlast,

    output logic [DATA_W-1:0] m_axis_dout_1_tdata,
    output logic              m_axis_dout_1_tvalid,
    input  logic              m_axis_dout_1_tready,
    output logic              m_axis_dout_1_tlast,

    // ── Channel 2 ──
    input  logic [DATA_W-1:0] s_axis_din_2_tdata,
    input  logic              s_axis_din_2_tvalid,
    output logic              s_axis_din_2_tready,
    input  logic              s_axis_din_2_tlast,

    output logic [DATA_W-1:0] m_axis_dout_2_tdata,
    output logic              m_axis_dout_2_tvalid,
    input  logic              m_axis_dout_2_tready,
    output logic              m_axis_dout_2_tlast,

    // ── Channel 3 ──
    input  logic [DATA_W-1:0] s_axis_din_3_tdata,
    input  logic              s_axis_din_3_tvalid,
    output logic              s_axis_din_3_tready,
    input  logic              s_axis_din_3_tlast,

    output logic [DATA_W-1:0] m_axis_dout_3_tdata,
    output logic              m_axis_dout_3_tvalid,
    input  logic              m_axis_dout_3_tready,
    output logic              m_axis_dout_3_tlast
);

    // Channel 0: wire passthrough
    assign m_axis_dout_0_tdata  = s_axis_din_0_tdata;
    assign m_axis_dout_0_tvalid = s_axis_din_0_tvalid;
    assign s_axis_din_0_tready  = m_axis_dout_0_tready;
    assign m_axis_dout_0_tlast  = s_axis_din_0_tlast;

    // Channel 1: wire passthrough
    assign m_axis_dout_1_tdata  = s_axis_din_1_tdata;
    assign m_axis_dout_1_tvalid = s_axis_din_1_tvalid;
    assign s_axis_din_1_tready  = m_axis_dout_1_tready;
    assign m_axis_dout_1_tlast  = s_axis_din_1_tlast;

    // Channel 2: wire passthrough
    assign m_axis_dout_2_tdata  = s_axis_din_2_tdata;
    assign m_axis_dout_2_tvalid = s_axis_din_2_tvalid;
    assign s_axis_din_2_tready  = m_axis_dout_2_tready;
    assign m_axis_dout_2_tlast  = s_axis_din_2_tlast;

    // Channel 3: wire passthrough
    assign m_axis_dout_3_tdata  = s_axis_din_3_tdata;
    assign m_axis_dout_3_tvalid = s_axis_din_3_tvalid;
    assign s_axis_din_3_tready  = m_axis_dout_3_tready;
    assign m_axis_dout_3_tlast  = s_axis_din_3_tlast;

endmodule
