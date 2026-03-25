// passthrough_64.sv — 64-bit bus (8 bytes/beat, 2 × int32)
module passthrough_64 #(parameter DATA_W = 64)(
    input  logic clk, input logic rst_n,
    input  logic [DATA_W-1:0] s_axis_tdata, input logic s_axis_tvalid,
    output logic s_axis_tready, input logic s_axis_tlast,
    output logic [DATA_W-1:0] m_axis_tdata, output logic m_axis_tvalid,
    input  logic m_axis_tready, output logic m_axis_tlast
);
    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready = m_axis_tready;
    assign m_axis_tlast  = s_axis_tlast;
endmodule
