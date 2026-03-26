// scale_core.sv — AXI4-Stream passthrough that multiplies each signed byte by 2 (saturating)
// Sub-module for scale_add composite kernel E2E test

module scale_core #(parameter DATA_W = 256)(
    input  logic clk, input logic rst_n,
    // AXI4-Stream Slave (input)
    input  logic [DATA_W-1:0] s_axis_tdata,
    input  logic              s_axis_tvalid,
    output logic              s_axis_tready,
    input  logic              s_axis_tlast,
    // AXI4-Stream Master (output)
    output logic [DATA_W-1:0] m_axis_tdata,
    output logic              m_axis_tvalid,
    input  logic              m_axis_tready,
    output logic              m_axis_tlast
);

    assign m_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready = m_axis_tready;
    assign m_axis_tlast  = s_axis_tlast;

    // Scale each byte by 2 with signed saturation [-128, 127]
    genvar i;
    generate
        for (i = 0; i < DATA_W / 8; i = i + 1) begin : scale
            wire signed [8:0] scaled = $signed(s_axis_tdata[i*8 +: 8]) * 2;
            assign m_axis_tdata[i*8 +: 8] =
                (scaled > 9'sd127)  ? 8'sd127  :
                (scaled < -9'sd128) ? -8'sd128 :
                scaled[7:0];
        end
    endgenerate

endmodule
