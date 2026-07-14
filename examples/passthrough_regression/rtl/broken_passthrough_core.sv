// broken_passthrough_core.sv
// AXI4-Stream passthrough that intentionally corrupts data:
// XORs every byte of tdata with 0x01 (flips bit 0).
// Used to test probe mismatch detection in vten BFMs.

module broken_passthrough_core #(parameter DATA_W = 256)(
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

    // Pass through handshake and tlast unchanged
    assign m_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready = m_axis_tready;
    assign m_axis_tlast  = s_axis_tlast;

    // Corrupt data: XOR each byte with 0x01
    genvar i;
    generate
        for (i = 0; i < DATA_W / 8; i = i + 1) begin : corrupt
            assign m_axis_tdata[i*8 +: 8] = s_axis_tdata[i*8 +: 8] ^ 8'h01;
        end
    endgenerate

endmodule
