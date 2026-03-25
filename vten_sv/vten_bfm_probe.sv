// vten_bfm_probe.sv — Passive probe BFM for golden comparison
// Reference: specs/05_bfm_library.md §4
//
// Passive monitor: observes AXI4-Stream signals without driving them.
// Compares each beat against golden data via DPI-C.

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_bfm_axi4s_probe #(
    parameter int DATA_W    = 256,
    parameter int BUFFER_ID = 0
)(
    input  logic clk,
    input  logic [DATA_W-1:0] tdata,
    input  logic              tvalid,
    output logic              tready,
    input  logic              tlast
);
    localparam int BYTES_PER_BEAT = DATA_W / 8;

    int beat_count;
    int cycle_count;

    initial begin
        beat_count  = 0;
        cycle_count = 0;
        tready      = 1'b1;  // Always ready (passive monitor)
    end

    byte golden_buf [0:BYTES_PER_BEAT-1];

    always @(posedge clk) begin : blk_probe
        logic [DATA_W-1:0] golden_beat;

        cycle_count <= cycle_count + 1;

        if (tvalid && tready) begin
            // Bulk golden read — single memcpy instead of per-byte
            vten_read_golden_bulk(BUFFER_ID,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, golden_buf);

            for (int i = 0; i < BYTES_PER_BEAT; i++)
                golden_beat[i*8 +: 8] = golden_buf[i];

            if (tdata !== golden_beat) begin
                vten_log_mismatch(cycle_count, beat_count,
                                  golden_beat[DATA_W-1:DATA_W/2],
                                  golden_beat[DATA_W/2-1:0],
                                  tdata[DATA_W-1:DATA_W/2],
                                  tdata[DATA_W/2-1:0]);
            end

            beat_count <= beat_count + 1;

            if (tlast) begin
                beat_count <= 0;  // Reset for next transfer
            end
        end
    end
endmodule
