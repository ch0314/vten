// vten_bfm_probe.sv — Passive probe BFM for golden comparison
// Reference: docs/architecture.md
//
// Passive monitor: observes AXI4-Stream signals without driving them.
// Compares each beat against golden data via DPI-C.
// Buffer ID is resolved at runtime via $value$plusargs.
//
// On mismatch:
//   1. Logs via vten_log_mismatch() → stderr + mismatches.jsonl
//   2. Asserts probe_error output → controller signals BACKEND_ERROR
//   3. If FLAG_PAUSE_ON_MISMATCH (0x08): $stop for GUI waveform inspection

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_bfm_axi4s_probe #(
    parameter int DATA_W      = 256,
    parameter int PROBE_INDEX = 0
)(
    input  logic clk,
    input  logic [DATA_W-1:0] tdata,
    input  logic              tvalid,
    output logic              tready,
    input  logic              tlast,
    // Error output — active-high, latched on first mismatch
    output logic              probe_error
);
    localparam int BYTES_PER_BEAT = DATA_W / 8;

    int buffer_id;
    int beat_count;
    int cycle_count;

    initial begin
        beat_count   = 0;
        cycle_count  = 0;
        tready       = 1'b1;  // Always ready (passive monitor)
        probe_error  = 1'b0;

        // Resolve golden buffer ID from plusarg at runtime:
        //   +PROBE_GOLDEN_0=42  → probe index 0 reads from buffer 42
        if (!$value$plusargs($sformatf("PROBE_GOLDEN_%0d=%%d", PROBE_INDEX), buffer_id)) begin
            buffer_id = -1;
            $display("[PROBE %0d] WARNING: no +PROBE_GOLDEN_%0d plusarg — probe disabled",
                     PROBE_INDEX, PROBE_INDEX);
        end else begin
            $display("[PROBE %0d] golden buffer_id=%0d", PROBE_INDEX, buffer_id);
        end
    end

    byte golden_buf [0:BYTES_PER_BEAT-1];

    always @(posedge clk) begin : blk_probe
        logic [DATA_W-1:0] golden_beat;

        cycle_count <= cycle_count + 1;

        if (tvalid && tready && buffer_id >= 0 && !probe_error) begin
            // Bulk golden read — single memcpy instead of per-byte
            vten_read_golden_bulk(buffer_id,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, golden_buf);

            for (int i = 0; i < BYTES_PER_BEAT; i++)
                golden_beat[i*8 +: 8] = golden_buf[i];

            if (tdata !== golden_beat) begin
                // 1. Log mismatch details to stderr + file
                vten_log_mismatch(PROBE_INDEX, cycle_count, beat_count,
                                  golden_beat[DATA_W-1:DATA_W/2],
                                  golden_beat[DATA_W/2-1:0],
                                  tdata[DATA_W-1:DATA_W/2],
                                  tdata[DATA_W/2-1:0]);

                // 2. Signal error → controller will abort simulation
                probe_error <= 1'b1;

                // 3. GUI mode: $stop for interactive waveform inspection
                if (vten_read_flags() & 8) begin
                    $display("[PROBE %0d] $stop at cycle %0d beat %0d — inspect waveform",
                             PROBE_INDEX, cycle_count, beat_count);
`ifndef VERILATOR
                    $stop;
`endif
                end
            end

            beat_count <= beat_count + 1;

            if (tlast) begin
                beat_count <= 0;  // Reset for next transfer
            end
        end
    end
endmodule
