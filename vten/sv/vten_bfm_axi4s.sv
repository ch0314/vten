// vten_bfm_axi4s.sv — AXI4-Stream BFM (MASTER/SLAVE via parameter)
// Reference: docs/architecture.md
//
// Diagnostics: +VTEN_VERBOSE (xsim) or +define+VTEN_VERBOSE (verilator)

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_bfm_axi4s #(
    parameter int DATA_W = 256,
    parameter     MODE   = "MASTER"
)(
    input  logic clk,
    input  logic rst_n,
    // AXI4-Stream
    output logic [DATA_W-1:0]   m_tdata,
    output logic                m_tvalid,
    input  logic                m_tready,
    output logic                m_tlast,
    input  logic [DATA_W-1:0]   s_tdata,
    input  logic                s_tvalid,
    output logic                s_tready,
    input  logic                s_tlast,
    // Scheduler interface
    vten_bfm_cmd_if.bfm         cmd_if,
    // Global cycle counter (from tb_top)
    input  int                  cycle_count
);
    localparam int BYTES_PER_BEAT = DATA_W / 8;

    bfm_cmd_t cmd_queue[$];
    bfm_cmd_t current_cmd;
    logic      cmd_active;
    int beat_count, expected_beats;
    int issue_cycle, first_active, last_active;
    int active_cycles, stall_cycles, total_beats;

    // Runtime verbose flag
    bit verbose;
    initial begin
`ifdef VTEN_VERBOSE
        verbose = 1;
`else
  `ifndef VERILATOR
        verbose = $test$plusargs("VTEN_VERBOSE");
  `else
        verbose = 0;
  `endif
`endif
    end

    // v0.4.1: idle signal — registered to avoid xsim continuous-assign issue
    // with queue.size() not re-evaluating in assign/always_comb.
    logic idle_r;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            idle_r <= 1'b1;
        else
            idle_r <= !cmd_active && (cmd_queue.size() == 0);
    end
    assign cmd_if.idle = idle_r;

    // Bulk transfer buffer: byte[] gives contiguous 1-byte stride on all
    // simulators (xsim, verilator), enabling memcpy-based DPI-C transfer.
    byte beat_buf [0:BYTES_PER_BEAT-1];

    // Command receive
    always_ff @(posedge clk) begin
        if (cmd_if.cmd_valid) cmd_queue.push_back(cmd_if.cmd_data);
    end

    // Command activation & execution
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_active <= 0;
            cmd_if.done_valid <= 0;
            m_tvalid <= 0;
            s_tready <= 0;
        end else begin
            cmd_if.done_valid <= 0;

            if (!cmd_active && cmd_queue.size() > 0) begin
                current_cmd = cmd_queue.pop_front();
                cmd_active <= 1;
                beat_count <= 0;
                expected_beats <= current_cmd.size / BYTES_PER_BEAT;
                issue_cycle <= cycle_count;
                first_active <= 0;
                last_active <= 0;
                active_cycles <= 0;
                stall_cycles <= 0;
                total_beats <= 0;
                if (verbose)
                    $display("[AXI4S    %t] %s %s iface=%0d cmd#%0d: buf=%0d, %0d bytes (%0d beats)",
                             $time, MODE, current_cmd.opcode.name(),
                             current_cmd.interface_id, current_cmd.cmd_id,
                             current_cmd.buffer_id,
                             current_cmd.size, current_cmd.size / BYTES_PER_BEAT);
            end

            // Periodic stats flush to SHM (every 512 cycles).
            // Placed before execute so finish_command() overwrites if it fires.
            if (cmd_active && cycle_count[8:0] == 9'b0)
                vten_write_cmd_stats(current_cmd.cmd_id, CMD_ISSUED,
                    issue_cycle, 0,
                    (first_active == 0) ? 0 : first_active,
                    last_active, active_cycles, total_beats, stall_cycles);

            if (cmd_active) begin
                if (MODE == "MASTER") execute_master();
                else                  execute_slave();
            end
        end
    end

    // MASTER mode: PUSH (SHM → DUT)
    // NBA-aware: on handshake, drive NEXT beat; on initial, drive beat 0.
    task automatic execute_master();
        if (m_tvalid && m_tready) begin
            // Current beat consumed
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;
            if (beat_count == expected_beats - 1) begin
                // Last beat — done
                m_tvalid <= 1'b0;
                beat_count <= beat_count + 1;
                finish_command();
            end else begin
                // Drive next beat data (beat_count+1) — bulk read
                vten_read_data_bulk(current_cmd.buffer_id,
                    (beat_count + 1) * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    m_tdata[i*8 +: 8] <= beat_buf[i];
                m_tlast <= ((beat_count + 1) == expected_beats - 1);
                beat_count <= beat_count + 1;
            end
        end else if (!m_tvalid) begin
            // Initial: drive first beat — bulk read
            vten_read_data_bulk(current_cmd.buffer_id,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
            for (int i = 0; i < BYTES_PER_BEAT; i++)
                m_tdata[i*8 +: 8] <= beat_buf[i];
            m_tvalid <= 1'b1;
            m_tlast  <= (expected_beats == 1);
        end else begin
            // m_tvalid=1 but not ready → stall
            stall_cycles <= stall_cycles + 1;
        end
    endtask

    // SLAVE mode: PULL (DUT → SHM)
    task automatic execute_slave();
        s_tready <= 1'b1;
        if (s_tvalid && s_tready) begin
            // Bulk write: pack tdata into byte buffer, then single memcpy
            for (int i = 0; i < BYTES_PER_BEAT; i++)
                beat_buf[i] = s_tdata[i*8 +: 8];
            vten_write_data_bulk(current_cmd.buffer_id,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
            beat_count <= beat_count + 1;
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;

            // Probe mode: beat-by-beat golden comparison
            if (current_cmd.probe) begin : probe_blk
                logic [DATA_W-1:0] golden;
                byte golden_buf [0:BYTES_PER_BEAT-1];
                vten_read_golden_bulk(current_cmd.golden_buf_id,
                    beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, golden_buf);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    golden[i*8 +: 8] = golden_buf[i];
                if (s_tdata !== golden) begin
                    vten_log_mismatch(current_cmd.cmd_id, cycle_count, beat_count,
                                      golden[DATA_W-1:DATA_W/2],
                                      golden[DATA_W/2-1:0],
                                      s_tdata[DATA_W-1:DATA_W/2],
                                      s_tdata[DATA_W/2-1:0]);
                    if (vten_read_flags() & 8) begin
                        // GUI mode: $stop for waveform inspection, then continue
                        // Don't signal error — let user resume and observe more mismatches
                        $display("[PROBE MISMATCH] cycle=%0d beat=%0d cmd_id=%0d — $stop for inspection",
                                 cycle_count, beat_count, current_cmd.cmd_id);
`ifndef VERILATOR
                        $stop;
`endif
                    end else begin
                        // Batch mode: early abort — signal error to scheduler
                        s_tready <= 1'b0;
                        cmd_active <= 0;
                        cmd_if.done_valid  <= 1'b1;
                        cmd_if.done_cmd_id <= current_cmd.cmd_id;
                        cmd_if.done_error  <= 1'b1;
                        cmd_if.done_error_code <= 16'd8;  // ERR_PROBE_MISMATCH
                        vten_write_cmd_stats(current_cmd.cmd_id,
                            CMD_ERROR, issue_cycle, cycle_count,
                            (first_active == 0) ? cycle_count : first_active,
                            cycle_count,
                            active_cycles, total_beats, stall_cycles);
                        return;
                    end
                end
            end

            if (beat_count == expected_beats - 1) begin
                s_tready <= 1'b0;
                finish_command();
            end
        end else if (!s_tvalid && s_tready)
            stall_cycles <= stall_cycles + 1;
    endtask

    task automatic finish_command();
        cmd_active <= 0;
        cmd_if.done_valid  <= 1'b1;
        cmd_if.done_cmd_id <= current_cmd.cmd_id;
        cmd_if.done_error  <= 1'b0;
        cmd_if.done_error_code <= 16'd0;
        // finish_command() runs in the SAME cycle as the final beat's
        // handshake, where active_cycles/total_beats are bumped by non-blocking
        // assignment and are therefore not yet visible here. Add the in-flight
        // final beat (+1) so it isn't undercounted — this is why a 1-beat
        // transfer previously reported 0 beats / 0 active cycles.
        vten_write_cmd_stats(current_cmd.cmd_id,
            CMD_COMMITTED, issue_cycle, cycle_count,
            (first_active == 0) ? cycle_count : first_active,
            cycle_count,
            active_cycles + 1, total_beats + 1, stall_cycles);
        if (verbose)
            $display("[AXI4S    %t] %s %s iface=%0d cmd#%0d done: %0d beats, %0d stall cyc, %0d active cyc",
                     $time, MODE, current_cmd.opcode.name(),
                     current_cmd.interface_id, current_cmd.cmd_id,
                     total_beats, stall_cycles, active_cycles);
    endtask
endmodule
