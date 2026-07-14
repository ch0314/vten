// vten_shm_controller.sv — Backend state machine (9 states)
// Reference: docs/architecture.md
//
// Design: Single always_ff block ensures DPI-C calls execute exactly once
// per posedge. All DPI-C calls in S_LOAD_BATCH happen in one cycle;
// S_FEED is pure SV handshake (no DPI-C).
//
// Diagnostics: run with +VTEN_VERBOSE for state transition logs (xsim),
//              or compile with +define+VTEN_VERBOSE (verilator).

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_shm_controller #(
    parameter string SESSION_ID = "default",
    parameter int    MAX_CMDS   = 256
)(
    input  logic clk,
    input  logic rst_n,
    // → Scheduler: command feed (handshake)
    output logic        feed_valid,
    output bfm_cmd_t    feed_data,
    input  logic        feed_ready,
    output logic        feed_done,     // S_FEED complete → Scheduler starts execution
    // → Scheduler: batch lifecycle
    output logic        batch_init,    // Asserted in S_LOAD_BATCH to reset scheduler state
    // ← Scheduler: status report
    input  logic        sched_all_committed,
    input  logic        sched_all_drained,
    // ← Scheduler: error report
    input  logic        sched_error,
    input  logic [15:0] sched_error_cmd_id,
    input  logic [15:0] sched_error_code,
    // ← Probe: mismatch error (active-high, latched)
    input  logic        probe_error
);

    typedef enum logic [3:0] {
        S_INIT, S_WAIT_HOST, S_LOAD_BATCH,
        S_FEED, S_EXECUTE, S_DRAIN,
        S_COMPLETE, S_ERROR, S_SHUTDOWN
    } state_t;

    state_t state;
    int num_commands;
    int feed_idx;
    int timeout_ms;

    // Command local cache (bulk read from SHM)
    bfm_cmd_t cmd_cache [0:MAX_CMDS-1];

    // Runtime session ID: plusarg overrides parameter default.
    string runtime_session_id;
    // Batch timing and numbering
    longint unsigned batch_start_time;
    int batch_number;
    // Runtime verbose: +VTEN_VERBOSE plusarg (xsim) or +define+VTEN_VERBOSE (verilator)
    bit verbose;
    initial begin
        // Global: show time in ns, no decimals, 8-char min width
        $timeformat(-9, 0, " ns", 8);
        if (!$value$plusargs("SESSION_ID=%s", runtime_session_id))
            runtime_session_id = SESSION_ID;
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

    // ── Single always_ff: state transitions + datapath + DPI-C ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_INIT;
            feed_valid   <= 0;
            feed_done    <= 0;
            feed_idx     <= 0;
            batch_init   <= 0;
            batch_number <= 0;
        end else begin
            // Default: deassert every cycle
            feed_valid <= 0;
            feed_done  <= 0;
            batch_init <= 0;

            case (state)
                // ── Init: SHM connection ──
                S_INIT: begin
                    if (vten_shm_init(runtime_session_id) == 0) begin
                        if (verbose)
                            $display("[CTRL     %t] INIT ok → WAIT_HOST (session=%s)", $time, runtime_session_id);
                        state <= S_WAIT_HOST;
                    end else begin
                        if (verbose)
                            $display("[CTRL     %t] INIT failed → ERROR", $time);
                        state <= S_ERROR;
                    end
                end

                // ── Wait for host signal (timed wait for GUI responsiveness) ──
                S_WAIT_HOST: begin
                    int result;
                    result = vten_wait_host_signal_safe(
                        timeout_ms > 0 ? timeout_ms : 10
                    );
                    if (result == 0) begin  // VTEN_OK
                        case (vten_read_host_status())
                            1: begin
                                if (verbose)
                                    $display("[CTRL     %t] CMD_READY → LOAD_BATCH", $time);
                                state <= S_LOAD_BATCH;
                            end
                            3: begin
                                if (verbose)
                                    $display("[CTRL     %t] SHUTDOWN signal → SHUTDOWN", $time);
                                state <= S_SHUTDOWN;
                            end
                            default: ;
                        endcase
                    end
                end

                // ── Batch load: SHM → local cache ──
                S_LOAD_BATCH: begin
                    // Re-mmap if host grew the SHM (dynamic resize)
                    if (vten_shm_remap() != 0) begin
                        if (verbose)
                            $display("[CTRL     %t] LOAD_BATCH: remap failed → ERROR", $time);
                        state <= S_ERROR;
                    end else begin
                        batch_init   <= 1;  // Reset scheduler for new batch
                        num_commands <= vten_read_num_commands();
                        timeout_ms   <= vten_read_timeout_ms();
                        batch_start_time <= $time;
                        batch_number <= batch_number + 1;
                        vten_set_backend_status(BACKEND_RUNNING);

                        // Bulk copy all commands to local cache in one cycle
                        for (int i = 0; i < vten_read_num_commands(); i++) begin
                            int op, iface, proto, rl, bufid, prb, flg, sz;
                            longint pa;
                            int ro, rv, rm, re, gbid;
                            int nd, ncd;
                            int d [0:3], cd [0:3];
                            vten_read_command(i, op, iface, proto, rl,
                                              bufid, prb, flg, sz, pa,
                                              ro, rv, rm, re, gbid,
                                              nd, ncd, d, cd);
                            cmd_cache[i].opcode       <= opcode_t'(op[3:0]);
                            cmd_cache[i].cmd_id       <= i[15:0];
                            cmd_cache[i].interface_id <= iface[15:0];
                            cmd_cache[i].protocol     <= protocol_t'(proto[7:0]);
                            cmd_cache[i].role         <= rl[0];
                            cmd_cache[i].buffer_id    <= bufid[15:0];
                            cmd_cache[i].probe        <= prb[0];
                            cmd_cache[i].sync         <= flg[0];
                            cmd_cache[i].size         <= sz;
                            cmd_cache[i].phys_addr    <= pa;
                            cmd_cache[i].reg_offset   <= ro;
                            cmd_cache[i].reg_value    <= rv;
                            cmd_cache[i].reg_mask     <= rm;
                            cmd_cache[i].reg_expected <= re;
                            cmd_cache[i].golden_buf_id <= gbid[15:0];
                        end

                        feed_idx <= 0;
                        state    <= S_FEED;
                    end
                end

                // ── Feed commands to Scheduler sequentially ──
                S_FEED: begin
                    if (feed_idx >= num_commands) begin
                        feed_done <= 1;
                        state     <= S_EXECUTE;
                    end else if (feed_ready) begin
                        feed_valid <= 1;
                        feed_data  <= cmd_cache[feed_idx];
                        feed_idx   <= feed_idx + 1;
                    end
                end

                // ── Monitor Scheduler execution ──
                S_EXECUTE: begin
                    if (sched_error || probe_error) begin
                        if (verbose) begin
                            if (probe_error && !sched_error)
                                $display("[CTRL     %t] EXECUTE → ERROR (probe mismatch, code=%0d)",
                                         $time, 8);
                            else
                                $display("[CTRL     %t] EXECUTE → ERROR (cmd=%0d, code=%0d)",
                                         $time, sched_error_cmd_id, sched_error_code);
                        end
                        state <= S_ERROR;
                    end else if (sched_all_drained) begin
                        if (verbose)
                            $display("[CTRL     %t] EXECUTE → COMPLETE (all drained)", $time);
                        state <= S_COMPLETE;  // Skip S_DRAIN
                    end else if (sched_all_committed) begin
                        if (verbose)
                            $display("[CTRL     %t] EXECUTE → DRAIN (all committed, waiting BFM idle)", $time);
                        state <= S_DRAIN;
                    end
                end

                // ── Drain BFM in-flight responses ──
                S_DRAIN: begin
                    if (sched_error || probe_error) begin
                        state <= S_ERROR;
                    end else if (sched_all_drained) begin
                        if (verbose)
                            $display("[CTRL     %t] DRAIN → COMPLETE", $time);
                        state <= S_COMPLETE;
                    end
                end

                // ── Complete → notify host ──
                S_COMPLETE: begin
                    vten_signal_complete();  // backend_status=DONE + sem_post
                    if (verbose)
                        $display("[CTRL     %t] ──── batch #%0d done: %0d cmds, %0d ns elapsed ────",
                                 $time, batch_number, num_commands, $time - batch_start_time);
                    state <= S_WAIT_HOST;
                end

                // ── Error → notify host ──
                S_ERROR: begin
                    if (probe_error && !sched_error) begin
                        // Probe mismatch: use ERR_PROBE_MISMATCH (8)
                        if (verbose)
                            $display("[CTRL     %t] ERROR: probe mismatch code=8 → WAIT_HOST", $time);
                        vten_signal_error_with_cmd(
                            8, 0,
                            "[Probe] mismatch detected on internal wire"
                        );
                    end else begin
                        if (verbose)
                            $display("[CTRL     %t] ERROR: cmd=%0d code=%0d → WAIT_HOST",
                                     $time, sched_error_cmd_id, sched_error_code);
                        vten_signal_error_with_cmd(
                            sched_error_code,
                            sched_error_cmd_id,
                            $sformatf("[Scheduler] error at cmd_id=%0d",
                                      sched_error_cmd_id)
                        );
                    end
                    state <= S_WAIT_HOST;
                end

                // ── Shutdown ──
                S_SHUTDOWN: begin
                    if (verbose)
                        $display("[CTRL     %t] SHUTDOWN → $finish", $time);
                    vten_cleanup();
                    $finish;
                end
            endcase
        end
    end
endmodule
