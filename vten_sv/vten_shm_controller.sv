// vten_shm_controller.sv — Backend state machine (9 states)
// Reference: specs/04_backend_xsim.md §4
//
// Design: Single always_ff block ensures DPI-C calls execute exactly once
// per posedge. All DPI-C calls in S_LOAD_BATCH happen in one cycle;
// S_FEED is pure SV handshake (no DPI-C).

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
    // ← Scheduler: status report
    input  logic        sched_all_committed,
    input  logic        sched_all_drained,
    // ← Scheduler: error report
    input  logic        sched_error,
    input  logic [15:0] sched_error_cmd_id,
    input  logic [15:0] sched_error_code
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

    // ── Single always_ff: state transitions + datapath + DPI-C ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_INIT;
            feed_valid <= 0;
            feed_done  <= 0;
            feed_idx   <= 0;
        end else begin
            // Default: deassert every cycle
            feed_valid <= 0;
            feed_done  <= 0;

            case (state)
                // ── Init: SHM connection ──
                S_INIT: begin
                    if (vten_shm_init(SESSION_ID) == 0)
                        state <= S_WAIT_HOST;
                    else
                        state <= S_ERROR;
                end

                // ── Wait for host signal (timed wait for GUI responsiveness) ──
                S_WAIT_HOST: begin
                    int result;
                    result = vten_wait_host_signal_safe(
                        timeout_ms > 0 ? timeout_ms : 10
                    );
                    if (result == 0) begin  // VTEN_OK
                        case (vten_read_host_status())
                            1: state <= S_LOAD_BATCH;   // CMD_READY
                            3: state <= S_SHUTDOWN;      // SHUTDOWN
                            default: ;                   // retry
                        endcase
                    end
                    // TIMEOUT → stay in S_WAIT_HOST (allow GUI events)
                end

                // ── Batch load: SHM → local cache ──
                S_LOAD_BATCH: begin
                    num_commands <= vten_read_num_commands();
                    timeout_ms   <= vten_read_timeout_ms();
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

                // ── Feed commands to Scheduler sequentially ──
                S_FEED: begin
                    if (feed_idx >= num_commands) begin
                        feed_done <= 1;    // Notify Scheduler "batch complete"
                        state     <= S_EXECUTE;
                    end else if (feed_ready) begin
                        feed_valid <= 1;
                        feed_data  <= cmd_cache[feed_idx];
                        feed_idx   <= feed_idx + 1;
                    end
                end

                // ── Monitor Scheduler execution ──
                S_EXECUTE: begin
                    if (sched_error)
                        state <= S_ERROR;
                    else if (sched_all_committed)
                        state <= S_DRAIN;
                end

                // ── Drain BFM in-flight responses ──
                S_DRAIN: begin
                    if (sched_all_drained)
                        state <= S_COMPLETE;
                end

                // ── Complete → notify host ──
                S_COMPLETE: begin
                    vten_signal_complete();  // backend_status=DONE + sem_post
                    state <= S_WAIT_HOST;
                end

                // ── Error → notify host ──
                S_ERROR: begin
                    vten_signal_error(
                        sched_error_code,
                        $sformatf("[Scheduler] error at cmd_id=%0d",
                                  sched_error_cmd_id)
                    );
                    state <= S_WAIT_HOST;
                end

                // ── Shutdown ──
                S_SHUTDOWN: begin
                    vten_cleanup();
                    $finish;
                end
            endcase
        end
    end
endmodule
