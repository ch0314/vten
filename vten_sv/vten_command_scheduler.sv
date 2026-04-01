// vten_command_scheduler.sv — Dependency-aware command dispatch
// Reference: specs/04_backend_xsim.md §10
//
// Diagnostics: +VTEN_VERBOSE (xsim) or +define+VTEN_VERBOSE (verilator)

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_command_scheduler #(
    parameter int MAX_CMDS   = 256,
    parameter int MAX_BFMS   = 8,
    parameter int MAX_IFACES = 16
)(
    input  logic clk,
    input  logic rst_n,
    // ← Controller: command feed
    input  logic        feed_valid,
    input  bfm_cmd_t    feed_data,
    output logic        feed_ready,
    input  logic        feed_done,     // Batch complete trigger
    // ← Controller: batch lifecycle
    input  logic        batch_init,    // Reset scheduler for new batch
    // → Controller: status report
    output logic        all_committed,
    output logic        all_drained,
    output logic        error_flag,
    output logic [15:0] error_cmd_id,
    output logic [15:0] error_code,
    // BFM interfaces
    vten_bfm_cmd_if.scheduler bfm [MAX_BFMS],
    // Cycle count (global)
    input  int          cycle_count,
    // BFM mapping: interface_id → BFM index. Codegen (tb_top.sv) wires this.
    input  int          iface_to_bfm [0:MAX_IFACES-1]
);

    // ── Command store ──
    bfm_cmd_t cmd_store [0:MAX_CMDS-1];
    int num_loaded;
    int num_commands;
    logic batch_active;

    // ── Dependency store (separated from bfm_cmd_t — Scheduler only) ──
    logic [1:0]  cmd_num_dep        [0:MAX_CMDS-1];
    logic [15:0] cmd_dep            [0:MAX_CMDS-1][0:3];
    logic [1:0]  cmd_num_commit_dep [0:MAX_CMDS-1];
    logic [15:0] cmd_commit_dep     [0:MAX_CMDS-1][0:3];

    // ── Sync chain & Barrier fence (preprocessing result) ──
    logic [15:0] prev_sync_cmd  [0:MAX_CMDS-1];  // DEP_NONE = none
    logic [15:0] barrier_fence  [0:MAX_CMDS-1];  // DEP_NONE = none

    // ── BFM mapping ──
    int cmd_bfm_map [0:MAX_CMDS-1];

    // ── State bitmaps ──
    logic issued    [0:MAX_CMDS-1];
    logic bfm_done  [0:MAX_CMDS-1];
    logic committed [0:MAX_CMDS-1];
    logic ready     [0:MAX_CMDS-1];  // Combinational
    logic stats_enabled;

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

    // ── Per-BFM intermediate signals (xvlog constant-index requirement) ──
    // Output registers (Scheduler → BFM)
    logic     bfm_cmd_valid_r [0:MAX_BFMS-1];
    bfm_cmd_t bfm_cmd_data_r  [0:MAX_BFMS-1];
    // Input wires (BFM → Scheduler)
    logic        bfm_done_valid_w    [0:MAX_BFMS-1];
    logic [15:0] bfm_done_cmd_id_w   [0:MAX_BFMS-1];
    logic        bfm_done_error_w    [0:MAX_BFMS-1];
    logic [15:0] bfm_done_err_code_w [0:MAX_BFMS-1];
    logic        bfm_idle_w          [0:MAX_BFMS-1];

    // Generate: wire interface ports to/from intermediate arrays
    genvar gi;
    generate
        for (gi = 0; gi < MAX_BFMS; gi++) begin : gen_bfm_wire
            assign bfm[gi].cmd_valid       = bfm_cmd_valid_r[gi];
            assign bfm[gi].cmd_data        = bfm_cmd_data_r[gi];
            assign bfm_done_valid_w[gi]    = bfm[gi].done_valid;
            assign bfm_done_cmd_id_w[gi]   = bfm[gi].done_cmd_id;
            assign bfm_done_error_w[gi]    = bfm[gi].done_error;
            assign bfm_done_err_code_w[gi] = bfm[gi].done_error_code;
            assign bfm_idle_w[gi]          = bfm[gi].idle;
        end
    endgenerate

    assign feed_ready = !batch_active && (num_loaded < MAX_CMDS);

    // ════════════════════════════════════════════════════════════════
    // Command receive + batch lifecycle
    // ════════════════════════════════════════════════════════════════
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            num_loaded   <= 0;
            num_commands <= 0;
            batch_active <= 0;
            error_flag   <= 0;
            stats_enabled <= 0;
        end else begin
            // batch_init: force-reset scheduler for new batch (clears stuck state
            // from prior error where batch_active never drained)
            if (batch_init) begin
                if (verbose && batch_active)
                    $display("[SCHED    %t] batch_init: force-clearing stuck batch (error_flag=%0d)",
                             $time, error_flag);
                batch_active <= 0;
                error_flag   <= 0;
                num_loaded   <= 0;
                num_commands <= 0;
            end
            if (feed_valid && feed_ready) begin
                cmd_store[num_loaded] <= feed_data;
                num_loaded <= num_loaded + 1;
            end
            if (feed_done) begin
                num_commands <= num_loaded;
                batch_active <= 1;
                stats_enabled <= (vten_read_flags() & 1) != 0;
                if (verbose)
                    $display("[SCHED    %t] ──── batch start: %0d commands ────",
                             $time, num_loaded);
                preprocess_batch(num_loaded);
            end
            if (batch_active && all_drained) begin
                batch_active <= 0;
                num_loaded   <= 0;
                num_commands <= 0;
            end
        end
    end

    // ════════════════════════════════════════════════════════════════
    // Preprocessing: executed once at batch start
    // ════════════════════════════════════════════════════════════════
    task automatic preprocess_batch(int n);
        // 1. Load dependencies from SHM via DPI-C
        for (int i = 0; i < n; i++) begin
            int nd, ncd;
            int d [0:3], cd [0:3];
            vten_read_command_deps(i, nd, d, ncd, cd);
            cmd_num_dep[i]        <= nd[1:0];
            cmd_num_commit_dep[i] <= ncd[1:0];
            for (int j = 0; j < 4; j++) begin
                cmd_dep[i][j]        <= d[j];
                cmd_commit_dep[i][j] <= cd[j];
            end
        end

        // 2. Bitmap init (LOAD = pre-committed, others = 0)
        for (int i = 0; i < n; i++) begin
            issued[i]    <= 1'b0;
            bfm_done[i]  <= 1'b0;
            committed[i] <= (cmd_store[i].opcode == OP_LOAD) ? 1'b1 : 1'b0;
        end

        // 3. Sync chain & Barrier fence
        build_sync_chain(n);
        build_barrier_fences(n);

        // 4. BFM mapping
        build_bfm_map(n);
    endtask

    task automatic build_sync_chain(int n);
        logic [15:0] last_sync;
        last_sync = DEP_NONE;
        for (int i = 0; i < n; i++) begin
            prev_sync_cmd[i] <= last_sync;
            if (cmd_store[i].sync) last_sync = i[15:0];
        end
    endtask

    task automatic build_barrier_fences(int n);
        logic [15:0] last_barrier;
        last_barrier = DEP_NONE;
        for (int i = 0; i < n; i++) begin
            if (cmd_store[i].opcode == OP_BARRIER) begin
                barrier_fence[i] <= last_barrier;
                last_barrier = i[15:0];
            end else begin
                barrier_fence[i] <= last_barrier;
            end
        end
    endtask

    task automatic build_bfm_map(int n);
        for (int i = 0; i < n; i++) begin
            case (cmd_store[i].opcode)
                OP_LOAD, OP_STORE, OP_BARRIER:
                    cmd_bfm_map[i] <= -1;  // No BFM needed
                default:
                    cmd_bfm_map[i] <= iface_to_bfm[cmd_store[i].interface_id];
            endcase
        end
    endtask

    // ════════════════════════════════════════════════════════════════
    // Ready evaluation (combinational)
    // ════════════════════════════════════════════════════════════════
    always_comb begin
        logic deps_met;
        for (int i = 0; i < MAX_CMDS; i++) ready[i] = 1'b0;

        for (int i = 0; i < num_commands; i++) begin
            if (issued[i] || committed[i]) continue;
            deps_met = 1'b1;

            // Special: BARRIER needs all prior commands committed
            if (cmd_store[i].opcode == OP_BARRIER) begin
                for (int j = 0; j < i; j++)
                    if (!committed[j]) deps_met = 1'b0;
            end else begin
                // 1. Issue dependencies
                for (int d = 0; d < cmd_num_dep[i]; d++)
                    if (!committed[cmd_dep[i][d]]) deps_met = 1'b0;

                // 2. Sync chain
                if (prev_sync_cmd[i] != DEP_NONE)
                    if (!committed[prev_sync_cmd[i]]) deps_met = 1'b0;

                // 3. Barrier fence
                if (barrier_fence[i] != DEP_NONE)
                    if (!committed[barrier_fence[i]]) deps_met = 1'b0;
            end

            // Commit deps do NOT affect readiness
            if (deps_met) ready[i] = 1'b1;
        end
    end

    // ════════════════════════════════════════════════════════════════
    // Dispatch logic
    // ════════════════════════════════════════════════════════════════
    logic bfm_used_this_cycle [0:MAX_BFMS-1];

    always_ff @(posedge clk) begin
        if (batch_active) begin
            for (int b = 0; b < MAX_BFMS; b++) begin
                bfm_cmd_valid_r[b] <= 1'b0;
                bfm_used_this_cycle[b] = 1'b0;
            end

            for (int i = 0; i < num_commands; i++) begin
                if (!ready[i]) continue;
                case (cmd_store[i].opcode)
                    OP_STORE, OP_BARRIER: begin
                        // Internal commit — no BFM dispatch
                        issued[i]    <= 1'b1;
                        bfm_done[i]  <= 1'b1;
                        committed[i] <= 1'b1;
                        if (stats_enabled)
                            vten_write_cmd_status(i, CMD_COMMITTED);
                    end
                    default: begin
                        if (cmd_bfm_map[i] < 0) begin
                            report_error(i, ERR_UNKNOWN_OPCODE);
                        end else if (!bfm_used_this_cycle[cmd_bfm_map[i]]) begin
                            dispatch_to_bfm(cmd_bfm_map[i], cmd_store[i]);
                            issued[i] <= 1'b1;
                            bfm_used_this_cycle[cmd_bfm_map[i]] = 1'b1;
                            if (stats_enabled)
                                vten_write_cmd_status(i, CMD_ISSUED);
                        end
                    end
                endcase
            end
        end
    end

    // Helper task: dispatch command to a specific BFM by index
    // Iterates with constant-guarded writes to intermediate registers
    task automatic dispatch_to_bfm(int bfm_idx, bfm_cmd_t cmd_data);
        for (int k = 0; k < MAX_BFMS; k++) begin
            if (k == bfm_idx) begin
                bfm_cmd_valid_r[k] <= 1'b1;
                bfm_cmd_data_r[k]  <= cmd_data;
            end
        end
    endtask

    // ════════════════════════════════════════════════════════════════
    // Completion collector
    // ════════════════════════════════════════════════════════════════
    // Current cycle done set (local blocking var, bypasses NBA)
    logic cur_done [0:MAX_CMDS-1];
    logic [15:0] cid;
    logic all_commit_deps;

    always_ff @(posedge clk) begin
        if (batch_active) begin
            for (int i = 0; i < MAX_CMDS; i++) cur_done[i] = 1'b0;

            // Collect BFM done signals (via intermediate wires)
            for (int b = 0; b < MAX_BFMS; b++) begin
                if (bfm_done_valid_w[b]) begin
                    cid = bfm_done_cmd_id_w[b];

                    // cmd_id == 0xFFFF: unattributed error (e.g. AXI4 DECERR)
                    if (cid == DEP_NONE) begin
                        report_error(16'hFFFF, bfm_done_err_code_w[b]);
                    end else begin
                        cur_done[cid] = 1'b1;
                        bfm_done[cid] <= 1'b1;
                        if (bfm_done_error_w[b])
                            report_error(cid, bfm_done_err_code_w[b]);
                    end
                end
            end

            // bfm_done → committed promotion (check commit deps)
            for (int i = 0; i < num_commands; i++) begin
                if ((bfm_done[i] || cur_done[i]) && !committed[i]) begin
                    all_commit_deps = 1'b1;
                    for (int d = 0; d < cmd_num_commit_dep[i]; d++) begin
                        if (!committed[cmd_commit_dep[i][d]])
                            all_commit_deps = 1'b0;
                    end
                    if (all_commit_deps) begin
                        committed[i] <= 1'b1;
                    end
                end
            end
        end
    end

    // ── Error report ──
    task automatic report_error(int cmd_id, int code_val);
        if (verbose)
            $display("[SCHED    %t] ERROR: cmd_id=%0d, code=%0d, first=%0b",
                     $time, cmd_id, code_val, !error_flag);
        if (!error_flag) begin  // Only record first error
            error_flag   <= 1'b1;
            error_cmd_id <= cmd_id[15:0];
            error_code   <= code_val[15:0];
        end
        if (stats_enabled && cmd_id < num_commands)
            vten_write_cmd_stats(cmd_id, CMD_ERROR,
                cycle_count, cycle_count, 0, 0, 0, 0, 0);
    endtask

    // ════════════════════════════════════════════════════════════════
    // Termination
    // ════════════════════════════════════════════════════════════════
    logic all_cmds_committed;
    always_comb begin
        all_cmds_committed = 1'b0;
        if (batch_active) begin
            all_cmds_committed = 1'b1;
            for (int i = 0; i < num_commands; i++)
                if (!committed[i]) all_cmds_committed = 1'b0;
        end
    end
    assign all_committed = all_cmds_committed;

    // v0.4.1: all_drained = all_committed AND all BFMs idle
    logic all_bfm_idle;
    always_comb begin
        all_bfm_idle = 1'b1;
        for (int b = 0; b < MAX_BFMS; b++)
            if (!bfm_idle_w[b]) all_bfm_idle = 1'b0;
    end
    assign all_drained = all_committed && all_bfm_idle;

endmodule
