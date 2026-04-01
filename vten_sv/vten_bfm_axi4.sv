// vten_bfm_axi4.sv — AXI4 Memory-Mapped BFM (slave)
// Reference: specs/05_bfm_library.md §2
//
// BFM is slave — DUT is master.
// Supports simultaneous AR/AW processing with fully independent
// Read Path and Write Path always_ff blocks.
//
// Diagnostics: +VTEN_VERBOSE (xsim) or +define+VTEN_VERBOSE (verilator)

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_bfm_axi4 #(
    parameter int DATA_W = 256,
    parameter int ADDR_W = 64
)(
    input  logic clk,
    input  logic rst_n,
    // AXI4 Slave (DUT is master)
    input  logic [ADDR_W-1:0]   s_araddr,
    input  logic [7:0]          s_arlen,
    input  logic [2:0]          s_arsize,
    input  logic [1:0]          s_arburst,
    input  logic                s_arvalid,
    output logic                s_arready,
    output logic [DATA_W-1:0]   s_rdata,
    output logic [1:0]          s_rresp,
    output logic                s_rlast,
    output logic                s_rvalid,
    input  logic                s_rready,
    input  logic [ADDR_W-1:0]   s_awaddr,
    input  logic [7:0]          s_awlen,
    input  logic [2:0]          s_awsize,
    input  logic [1:0]          s_awburst,
    input  logic                s_awvalid,
    output logic                s_awready,
    input  logic [DATA_W-1:0]   s_wdata,
    input  logic [DATA_W/8-1:0] s_wstrb,
    input  logic                s_wlast,
    input  logic                s_wvalid,
    output logic                s_wready,
    output logic [1:0]          s_bresp,
    output logic                s_bvalid,
    input  logic                s_bready,
    // Scheduler interface
    vten_bfm_cmd_if.bfm         cmd_if,
    // Global cycle counter
    input  int                  cycle_count
);
    localparam int BYTES_PER_BEAT = DATA_W / 8;

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

    // Bulk transfer buffers: byte[] for cross-simulator memcpy
    byte r_beat_buf [0:BYTES_PER_BEAT-1];
    byte w_beat_buf [0:BYTES_PER_BEAT-1];

    // ── Active Table ──
    typedef struct {
        bfm_cmd_t   cmd;
        logic       active;
        int         transferred_bytes;
        int         expected_bytes;
        int         issue_cycle;
        int         first_active;
        int         last_active;
        int         active_cycles;
        int         stall_cycles;
        int         total_beats;
    } active_entry_t;

    active_entry_t active_table[$];

    // ── Done Queue (v0.4.1) ──
    typedef struct {
        logic [15:0] cmd_id;
        logic        error;
        logic [15:0] error_code;
    } done_event_t;

    done_event_t done_queue[$];

    // ── Read/Write pending queues ──
    typedef struct {
        int          entry_idx;
        logic [ADDR_W-1:0] addr;
        logic [7:0]  len;
        logic [2:0]  size;
        logic [1:0]  burst;
    } pending_t;

    pending_t read_pending[$];
    pending_t write_pending[$];

    typedef struct {
        logic [1:0] resp;
    } b_entry_t;

    b_entry_t b_queue[$];

    pending_t current_read;
    pending_t current_write;
    logic r_active;
    logic w_active;
    int r_beat;
    int w_beat;

    // v0.4.1: idle signal — registered, not continuous assign.
    // xsim doesn't re-evaluate assign/always_comb when queue sizes change,
    // but always_ff evaluates .size() procedurally every posedge clk.
    logic idle_r;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idle_r <= 1'b1;
        end else begin : blk_idle_eval
            logic has_active;
            has_active = 1'b0;
            foreach (active_table[i])
                if (active_table[i].active) has_active = 1'b1;
            idle_r <= !has_active
                   && (read_pending.size() == 0)
                   && (write_pending.size() == 0)
                   && (b_queue.size() == 0)
                   && (done_queue.size() == 0)
                   && !r_active && !w_active;
        end
    end
    assign cmd_if.idle = idle_r;

    // ── Address matching ──
    function automatic int find_entry(logic [ADDR_W-1:0] addr, opcode_t op);
        logic [63:0] base;
        logic [63:0] top;
        foreach (active_table[i]) begin
            if (!active_table[i].active) continue;
            if (active_table[i].cmd.opcode != op) continue;
            base = active_table[i].cmd.phys_addr;
            top  = base + active_table[i].cmd.size;
            if (addr >= base && addr < top) return i;
        end
        return -1;  // DECERR
    endfunction

    // ── Command receive ──
    always_ff @(posedge clk) begin
        if (cmd_if.cmd_valid) begin
            active_entry_t entry;
            entry.cmd              = cmd_if.cmd_data;
            entry.active           = 1'b1;
            entry.transferred_bytes = 0;
            entry.expected_bytes   = cmd_if.cmd_data.size;
            entry.issue_cycle      = cycle_count;
            entry.first_active     = 0;
            entry.last_active      = 0;
            entry.active_cycles    = 0;
            entry.stall_cycles     = 0;
            entry.total_beats      = 0;
            active_table.push_back(entry);
            if (verbose)
                $display("[AXI4     %t] %s iface=%0d cmd#%0d: buf=%0d, %0d bytes, phys=0x%016h",
                         $time, cmd_if.cmd_data.opcode.name(),
                         cmd_if.cmd_data.interface_id, cmd_if.cmd_data.cmd_id,
                         cmd_if.cmd_data.buffer_id,
                         cmd_if.cmd_data.size, cmd_if.cmd_data.phys_addr);
        end
    end

    // ── AR channel: ideal slave, always accept ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_arready <= 1'b1;
        end else begin
            s_arready <= 1'b1;
            if (s_arvalid && s_arready) begin : blk_ar_accept
                int idx;
                pending_t p;
                idx = find_entry(s_araddr, OP_PUSH);
                p.entry_idx = idx;
                p.addr      = s_araddr;
                p.len       = s_arlen;
                p.size      = s_arsize;
                p.burst     = s_arburst;
                read_pending.push_back(p);
            end
        end
    end

    // ── R channel: serve data from SHM ──
    // Data is loaded via NBA → visible to DUT next cycle.
    // On handshake, we immediately load the NEXT beat so back-to-back
    // streaming works at full throughput and rlast aligns with the
    // correct data beat.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_active <= 0;
            s_rvalid <= 0;
            s_rlast  <= 0;
        end else begin
            if (read_pending.size() > 0 && !r_active) begin
                current_read = read_pending.pop_front();
                r_active <= 1;
                r_beat <= 0;
            end

            if (r_active) begin
                if (current_read.entry_idx < 0) begin
                    // DECERR response
                    s_rvalid <= 1;
                    s_rresp  <= 2'b11;
                    s_rlast  <= 1;
                    s_rdata  <= '0;
                    if (s_rvalid && s_rready) begin
                        r_active <= 0;
                        s_rvalid <= 0;
                        done_queue.push_back('{
                            cmd_id:     DEP_NONE,
                            error:      1'b1,
                            error_code: ERR_ADDR_UNMATCH
                        });
                    end
                end else begin : blk_r_serve
                    int idx;
                    int offset;
                    int next_beat;
                    idx = current_read.entry_idx;

                    if (s_rvalid && s_rready) begin
                        // ── Handshake: current beat accepted ──
                        active_table[idx].active_cycles++;
                        active_table[idx].total_beats++;
                        if (active_table[idx].first_active == 0)
                            active_table[idx].first_active = cycle_count;
                        active_table[idx].last_active = cycle_count;

                        if (r_beat == current_read.len) begin
                            // Last beat accepted — done
                            s_rvalid <= 0;
                            s_rlast  <= 0;
                            r_active <= 0;
                            active_table[idx].transferred_bytes +=
                                (current_read.len + 1) * (1 << current_read.size);
                            check_completion(idx);
                        end else begin
                            // Load NEXT beat immediately
                            next_beat = r_beat + 1;
                            r_beat <= next_beat;
                            offset = (current_read.addr - active_table[idx].cmd.phys_addr)
                                     + next_beat * (1 << current_read.size);
                            vten_read_data_bulk(
                                active_table[idx].cmd.buffer_id, offset,
                                BYTES_PER_BEAT, r_beat_buf);
                            for (int i = 0; i < BYTES_PER_BEAT; i++)
                                s_rdata[i*8 +: 8] <= r_beat_buf[i];
                            s_rvalid <= 1;
                            s_rresp  <= 2'b00;
                            s_rlast  <= (next_beat == current_read.len);
                        end
                    end else if (!s_rvalid) begin
                        // ── First presentation: load beat 0 ──
                        offset = (current_read.addr - active_table[idx].cmd.phys_addr)
                                 + r_beat * (1 << current_read.size);
                        vten_read_data_bulk(
                            active_table[idx].cmd.buffer_id, offset,
                            BYTES_PER_BEAT, r_beat_buf);
                        for (int i = 0; i < BYTES_PER_BEAT; i++)
                            s_rdata[i*8 +: 8] <= r_beat_buf[i];
                        s_rvalid <= 1;
                        s_rresp  <= 2'b00;
                        s_rlast  <= (r_beat == current_read.len);
                    end
                    // else: rvalid=1, rready=0 → hold current data (regs retain)
                end
            end
        end
    end

    // ── AW channel: ideal slave ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_awready <= 1'b1;
        end else begin
            s_awready <= 1'b1;
            if (s_awvalid && s_awready) begin : blk_aw_accept
                int idx;
                pending_t p;
                idx = find_entry(s_awaddr, OP_PULL);
                p.entry_idx = idx;
                p.addr      = s_awaddr;
                p.len       = s_awlen;
                p.size      = s_awsize;
                p.burst     = s_awburst;
                write_pending.push_back(p);
            end
        end
    end

    // ── W channel: capture data to SHM ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_active <= 0;
            s_wready <= 0;
        end else begin
            if (write_pending.size() > 0 && !w_active) begin
                current_write = write_pending.pop_front();
                w_active <= 1;
                w_beat <= 0;
            end

            s_wready <= w_active;

            if (w_active && s_wvalid && s_wready) begin : blk_w_data
                int idx;
                int offset;
                int transfer_size;
                logic [DATA_W-1:0] golden;

                if (current_write.entry_idx >= 0) begin
                    idx = current_write.entry_idx;
                    offset = (current_write.addr - active_table[idx].cmd.phys_addr)
                             + w_beat * (1 << current_write.size);

                    // WSTRB handling: fast path (all strobes) vs slow path (partial)
                    transfer_size = 1 << current_write.size;
                    if (s_wstrb == {BYTES_PER_BEAT{1'b1}}) begin
                        // Fast path: all bytes valid → bulk write
                        for (int b = 0; b < transfer_size; b++)
                            w_beat_buf[b] = s_wdata[b*8 +: 8];
                        vten_write_data_bulk(
                            active_table[idx].cmd.buffer_id,
                            offset, transfer_size, w_beat_buf);
                    end else begin
                        // Slow path: partial WSTRB → byte-by-byte
                        for (int b = 0; b < transfer_size; b++) begin
                            if (s_wstrb[b])
                                vten_write_data_byte(
                                    active_table[idx].cmd.buffer_id,
                                    offset + b,
                                    s_wdata[b*8 +: 8]);
                        end
                    end

                    active_table[idx].active_cycles++;
                    active_table[idx].total_beats++;
                    if (active_table[idx].first_active == 0)
                        active_table[idx].first_active = cycle_count;
                    active_table[idx].last_active = cycle_count;

                    // Probe mode: bulk golden comparison
                    if (active_table[idx].cmd.probe) begin
                        byte golden_byte_buf [0:BYTES_PER_BEAT-1];
                        vten_read_golden_bulk(
                            active_table[idx].cmd.golden_buf_id,
                            (active_table[idx].total_beats - 1) * BYTES_PER_BEAT,
                            BYTES_PER_BEAT, golden_byte_buf);
                        for (int i = 0; i < BYTES_PER_BEAT; i++)
                            golden[i*8 +: 8] = golden_byte_buf[i];
                        if (s_wdata !== golden)
                            vten_log_mismatch(active_table[idx].cmd.cmd_id,
                                              cycle_count,
                                              active_table[idx].total_beats - 1,
                                              golden[DATA_W-1:DATA_W/2],
                                              golden[DATA_W/2-1:0],
                                              s_wdata[DATA_W-1:DATA_W/2],
                                              s_wdata[DATA_W/2-1:0]);
                    end
                end

                w_beat <= w_beat + 1;
                if (s_wlast) begin
                    w_active <= 0;
                    if (current_write.entry_idx >= 0) begin
                        idx = current_write.entry_idx;
                        active_table[idx].transferred_bytes +=
                            (current_write.len + 1) * (1 << current_write.size);
                        check_completion(idx);
                    end
                    // B channel response
                    begin : blk_b_resp
                        logic [1:0] resp_val;
                        resp_val = (current_write.entry_idx >= 0) ? 2'b00 : 2'b11;
                        b_queue.push_back('{resp: resp_val});
                    end
                    // DECERR for unmatched write
                    if (current_write.entry_idx < 0) begin
                        done_queue.push_back('{
                            cmd_id:     DEP_NONE,
                            error:      1'b1,
                            error_code: ERR_ADDR_UNMATCH
                        });
                    end
                end
            end
        end
    end

    // ── B channel: write response ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_bvalid <= 0;
        end else begin
            if (b_queue.size() > 0) begin
                s_bvalid <= 1;
                s_bresp  <= b_queue[0].resp;
                if (s_bvalid && s_bready)
                    void'(b_queue.pop_front());
            end else begin
                s_bvalid <= 0;
            end
        end
    end

    // ── Completion tracking ──
    task automatic check_completion(int idx);
        if (active_table[idx].transferred_bytes >= active_table[idx].expected_bytes) begin
            // Extract locals to avoid xsim hierarchical reference limitations
            automatic logic [15:0] cid         = active_table[idx].cmd.cmd_id;
            automatic int          iss_cycle   = active_table[idx].issue_cycle;
            automatic int          first_act   = active_table[idx].first_active;
            automatic int          last_act    = active_table[idx].last_active;
            automatic int          act_cycles  = active_table[idx].active_cycles;
            automatic int          tot_beats   = active_table[idx].total_beats;
            automatic int          stl_cycles  = active_table[idx].stall_cycles;
            active_table[idx].active = 0;
            done_queue.push_back('{cmd_id: cid, error: 1'b0, error_code: 16'd0});
            vten_write_cmd_stats(cid, CMD_COMMITTED, iss_cycle, cycle_count,
                first_act, last_act, act_cycles, tot_beats, stl_cycles);
            if (verbose)
                $display("[AXI4     %t] %s iface=%0d cmd#%0d done: %0d beats, %0d stall cyc, %0d active cyc",
                         $time, active_table[idx].cmd.opcode.name(),
                         active_table[idx].cmd.interface_id, cid,
                         tot_beats, stl_cycles, act_cycles);
        end
    endtask

    // ── Done Queue drain: one per cycle to Scheduler ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_if.done_valid <= 0;
        end else begin
            cmd_if.done_valid <= 0;  // default deassert
            if (done_queue.size() > 0) begin : blk_done_drain
                done_event_t ev;
                ev = done_queue.pop_front();
                cmd_if.done_valid      <= 1;
                cmd_if.done_cmd_id     <= ev.cmd_id;
                cmd_if.done_error      <= ev.error;
                cmd_if.done_error_code <= ev.error_code;
            end
        end
    end

endmodule
