// vten_bfm_axi4.sv — AXI4 Memory-Mapped BFM (slave)
// Reference: specs/05_bfm_library.md §2
//
// BFM is slave — DUT is master.
// Supports simultaneous AR/AW processing with fully independent
// Read Path and Write Path always_ff blocks.

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

    // v0.4.1: idle signal
    assign cmd_if.idle = (active_table.size() == 0)
                      && (read_pending.size() == 0)
                      && (write_pending.size() == 0)
                      && (b_queue.size() == 0)
                      && (done_queue.size() == 0)
                      && !r_active && !w_active;

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
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_active <= 0;
            s_rvalid <= 0;
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
                    bit [7:0] beat_data [0:BYTES_PER_BEAT-1];
                    idx = current_read.entry_idx;
                    offset = (current_read.addr - active_table[idx].cmd.phys_addr)
                             + r_beat * (1 << current_read.size);

                    vten_read_data(active_table[idx].cmd.buffer_id, offset,
                                  1 << current_read.size, beat_data);

                    for (int i = 0; i < BYTES_PER_BEAT; i++)
                        s_rdata[i*8 +: 8] <= beat_data[i];

                    s_rvalid <= 1;
                    s_rresp  <= 2'b00;
                    s_rlast  <= (r_beat == current_read.len);

                    if (s_rvalid && s_rready) begin
                        r_beat <= r_beat + 1;
                        active_table[idx].active_cycles++;
                        active_table[idx].total_beats++;
                        if (active_table[idx].first_active == 0)
                            active_table[idx].first_active = cycle_count;
                        active_table[idx].last_active = cycle_count;

                        if (r_beat == current_read.len) begin
                            s_rvalid <= 0;
                            r_active <= 0;
                            active_table[idx].transferred_bytes +=
                                (current_read.len + 1) * (1 << current_read.size);
                            check_completion(idx);
                        end
                    end
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
                bit [7:0] existing [0:BYTES_PER_BEAT-1];
                bit [7:0] wr_data [0:BYTES_PER_BEAT-1];
                bit [7:0] golden_data [0:BYTES_PER_BEAT-1];
                logic [DATA_W-1:0] golden;

                if (current_write.entry_idx >= 0) begin
                    idx = current_write.entry_idx;
                    offset = (current_write.addr - active_table[idx].cmd.phys_addr)
                             + w_beat * (1 << current_write.size);

                    // WSTRB handling: byte-level selective write
                    transfer_size = 1 << current_write.size;
                    vten_read_data(active_table[idx].cmd.buffer_id, offset,
                                  transfer_size, existing);
                    for (int b = 0; b < DATA_W/8; b++) begin
                        if (s_wstrb[b])
                            existing[b] = s_wdata[b*8 +: 8];
                    end
                    for (int b = 0; b < BYTES_PER_BEAT; b++)
                        wr_data[b] = existing[b];
                    vten_write_data(active_table[idx].cmd.buffer_id, offset,
                                   transfer_size, wr_data);

                    active_table[idx].active_cycles++;
                    active_table[idx].total_beats++;
                    if (active_table[idx].first_active == 0)
                        active_table[idx].first_active = cycle_count;
                    active_table[idx].last_active = cycle_count;

                    // Probe mode
                    if (active_table[idx].cmd.probe) begin
                        vten_read_golden(active_table[idx].cmd.golden_buf_id,
                                         active_table[idx].total_beats - 1,
                                         golden_data);
                        for (int i = 0; i < BYTES_PER_BEAT; i++)
                            golden[i*8 +: 8] = golden_data[i];
                        if (s_wdata !== golden)
                            vten_log_mismatch(cycle_count,
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
                    b_queue.push_back('{
                        resp: (current_write.entry_idx >= 0) ? 2'b00 : 2'b11
                    });
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
            active_table[idx].active = 0;
            done_queue.push_back('{
                cmd_id:     active_table[idx].cmd.cmd_id,
                error:      1'b0,
                error_code: 16'd0
            });
            vten_write_cmd_stats(active_table[idx].cmd.cmd_id,
                CMD_COMMITTED, active_table[idx].issue_cycle, cycle_count,
                active_table[idx].first_active, active_table[idx].last_active,
                active_table[idx].active_cycles, active_table[idx].total_beats,
                active_table[idx].stall_cycles);
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
