// mm_array_lb_core.sv — 4-channel AXI4 Memory-Mapped Loopback
// Each channel independently reads from mem_in[i] and writes to mem_out[i].
// Tests array interface tensor distribution (block split across 4 AXI4 ports).
//
// Register Map (driven by auto-generated AXI-Lite controller):
//   ctrl       — bit[0] = start (pulse)
//   status     — bit[0] = done  (read-only, set when ALL channels complete)
//   src_addr_lo/hi — source base address [63:0]
//   dst_addr_lo/hi — destination base address [63:0]
//   length     — total number of beats (across all 4 channels)
//
// Each channel transfers (length / 4) beats independently.
// All channels start simultaneously and signal done when all complete.

module mm_array_lb_core #(
    parameter DATA_W = 256,
    parameter ADDR_W = 64,
    parameter N_CH   = 4
)(
    input  logic clk,
    input  logic rst_n,

    // ── Registers (driven by auto-generated AXI-Lite controller) ──
    input  logic [31:0] reg_src_addr_lo,
    input  logic [31:0] reg_src_addr_hi,
    input  logic [31:0] reg_dst_addr_lo,
    input  logic [31:0] reg_dst_addr_hi,
    input  logic [31:0] reg_length,
    input  logic        reg_ctrl,          // pulse: 1-cycle start trigger
    output logic        reg_status,        // read-only: done flag

    // ── AXI4 Master Arrays (SV interface) ──
    vten_aximm_if.master mem_in  [N_CH],
    vten_aximm_if.master mem_out [N_CH]
);

    localparam int BYTES_PER_BEAT = DATA_W / 8;

    // ── Edge detection on reg_ctrl ──
    logic reg_ctrl_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) reg_ctrl_d <= 1'b0;
        else        reg_ctrl_d <= reg_ctrl;
    end
    wire start_pulse = reg_ctrl & ~reg_ctrl_d;

    wire [63:0] src_addr = {reg_src_addr_hi, reg_src_addr_lo};
    wire [63:0] dst_addr = {reg_dst_addr_hi, reg_dst_addr_lo};
    wire [31:0] beats_per_ch = reg_length >> 2;  // length / 4

    // ── Per-channel FSM ──
    typedef enum logic [2:0] {
        S_IDLE,
        S_AR,
        S_R,
        S_AW,
        S_W,
        S_B,
        S_DONE
    } state_t;

    state_t            ch_state [N_CH];
    logic [31:0]       ch_beat  [N_CH];
    logic [DATA_W-1:0] ch_buf   [N_CH];
    logic              ch_done  [N_CH];
    logic              ch_start;

    // ── Global start latch ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)       ch_start <= 1'b0;
        else if (start_pulse) ch_start <= 1'b1;
        else              ch_start <= 1'b0;
    end

    // ── All-done detection ──
    logic all_done;
    always_comb begin
        all_done = 1'b1;
        for (int i = 0; i < N_CH; i++)
            all_done = all_done & ch_done[i];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)          reg_status <= 1'b0;
        else if (start_pulse) reg_status <= 1'b0;
        else if (all_done)   reg_status <= 1'b1;
    end

    // ── Generate per-channel FSM ──
    genvar gi;
    generate
    for (gi = 0; gi < N_CH; gi++) begin : gen_ch

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                ch_state[gi] <= S_IDLE;
                ch_beat[gi]  <= '0;
                ch_buf[gi]   <= '0;
                ch_done[gi]  <= 1'b1;  // idle = done

                mem_in[gi].arvalid  <= 1'b0;
                mem_in[gi].rready   <= 1'b0;
                mem_out[gi].awvalid <= 1'b0;
                mem_out[gi].wvalid  <= 1'b0;
                mem_out[gi].bready  <= 1'b0;
            end else begin
                case (ch_state[gi])
                    S_IDLE: begin
                        if (ch_start) begin
                            ch_state[gi] <= S_AR;
                            ch_beat[gi]  <= '0;
                            ch_done[gi]  <= 1'b0;
                            $display("[mm_array_lb ch%0d] START: beats=%0d",
                                     gi, beats_per_ch);
                        end
                    end

                    S_AR: begin
                        mem_in[gi].araddr  <= src_addr +
                            (ch_beat[gi] * BYTES_PER_BEAT);
                        mem_in[gi].arlen   <= 8'd0;
                        mem_in[gi].arsize  <= $clog2(BYTES_PER_BEAT);
                        mem_in[gi].arburst <= 2'b01;
                        mem_in[gi].arvalid <= 1'b1;
                        if (mem_in[gi].arvalid && mem_in[gi].arready) begin
                            mem_in[gi].arvalid <= 1'b0;
                            mem_in[gi].rready  <= 1'b1;
                            ch_state[gi] <= S_R;
                        end
                    end

                    S_R: begin
                        if (mem_in[gi].rvalid && mem_in[gi].rready) begin
                            ch_buf[gi]         <= mem_in[gi].rdata;
                            mem_in[gi].rready  <= 1'b0;
                            ch_state[gi]       <= S_AW;
                        end
                    end

                    S_AW: begin
                        mem_out[gi].awaddr  <= dst_addr +
                            (ch_beat[gi] * BYTES_PER_BEAT);
                        mem_out[gi].awlen   <= 8'd0;
                        mem_out[gi].awsize  <= $clog2(BYTES_PER_BEAT);
                        mem_out[gi].awburst <= 2'b01;
                        mem_out[gi].awvalid <= 1'b1;
                        mem_out[gi].wdata   <= ch_buf[gi];
                        mem_out[gi].wstrb   <= {BYTES_PER_BEAT{1'b1}};
                        mem_out[gi].wlast   <= 1'b1;
                        mem_out[gi].wvalid  <= 1'b1;
                        ch_state[gi]        <= S_W;
                    end

                    S_W: begin
                        if (mem_out[gi].awready && mem_out[gi].awvalid)
                            mem_out[gi].awvalid <= 1'b0;
                        if (mem_out[gi].wready && mem_out[gi].wvalid)
                            mem_out[gi].wvalid <= 1'b0;

                        if ((!mem_out[gi].awvalid || mem_out[gi].awready) &&
                            (!mem_out[gi].wvalid  || mem_out[gi].wready)) begin
                            mem_out[gi].awvalid <= 1'b0;
                            mem_out[gi].wvalid  <= 1'b0;
                            mem_out[gi].bready  <= 1'b1;
                            ch_state[gi]        <= S_B;
                        end
                    end

                    S_B: begin
                        if (mem_out[gi].bvalid && mem_out[gi].bready) begin
                            mem_out[gi].bready <= 1'b0;
                            ch_beat[gi]        <= ch_beat[gi] + 1;
                            if (ch_beat[gi] + 1 >= beats_per_ch) begin
                                ch_state[gi] <= S_DONE;
                            end else begin
                                ch_state[gi] <= S_AR;
                            end
                        end
                    end

                    S_DONE: begin
                        $display("[mm_array_lb ch%0d] DONE: %0d beats",
                                 gi, ch_beat[gi]);
                        ch_done[gi]  <= 1'b1;
                        ch_state[gi] <= S_IDLE;
                    end

                    default: ch_state[gi] <= S_IDLE;
                endcase
            end
        end

        // ── Tie-offs: mem_in write channel (unused) ──
        assign mem_in[gi].awaddr  = '0;
        assign mem_in[gi].awlen   = '0;
        assign mem_in[gi].awsize  = '0;
        assign mem_in[gi].awburst = '0;
        assign mem_in[gi].awvalid = 1'b0;
        assign mem_in[gi].wdata   = '0;
        assign mem_in[gi].wstrb   = '0;
        assign mem_in[gi].wlast   = 1'b0;
        assign mem_in[gi].wvalid  = 1'b0;
        assign mem_in[gi].bready  = 1'b0;

        // ── Tie-offs: mem_out read channel (unused) ──
        assign mem_out[gi].araddr  = '0;
        assign mem_out[gi].arlen   = '0;
        assign mem_out[gi].arsize  = '0;
        assign mem_out[gi].arburst = '0;
        assign mem_out[gi].arvalid = 1'b0;
        assign mem_out[gi].rready  = 1'b0;

    end
    endgenerate

endmodule
