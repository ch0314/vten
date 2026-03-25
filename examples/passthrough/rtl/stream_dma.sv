// stream_dma.sv — AXI-Lite controlled Stream-to-DMA bridge
// Receives data via AXI4-Stream, writes to memory via AXI4 master.
// Control/status via AXI-Lite slave registers.
//
// Register Map (32-bit, AXI-Lite):
//   0x00: DST_ADDR_LO  — lower 32 bits of DMA destination
//   0x04: DST_ADDR_HI  — upper 32 bits of DMA destination
//   0x08: LENGTH        — number of beats to transfer
//   0x0C: CTRL          — bit 0: START (write-1-to-start)
//   0x10: STATUS        — bit 0: DONE (read-only)

module stream_dma #(
    parameter DATA_W = 256,
    parameter ADDR_W = 64
)(
    input  logic clk,
    input  logic rst_n,

    // ── AXI-Lite Slave (control) ──
    input  logic [15:0]          s_axilite_awaddr,
    input  logic                 s_axilite_awvalid,
    output logic                 s_axilite_awready,
    input  logic [31:0]          s_axilite_wdata,
    input  logic [3:0]           s_axilite_wstrb,
    input  logic                 s_axilite_wvalid,
    output logic                 s_axilite_wready,
    output logic [1:0]           s_axilite_bresp,
    output logic                 s_axilite_bvalid,
    input  logic                 s_axilite_bready,
    input  logic [15:0]          s_axilite_araddr,
    input  logic                 s_axilite_arvalid,
    output logic                 s_axilite_arready,
    output logic [31:0]          s_axilite_rdata,
    output logic [1:0]           s_axilite_rresp,
    output logic                 s_axilite_rvalid,
    input  logic                 s_axilite_rready,

    // ── AXI4-Stream Slave (input data) ──
    input  logic [DATA_W-1:0]    s_axis_tdata,
    input  logic                 s_axis_tvalid,
    output logic                 s_axis_tready,
    input  logic                 s_axis_tlast,

    // ── AXI4 Master (DMA write output) ──
    // AR channel (unused — no read)
    output logic [ADDR_W-1:0]    m_axi_araddr,
    output logic [7:0]           m_axi_arlen,
    output logic [2:0]           m_axi_arsize,
    output logic [1:0]           m_axi_arburst,
    output logic                 m_axi_arvalid,
    input  logic                 m_axi_arready,
    input  logic [DATA_W-1:0]    m_axi_rdata,
    input  logic [1:0]           m_axi_rresp,
    input  logic                 m_axi_rlast,
    input  logic                 m_axi_rvalid,
    output logic                 m_axi_rready,
    // AW channel
    output logic [ADDR_W-1:0]    m_axi_awaddr,
    output logic [7:0]           m_axi_awlen,
    output logic [2:0]           m_axi_awsize,
    output logic [1:0]           m_axi_awburst,
    output logic                 m_axi_awvalid,
    input  logic                 m_axi_awready,
    // W channel
    output logic [DATA_W-1:0]    m_axi_wdata,
    output logic [DATA_W/8-1:0]  m_axi_wstrb,
    output logic                 m_axi_wlast,
    output logic                 m_axi_wvalid,
    input  logic                 m_axi_wready,
    // B channel
    input  logic [1:0]           m_axi_bresp,
    input  logic                 m_axi_bvalid,
    output logic                 m_axi_bready
);

    localparam int BYTES_PER_BEAT = DATA_W / 8;

    // ══════════════════════════════════════════════════════════════
    // AXI4 Master Read — tie off (unused)
    // ══════════════════════════════════════════════════════════════
    assign m_axi_araddr  = '0;
    assign m_axi_arlen   = '0;
    assign m_axi_arsize  = '0;
    assign m_axi_arburst = '0;
    assign m_axi_arvalid = 1'b0;
    assign m_axi_rready  = 1'b0;

    // ══════════════════════════════════════════════════════════════
    // Registers
    // ══════════════════════════════════════════════════════════════
    logic [31:0] reg_dst_addr_lo;
    logic [31:0] reg_dst_addr_hi;
    logic [31:0] reg_length;
    logic        reg_start;       // pulse
    logic        reg_done;

    // ══════════════════════════════════════════════════════════════
    // AXI-Lite Slave — Write Path
    // ══════════════════════════════════════════════════════════════
    logic        aw_done, w_done;
    logic [15:0] aw_addr_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axilite_awready <= 1'b0;
            s_axilite_wready  <= 1'b0;
            s_axilite_bvalid  <= 1'b0;
            s_axilite_bresp   <= 2'b00;
            aw_done           <= 1'b0;
            w_done            <= 1'b0;
            reg_dst_addr_lo   <= '0;
            reg_dst_addr_hi   <= '0;
            reg_length        <= '0;
            reg_start         <= 1'b0;
        end else begin
            reg_start <= 1'b0;  // pulse — clear every cycle

            // AW handshake
            if (s_axilite_awvalid && !aw_done && !s_axilite_bvalid) begin
                s_axilite_awready <= 1'b1;
                aw_addr_latch     <= s_axilite_awaddr;
                aw_done           <= 1'b1;
            end else begin
                s_axilite_awready <= 1'b0;
            end

            // W handshake
            if (s_axilite_wvalid && !w_done && !s_axilite_bvalid) begin
                s_axilite_wready <= 1'b1;
                w_done           <= 1'b1;
            end else begin
                s_axilite_wready <= 1'b0;
            end

            // Both AW + W done → write register + send B response
            if (aw_done && w_done && !s_axilite_bvalid) begin
                case (aw_addr_latch[7:0])
                    8'h00: reg_dst_addr_lo <= s_axilite_wdata;
                    8'h04: reg_dst_addr_hi <= s_axilite_wdata;
                    8'h08: reg_length      <= s_axilite_wdata;
                    8'h0C: reg_start       <= s_axilite_wdata[0];
                    default: ;
                endcase
                s_axilite_bvalid <= 1'b1;
                s_axilite_bresp  <= 2'b00;
                aw_done          <= 1'b0;
                w_done           <= 1'b0;
            end

            // B response consumed
            if (s_axilite_bvalid && s_axilite_bready) begin
                s_axilite_bvalid <= 1'b0;
            end
        end
    end

    // ══════════════════════════════════════════════════════════════
    // AXI-Lite Slave — Read Path
    // ══════════════════════════════════════════════════════════════
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axilite_arready <= 1'b0;
            s_axilite_rvalid  <= 1'b0;
            s_axilite_rdata   <= '0;
            s_axilite_rresp   <= 2'b00;
        end else begin
            if (s_axilite_arvalid && !s_axilite_rvalid) begin
                s_axilite_arready <= 1'b1;
                s_axilite_rvalid  <= 1'b1;
                s_axilite_rresp   <= 2'b00;
                case (s_axilite_araddr[7:0])
                    8'h00:   s_axilite_rdata <= reg_dst_addr_lo;
                    8'h04:   s_axilite_rdata <= reg_dst_addr_hi;
                    8'h08:   s_axilite_rdata <= reg_length;
                    8'h0C:   s_axilite_rdata <= 32'd0;
                    8'h10:   s_axilite_rdata <= {31'd0, reg_done};
                    default: s_axilite_rdata <= 32'hDEAD_BEEF;
                endcase
            end else begin
                s_axilite_arready <= 1'b0;
            end
            if (s_axilite_rvalid && s_axilite_rready) begin
                s_axilite_rvalid <= 1'b0;
            end
        end
    end

    // ══════════════════════════════════════════════════════════════
    // DMA Engine — Stream to AXI4 Write
    // ══════════════════════════════════════════════════════════════
    typedef enum logic [2:0] {
        DMA_IDLE,
        DMA_ACCEPT_STREAM,
        DMA_WRITE_AW,
        DMA_WRITE_W,
        DMA_WAIT_B,
        DMA_DONE
    } dma_state_t;

    dma_state_t dma_state;
    logic [ADDR_W-1:0] dma_dst_addr;
    int                dma_beats_left;
    int                dma_beat_count;
    logic [DATA_W-1:0] dma_data_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_state      <= DMA_IDLE;
            s_axis_tready  <= 1'b0;
            m_axi_awvalid  <= 1'b0;
            m_axi_wvalid   <= 1'b0;
            m_axi_bready   <= 1'b0;
            reg_done       <= 1'b0;
            dma_beat_count <= 0;
        end else begin
            case (dma_state)
                DMA_IDLE: begin
                    s_axis_tready <= 1'b0;
                    if (reg_start) begin
                        dma_dst_addr   <= {reg_dst_addr_hi, reg_dst_addr_lo};
                        dma_beats_left <= reg_length;
                        dma_beat_count <= 0;
                        reg_done       <= 1'b0;
                        if (reg_length == 0) begin
                            dma_state <= DMA_DONE;
                        end else begin
                            dma_state     <= DMA_ACCEPT_STREAM;
                            s_axis_tready <= 1'b1;
                        end
                    end
                end

                DMA_ACCEPT_STREAM: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        dma_data_latch <= s_axis_tdata;
                        s_axis_tready  <= 1'b0;
                        dma_state      <= DMA_WRITE_AW;
                    end
                end

                DMA_WRITE_AW: begin
                    m_axi_awaddr  <= dma_dst_addr + dma_beat_count * BYTES_PER_BEAT;
                    m_axi_awlen   <= 8'd0;   // single beat
                    m_axi_awsize  <= $clog2(BYTES_PER_BEAT);
                    m_axi_awburst <= 2'b01;  // INCR
                    m_axi_awvalid <= 1'b1;
                    // Drive W simultaneously
                    m_axi_wdata   <= dma_data_latch;
                    m_axi_wstrb   <= {BYTES_PER_BEAT{1'b1}};
                    m_axi_wlast   <= 1'b1;
                    m_axi_wvalid  <= 1'b1;
                    dma_state     <= DMA_WRITE_W;
                end

                DMA_WRITE_W: begin
                    // Wait for both AW and W accepted
                    if (m_axi_awready && m_axi_awvalid)
                        m_axi_awvalid <= 1'b0;
                    if (m_axi_wready && m_axi_wvalid)
                        m_axi_wvalid <= 1'b0;

                    if ((!m_axi_awvalid || m_axi_awready) &&
                        (!m_axi_wvalid  || m_axi_wready)) begin
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b0;
                        m_axi_bready  <= 1'b1;
                        dma_state     <= DMA_WAIT_B;
                    end
                end

                DMA_WAIT_B: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready   <= 1'b0;
                        dma_beat_count <= dma_beat_count + 1;
                        dma_beats_left <= dma_beats_left - 1;
                        if (dma_beats_left == 1) begin
                            dma_state <= DMA_DONE;
                        end else begin
                            s_axis_tready <= 1'b1;
                            dma_state     <= DMA_ACCEPT_STREAM;
                        end
                    end
                end

                DMA_DONE: begin
                    reg_done  <= 1'b1;
                    dma_state <= DMA_IDLE;
                end
            endcase
        end
    end

endmodule
