// vten_fifo_model.v — Behavioral synchronous FWFT FIFO model for Verilator
//
// Drop-in replacement for Xilinx fifo_generator IP (FWFT mode).
// Port-compatible with Vivado-generated `fifo_ofm_256x512` module.
//
// FWFT (First-Word-Fall-Through) behavior:
//   - Output register auto-fills from backing FIFO when empty
//   - dout shows head-of-FIFO without needing rd_en
//   - rd_en = "pop" — triggers load of next entry into output register
//   - 1-cycle latency: data at output register 1 cycle after it enters backing store
//   - Synchronous reset (srst, active-high)

`timescale 1 ps / 1 ps

module fifo_ofm_256x512 (
    input              clk,
    input              srst,

    // Write interface
    input  [255:0]     din,
    input              wr_en,
    output             full,
    output             wr_rst_busy,

    // Read interface
    input              rd_en,
    output [255:0]     dout,
    output             empty,
    output             valid,
    output             rd_rst_busy
);

    localparam WIDTH = 256;
    localparam DEPTH = 512;
    localparam ADDR_BITS = 9;  // log2(512)

    // ── Backing store ──
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_BITS:0] wr_ptr;  // Extra bit for full/empty distinction
    reg [ADDR_BITS:0] rd_ptr;

    wire [ADDR_BITS:0] count = wr_ptr - rd_ptr;
    wire fifo_empty_i = (count == 0);
    wire fifo_full_i  = (count == DEPTH);

    // ── Output register (FWFT prefetch) ──
    reg [WIDTH-1:0] dout_reg;
    reg             dout_valid;

    // Internal read: auto-fill when output consumed or initially empty
    wire do_read = !fifo_empty_i && (!dout_valid || rd_en);

    assign dout  = dout_reg;
    assign valid = dout_valid;
    assign empty = !dout_valid;
    assign full  = fifo_full_i;

    // ── Reset busy (2-cycle model) ──
    reg [1:0] rst_busy_sr;
    assign wr_rst_busy = rst_busy_sr[0];
    assign rd_rst_busy = rst_busy_sr[0];

    always @(posedge clk) begin
        if (srst) begin
            wr_ptr      <= 0;
            rd_ptr      <= 0;
            dout_valid  <= 1'b0;
            dout_reg    <= {WIDTH{1'b0}};
            rst_busy_sr <= 2'b11;
        end else begin
            rst_busy_sr <= {1'b0, rst_busy_sr[1]};

            // Write to backing store (always, regardless of FWFT state)
            if (wr_en && !fifo_full_i) begin
                mem[wr_ptr[ADDR_BITS-1:0]] <= din;
                wr_ptr <= wr_ptr + 1;
            end

            // FWFT output register management
            if (do_read) begin
                // Load next entry from backing store into output register
                dout_reg   <= mem[rd_ptr[ADDR_BITS-1:0]];
                rd_ptr     <= rd_ptr + 1;
                dout_valid <= 1'b1;
            end else if (rd_en && dout_valid) begin
                // Pop but backing store is empty — output goes invalid
                dout_valid <= 1'b0;
            end
        end
    end

endmodule
