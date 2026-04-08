// vten_sram_model.v — Behavioral dual-port SRAM model for Verilator
//
// Drop-in replacement for Xilinx blk_mem_gen IP.
// Port-compatible with Vivado-generated `sram_64x256` module.
//
// Behavior:
//   - Simple dual-port: Port A = write, Port B = read
//   - 1-cycle read latency (output register on Port B)
//   - Independent clocks (clka, clkb)

`timescale 1 ps / 1 ps

module sram_64x256 (
    // Port A (write)
    input         clka,
    input         ena,
    input  [0:0]  wea,
    input  [7:0]  addra,
    input  [63:0] dina,

    // Port B (read)
    input         clkb,
    input         enb,
    input  [7:0]  addrb,
    output [63:0] doutb
);

    localparam DEPTH = 256;
    localparam WIDTH = 64;

    // ── Memory array ──
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    // ── Port A: write ──
    always @(posedge clka) begin
        if (ena && wea[0]) begin
            mem[addra] <= dina;
        end
    end

    // ── Port B: read with output register ──
    reg [WIDTH-1:0] doutb_r;
    assign doutb = doutb_r;

    always @(posedge clkb) begin
        if (enb) begin
            doutb_r <= mem[addrb];
        end
    end

endmodule
