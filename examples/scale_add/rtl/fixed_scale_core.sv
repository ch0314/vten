// fixed_scale_core.sv — AXI4-Stream: multiply each signed byte by a Q8.8
// fixed-point coefficient with round-half-up and signed saturation.
//
// Per-lane datapath (the arithmetic the QuantSpec golden must reproduce):
//   prod24  = $signed(x8) * $signed(coeff16)        // Q1.7 * Q8.8 = Q9.15
//   rounded = (prod24 + 24'sd128) >>> 8             // half-up: +half LSB, floor
//   out8    = clamp(rounded, -128, 127)             // saturate back to Q1.7
//
// Uses SV interfaces for AXI4-Stream (compatible with generated wrapper)

module fixed_scale_core #(parameter DATA_W = 256)(
    input  logic        clk,
    input  logic        rst_n,

    // Register wires (from AXI-Lite controller)
    input  logic [15:0] reg_coeff,        // SIGNED Q8.8 coefficient (256 = 1.0)
    input  logic [31:0] reg_length,       // total beats to process
    input  logic [0:0]  reg_ctrl,         // start pulse
    output logic [31:0] reg_status,       // bit[0] = done

    // AXI4-Stream (SV interfaces)
    vten_axis_if.slave  input_stream,
    vten_axis_if.master output_stream
);

    // ── FSM ──
    typedef enum logic [1:0] { S_IDLE, S_RUN, S_DONE } state_t;
    state_t state, state_next;

    logic [31:0] beat_count;
    logic        handshake;

    assign handshake = input_stream.tvalid & input_stream.tready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_next;
    end

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE: if (reg_ctrl[0]) state_next = S_RUN;
            S_RUN:  if (handshake && (beat_count + 1 >= reg_length))
                        state_next = S_DONE;
            S_DONE: if (reg_ctrl[0]) state_next = S_RUN;  // Restart directly into RUN
            default: state_next = S_IDLE;
        endcase
    end

    // Beat counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            beat_count <= 0;
        else if (state == S_IDLE || state == S_DONE)
            beat_count <= 0;
        else if (handshake)
            beat_count <= beat_count + 1;
    end

    // Status (full-width drive so AXI-Lite readback has no Z bits)
    assign reg_status = {31'b0, state == S_DONE};

    // Flow control
    assign input_stream.tready  = (state == S_RUN) & output_stream.tready;
    assign output_stream.tvalid = (state == S_RUN) & input_stream.tvalid;
    assign output_stream.tlast  = input_stream.tlast;

    // ── Data path: Q8.8 multiply, round-half-up, signed saturation ──
    genvar i;
    generate
        for (i = 0; i < DATA_W / 8; i = i + 1) begin : fscale
            // Q1.7 (int8) x Q8.8 (int16) → Q9.15 in 24 bits
            // (|prod| <= 128 * 32768 = 2^22, fits signed 24-bit)
            wire signed [23:0] product =
                $signed(input_stream.tdata[i*8 +: 8]) * $signed(reg_coeff);
            // round half-up: add half an output LSB (1 << 7), arithmetic
            // shift right by 8 — ties round toward +inf for both signs
            wire signed [23:0] rounded = (product + 24'sd128) >>> 8;
            assign output_stream.tdata[i*8 +: 8] =
                (rounded > 24'sd127)  ? 8'sd127  :
                (rounded < -24'sd128) ? -8'sd128 :
                rounded[7:0];
        end
    endgenerate

endmodule
