// offset_core.sv — AXI4-Stream: add signed offset_value to each byte (saturating)
// Uses SV interfaces for AXI4-Stream (compatible with generated wrapper)

module offset_core #(parameter DATA_W = 256)(
    input  logic        clk,
    input  logic        rst_n,

    // Register wires (from AXI-Lite controller)
    input  logic [7:0]  reg_offset_value,
    input  logic [31:0] reg_length,       // total beats to process
    input  logic [0:0]  reg_ctrl,         // start pulse
    output logic [0:0]  reg_status,       // done

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

    // Status
    assign reg_status[0] = (state == S_DONE);

    // Flow control
    assign input_stream.tready  = (state == S_RUN) & output_stream.tready;
    assign output_stream.tvalid = (state == S_RUN) & input_stream.tvalid;
    assign output_stream.tlast  = input_stream.tlast;

    // ── Data path: add offset_value to each byte, signed saturation ──
    genvar i;
    generate
        for (i = 0; i < DATA_W / 8; i = i + 1) begin : offset
            wire signed [8:0] sum =
                $signed(input_stream.tdata[i*8 +: 8]) + $signed(reg_offset_value);
            assign output_stream.tdata[i*8 +: 8] =
                (sum > 9'sd127)  ? 8'sd127  :
                (sum < -9'sd128) ? -8'sd128 :
                sum[7:0];
        end
    endgenerate

endmodule
