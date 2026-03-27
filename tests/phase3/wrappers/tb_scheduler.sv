// tb_scheduler.sv — Verilator wrapper for vten_command_scheduler
//
// Flattens the BFM interface array into flat ports for verilator.
// Uses MAX_BFMS=4, MAX_CMDS=32, MAX_IFACES=8 for testing.

`include "vten_types.svh"

module tb_scheduler #(
    parameter int MAX_BFMS  = 4,
    parameter int MAX_CMDS  = 32,
    parameter int MAX_IFACES = 8
)(
    input  logic clk,
    input  logic rst_n,

    // Controller → Scheduler: command feed
    input  logic        feed_valid,
    input  logic [303:0] feed_data_flat,   // bfm_cmd_t flattened
    output logic        feed_ready,
    input  logic        feed_done,
    input  logic        batch_init,

    // Scheduler → Controller: status
    output logic        all_committed,
    output logic        all_drained,
    output logic        error_flag,
    output logic [15:0] error_cmd_id,
    output logic [15:0] error_code,

    // Cycle count
    input  int          cycle_count,

    // BFM mapping: iface_to_bfm[0..7]
    input  int          itb_0, itb_1, itb_2, itb_3,
    input  int          itb_4, itb_5, itb_6, itb_7,

    // BFM 0 flat ports
    output logic        bfm0_cmd_valid,
    output logic [303:0] bfm0_cmd_data,
    input  logic        bfm0_done_valid,
    input  logic [15:0] bfm0_done_cmd_id,
    input  logic        bfm0_done_error,
    input  logic [15:0] bfm0_done_error_code,
    input  logic        bfm0_idle,

    // BFM 1 flat ports
    output logic        bfm1_cmd_valid,
    output logic [303:0] bfm1_cmd_data,
    input  logic        bfm1_done_valid,
    input  logic [15:0] bfm1_done_cmd_id,
    input  logic        bfm1_done_error,
    input  logic [15:0] bfm1_done_error_code,
    input  logic        bfm1_idle,

    // BFM 2 flat ports
    output logic        bfm2_cmd_valid,
    output logic [303:0] bfm2_cmd_data,
    input  logic        bfm2_done_valid,
    input  logic [15:0] bfm2_done_cmd_id,
    input  logic        bfm2_done_error,
    input  logic [15:0] bfm2_done_error_code,
    input  logic        bfm2_idle,

    // BFM 3 flat ports
    output logic        bfm3_cmd_valid,
    output logic [303:0] bfm3_cmd_data,
    input  logic        bfm3_done_valid,
    input  logic [15:0] bfm3_done_cmd_id,
    input  logic        bfm3_done_error,
    input  logic [15:0] bfm3_done_error_code,
    input  logic        bfm3_idle
);

    // ── Interface instances ──
    vten_bfm_cmd_if bfm_if [MAX_BFMS] ();

    // ── iface_to_bfm mapping array ──
    int iface_to_bfm [0:MAX_IFACES-1];
    assign iface_to_bfm[0] = itb_0;
    assign iface_to_bfm[1] = itb_1;
    assign iface_to_bfm[2] = itb_2;
    assign iface_to_bfm[3] = itb_3;
    assign iface_to_bfm[4] = itb_4;
    assign iface_to_bfm[5] = itb_5;
    assign iface_to_bfm[6] = itb_6;
    assign iface_to_bfm[7] = itb_7;

    // ── Cast flat feed_data to bfm_cmd_t ──
    bfm_cmd_t feed_cmd;
    assign feed_cmd = feed_data_flat;

    // ── Scheduler instance ──
    vten_command_scheduler #(
        .MAX_CMDS(MAX_CMDS),
        .MAX_BFMS(MAX_BFMS),
        .MAX_IFACES(MAX_IFACES)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .feed_valid(feed_valid),
        .feed_data(feed_cmd),
        .feed_ready(feed_ready),
        .feed_done(feed_done),
        .batch_init(batch_init),
        .all_committed(all_committed),
        .all_drained(all_drained),
        .error_flag(error_flag),
        .error_cmd_id(error_cmd_id),
        .error_code(error_code),
        .bfm(bfm_if),
        .cycle_count(cycle_count),
        .iface_to_bfm(iface_to_bfm)
    );

    // ── BFM 0 wiring ──
    assign bfm0_cmd_valid = bfm_if[0].cmd_valid;
    assign bfm0_cmd_data  = bfm_if[0].cmd_data;
    assign bfm_if[0].done_valid    = bfm0_done_valid;
    assign bfm_if[0].done_cmd_id   = bfm0_done_cmd_id;
    assign bfm_if[0].done_error    = bfm0_done_error;
    assign bfm_if[0].done_error_code = bfm0_done_error_code;
    assign bfm_if[0].idle          = bfm0_idle;

    // ── BFM 1 wiring ──
    assign bfm1_cmd_valid = bfm_if[1].cmd_valid;
    assign bfm1_cmd_data  = bfm_if[1].cmd_data;
    assign bfm_if[1].done_valid    = bfm1_done_valid;
    assign bfm_if[1].done_cmd_id   = bfm1_done_cmd_id;
    assign bfm_if[1].done_error    = bfm1_done_error;
    assign bfm_if[1].done_error_code = bfm1_done_error_code;
    assign bfm_if[1].idle          = bfm1_idle;

    // ── BFM 2 wiring ──
    assign bfm2_cmd_valid = bfm_if[2].cmd_valid;
    assign bfm2_cmd_data  = bfm_if[2].cmd_data;
    assign bfm_if[2].done_valid    = bfm2_done_valid;
    assign bfm_if[2].done_cmd_id   = bfm2_done_cmd_id;
    assign bfm_if[2].done_error    = bfm2_done_error;
    assign bfm_if[2].done_error_code = bfm2_done_error_code;
    assign bfm_if[2].idle          = bfm2_idle;

    // ── BFM 3 wiring ──
    assign bfm3_cmd_valid = bfm_if[3].cmd_valid;
    assign bfm3_cmd_data  = bfm_if[3].cmd_data;
    assign bfm_if[3].done_valid    = bfm3_done_valid;
    assign bfm_if[3].done_cmd_id   = bfm3_done_cmd_id;
    assign bfm_if[3].done_error    = bfm3_done_error;
    assign bfm_if[3].done_error_code = bfm3_done_error_code;
    assign bfm_if[3].idle          = bfm3_idle;

endmodule
