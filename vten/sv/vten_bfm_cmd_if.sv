// vten_bfm_cmd_if.sv — Scheduler ↔ BFM command interface
// Reference: docs/architecture.md

`include "vten_types.svh"

interface vten_bfm_cmd_if;
    logic        cmd_valid;
    bfm_cmd_t    cmd_data;

    logic        done_valid;
    logic [15:0] done_cmd_id;
    logic        done_error;
    logic [15:0] done_error_code;

    // v0.4.1: BFM → Scheduler idle signal
    // All internal queues and pending states are empty.
    // Used for all_drained computation.
    logic        idle;

    modport scheduler (
        output cmd_valid, cmd_data,
        input  done_valid, done_cmd_id, done_error, done_error_code, idle
    );

    modport bfm (
        input  cmd_valid, cmd_data,
        output done_valid, done_cmd_id, done_error, done_error_code, idle
    );
endinterface
