// vten_dpi_imports.svh — DPI-C function import declarations
// Reference: specs/04_backend_xsim.md §6
// Include this file in tb_top.sv. All BFM/Controller/Scheduler modules reference these.

`ifndef VTEN_DPI_IMPORTS_SVH
`define VTEN_DPI_IMPORTS_SVH

// ── Lifecycle ──
import "DPI-C" function int  vten_shm_init(input string session_id);
import "DPI-C" function void vten_cleanup();

// ── Host/Backend Synchronization ──
import "DPI-C" function int  vten_wait_host_signal_safe(input int timeout_ms);
import "DPI-C" function int  vten_read_host_status();
import "DPI-C" function void vten_set_backend_status(input int status);
import "DPI-C" function void vten_signal_complete();
import "DPI-C" function void vten_signal_error(input int code, input string msg);

// ── Control Header ──
import "DPI-C" function int  vten_read_num_commands();
import "DPI-C" function int  vten_read_num_buffers();
import "DPI-C" function int  vten_read_timeout_ms();
import "DPI-C" function int  vten_read_flags();

// ── Command Region ──
import "DPI-C" function int  vten_read_command(
    input int cmd_id,
    output int opcode, output int interface_id,
    output int protocol, output int role,
    output int buffer_id, output int probe, output int flags,
    output int size, output longint phys_addr,
    output int reg_offset, output int reg_value,
    output int reg_mask, output int reg_expected,
    output int golden_buf_id,
    output int num_deps, output int num_commit_deps,
    output int dep_ids [0:3], output int commit_dep_ids [0:3]);

import "DPI-C" function void vten_read_command_deps(
    input int cmd_id,
    output int num_dep, output int dep_ids [0:3],
    output int num_cdep, output int cdep_ids [0:3]);

// ── Data Region ──
import "DPI-C" function void vten_read_data(
    input int buf_id, input int offset, input int size,
    output bit [7:0] dst []);

import "DPI-C" function void vten_write_data(
    input int buf_id, input int offset, input int size,
    input bit [7:0] src []);

// Scalar byte access — portable across all simulators (no open array issues)
import "DPI-C" function int  vten_read_data_byte(
    input int buf_id, input int offset);

import "DPI-C" function void vten_write_data_byte(
    input int buf_id, input int offset, input int value);

// ── Stats Region ──
import "DPI-C" function void vten_write_cmd_stats(
    input int cmd_id, input int status,
    input int issue_cycle, input int commit_cycle,
    input int first_active, input int last_active,
    input int active_cycles, input int total_beats, input int stall_cycles);

import "DPI-C" function void vten_write_cmd_status(
    input int cmd_id, input int status);

// ── Probe ──
import "DPI-C" function void vten_read_golden(
    input int buf_id, input int beat_index,
    output bit [7:0] dst []);

import "DPI-C" function int  vten_read_golden_byte(
    input int buf_id, input int byte_offset);

import "DPI-C" function void vten_log_mismatch(
    input int cycle, input int beat,
    input int expected_hi, input int expected_lo,
    input int actual_hi, input int actual_lo);

`endif // VTEN_DPI_IMPORTS_SVH
