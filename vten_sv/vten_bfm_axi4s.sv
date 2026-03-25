// vten_bfm_axi4s.sv — AXI4-Stream BFM (MASTER/SLAVE via parameter)
// Reference: specs/05_bfm_library.md §1

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_bfm_axi4s #(
    parameter int DATA_W = 256,
    parameter     MODE   = "MASTER"
)(
    input  logic clk,
    input  logic rst_n,
    // AXI4-Stream
    output logic [DATA_W-1:0]   m_tdata,
    output logic                m_tvalid,
    input  logic                m_tready,
    output logic                m_tlast,
    input  logic [DATA_W-1:0]   s_tdata,
    input  logic                s_tvalid,
    output logic                s_tready,
    input  logic                s_tlast,
    // Scheduler interface
    vten_bfm_cmd_if.bfm         cmd_if,
    // Global cycle counter (from tb_top)
    input  int                  cycle_count
);
    localparam int BYTES_PER_BEAT = DATA_W / 8;

    bfm_cmd_t cmd_queue[$];
    bfm_cmd_t current_cmd;
    logic      cmd_active;
    int beat_count, expected_beats;
    int issue_cycle, first_active, last_active;
    int active_cycles, stall_cycles, total_beats;

    // v0.4.1: idle signal — registered to avoid xsim continuous-assign issue
    // with queue.size() not re-evaluating in assign/always_comb.
    logic idle_r;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            idle_r <= 1'b1;
        else
            idle_r <= !cmd_active && (cmd_queue.size() == 0);
    end
    assign cmd_if.idle = idle_r;

    // Bulk transfer buffer: byte[] gives contiguous 1-byte stride on all
    // simulators (xsim, verilator), enabling memcpy-based DPI-C transfer.
    byte beat_buf [0:BYTES_PER_BEAT-1];

    // Command receive
    always_ff @(posedge clk) begin
        if (cmd_if.cmd_valid) cmd_queue.push_back(cmd_if.cmd_data);
    end

    // Command activation & execution
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_active <= 0;
            cmd_if.done_valid <= 0;
            m_tvalid <= 0;
            s_tready <= 0;
        end else begin
            cmd_if.done_valid <= 0;

            if (!cmd_active && cmd_queue.size() > 0) begin
                current_cmd = cmd_queue.pop_front();
                cmd_active <= 1;
                beat_count <= 0;
                expected_beats <= current_cmd.size / BYTES_PER_BEAT;
                issue_cycle <= cycle_count;
                first_active <= 0;
                last_active <= 0;
                active_cycles <= 0;
                stall_cycles <= 0;
                total_beats <= 0;
            end

            if (cmd_active) begin
                if (MODE == "MASTER") execute_master();
                else                  execute_slave();
            end
        end
    end

    // MASTER mode: PUSH (SHM → DUT)
    // NBA-aware: on handshake, drive NEXT beat; on initial, drive beat 0.
    task automatic execute_master();
        if (m_tvalid && m_tready) begin
            // Current beat consumed
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;
            if (beat_count == expected_beats - 1) begin
                // Last beat — done
                m_tvalid <= 1'b0;
                beat_count <= beat_count + 1;
                finish_command();
            end else begin
                // Drive next beat data (beat_count+1) — bulk read
                vten_read_data_bulk(current_cmd.buffer_id,
                    (beat_count + 1) * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    m_tdata[i*8 +: 8] <= beat_buf[i];
                m_tlast <= ((beat_count + 1) == expected_beats - 1);
                beat_count <= beat_count + 1;
            end
        end else if (!m_tvalid) begin
            // Initial: drive first beat — bulk read
            vten_read_data_bulk(current_cmd.buffer_id,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
            for (int i = 0; i < BYTES_PER_BEAT; i++)
                m_tdata[i*8 +: 8] <= beat_buf[i];
            m_tvalid <= 1'b1;
            m_tlast  <= (expected_beats == 1);
        end else begin
            // m_tvalid=1 but not ready → stall
            stall_cycles <= stall_cycles + 1;
        end
    endtask

    // SLAVE mode: PULL (DUT → SHM)
    task automatic execute_slave();
        s_tready <= 1'b1;
        if (s_tvalid && s_tready) begin
            // Bulk write: pack tdata into byte buffer, then single memcpy
            for (int i = 0; i < BYTES_PER_BEAT; i++)
                beat_buf[i] = s_tdata[i*8 +: 8];
            vten_write_data_bulk(current_cmd.buffer_id,
                beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_buf);
            beat_count <= beat_count + 1;
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;

            // Probe mode: beat-by-beat golden comparison
            if (current_cmd.probe) begin : probe_blk
                logic [DATA_W-1:0] golden;
                byte golden_buf [0:BYTES_PER_BEAT-1];
                vten_read_golden_bulk(current_cmd.golden_buf_id,
                    beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, golden_buf);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    golden[i*8 +: 8] = golden_buf[i];
                if (s_tdata !== golden)
                    vten_log_mismatch(cycle_count, beat_count,
                                      golden[DATA_W-1:DATA_W/2],
                                      golden[DATA_W/2-1:0],
                                      s_tdata[DATA_W-1:DATA_W/2],
                                      s_tdata[DATA_W/2-1:0]);
            end

            if (beat_count == expected_beats - 1) begin
                s_tready <= 1'b0;
                finish_command();
            end
        end else if (!s_tvalid && s_tready)
            stall_cycles <= stall_cycles + 1;
    endtask

    task automatic finish_command();
        cmd_active <= 0;
        cmd_if.done_valid  <= 1'b1;
        cmd_if.done_cmd_id <= current_cmd.cmd_id;
        cmd_if.done_error  <= 1'b0;
        cmd_if.done_error_code <= 16'd0;
        vten_write_cmd_stats(current_cmd.cmd_id,
            CMD_COMMITTED, issue_cycle, cycle_count,
            (first_active == 0) ? cycle_count : first_active,
            cycle_count,
            active_cycles, total_beats, stall_cycles);
    endtask
endmodule
