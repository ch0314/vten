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

    // v0.4.1: idle signal — used by Scheduler for all_drained
    assign cmd_if.idle = !cmd_active && (cmd_queue.size() == 0);

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
    task automatic execute_master();
        bit [7:0] beat_data [0:BYTES_PER_BEAT-1];
        vten_read_data(current_cmd.buffer_id,
                       beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, beat_data);

        for (int i = 0; i < BYTES_PER_BEAT; i++)
            m_tdata[i*8 +: 8] <= beat_data[i];

        m_tvalid <= 1'b1;
        m_tlast  <= (beat_count == expected_beats - 1);

        if (m_tvalid && m_tready) begin
            beat_count <= beat_count + 1;
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;
            if (beat_count == expected_beats - 1) begin
                m_tvalid <= 1'b0;
                finish_command();
            end
        end else if (m_tvalid && !m_tready)
            stall_cycles <= stall_cycles + 1;
    endtask

    // SLAVE mode: PULL (DUT → SHM)
    task automatic execute_slave();
        s_tready <= 1'b1;
        if (s_tvalid && s_tready) begin
            bit [7:0] wr_data [0:BYTES_PER_BEAT-1];
            for (int i = 0; i < BYTES_PER_BEAT; i++)
                wr_data[i] = s_tdata[i*8 +: 8];

            vten_write_data(current_cmd.buffer_id,
                            beat_count * BYTES_PER_BEAT, BYTES_PER_BEAT, wr_data);
            beat_count <= beat_count + 1;
            active_cycles <= active_cycles + 1;
            total_beats <= total_beats + 1;
            if (first_active == 0) first_active <= cycle_count;
            last_active <= cycle_count;

            // Probe mode: beat-by-beat golden comparison
            if (current_cmd.probe) begin : probe_blk
                bit [7:0] golden_data [0:BYTES_PER_BEAT-1];
                logic [DATA_W-1:0] golden;
                vten_read_golden(current_cmd.golden_buf_id, beat_count, golden_data);
                for (int i = 0; i < BYTES_PER_BEAT; i++)
                    golden[i*8 +: 8] = golden_data[i];
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
        // NBA timing: total_beats/active_cycles increments are scheduled via
        // NBA in the same cycle — the final increment is not yet visible here.
        // Stats intentionally report the pre-NBA values (N-1).
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
