// vten_bfm_axilite.sv — AXI4-Lite BFM (master)
// Reference: specs/05_bfm_library.md §3
//
// BFM is master — drives AXI-Lite transactions.
// Supports WRITE_REG, READ_REG, POLL_REG.

`include "vten_types.svh"
`include "vten_dpi_imports.svh"

module vten_bfm_axilite #(
    parameter int ADDR_W       = 32,
    parameter int DATA_W       = 32,
    parameter int POLL_INTERVAL = 1,
    parameter int POLL_TIMEOUT  = 100000
)(
    input  logic clk,
    input  logic rst_n,
    // AXI4-Lite Master
    output logic [ADDR_W-1:0]   m_awaddr,
    output logic                m_awvalid,
    input  logic                m_awready,
    output logic [DATA_W-1:0]   m_wdata,
    output logic [DATA_W/8-1:0] m_wstrb,
    output logic                m_wvalid,
    input  logic                m_wready,
    input  logic [1:0]          m_bresp,
    input  logic                m_bvalid,
    output logic                m_bready,
    output logic [ADDR_W-1:0]   m_araddr,
    output logic                m_arvalid,
    input  logic                m_arready,
    input  logic [DATA_W-1:0]   m_rdata,
    input  logic [1:0]          m_rresp,
    input  logic                m_rvalid,
    output logic                m_rready,
    // Scheduler interface
    vten_bfm_cmd_if.bfm         cmd_if,
    // Global cycle counter
    input  int                  cycle_count
);
    bfm_cmd_t cmd_queue[$];
    bfm_cmd_t current_cmd;
    logic cmd_active;
    int poll_count;
    int issue_cycle;

    // v0.4.1: idle signal
    assign cmd_if.idle = !cmd_active && (cmd_queue.size() == 0);

    // Command receive
    always_ff @(posedge clk) begin
        if (cmd_if.cmd_valid) cmd_queue.push_back(cmd_if.cmd_data);
    end

    // Command execution
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_active   <= 0;
            cmd_if.done_valid <= 0;
            m_awvalid    <= 0;
            m_wvalid     <= 0;
            m_bready     <= 0;
            m_arvalid    <= 0;
            m_rready     <= 0;
        end else begin
            cmd_if.done_valid <= 0;

            if (!cmd_active && cmd_queue.size() > 0) begin
                current_cmd = cmd_queue.pop_front();
                cmd_active  <= 1;
                poll_count  <= 0;
                issue_cycle <= cycle_count;
            end

            if (cmd_active) begin
                case (current_cmd.opcode)
                    OP_WRITE_REG: do_write();
                    OP_READ_REG:  do_read();
                    OP_POLL_REG:  do_poll();
                    default: begin
                        // Unknown opcode for AXI-Lite
                        finish_cmd(1'b1);
                    end
                endcase
            end
        end
    end

    // ── WRITE_REG ──
    task automatic do_write();
        // Drive AW + W simultaneously
        m_awaddr  <= current_cmd.reg_offset[ADDR_W-1:0];
        m_awvalid <= 1;
        m_wdata   <= current_cmd.reg_value[DATA_W-1:0];
        m_wstrb   <= {(DATA_W/8){1'b1}};
        m_wvalid  <= 1;
        m_bready  <= 1;

        if (m_awready) m_awvalid <= 0;
        if (m_wready)  m_wvalid  <= 0;
        if (m_bvalid && m_bready) begin
            m_bready <= 0;
            finish_cmd(m_bresp != 2'b00);
        end
    endtask

    // ── READ_REG ──
    task automatic do_read();
        m_araddr  <= current_cmd.reg_offset[ADDR_W-1:0];
        m_arvalid <= 1;
        m_rready  <= 1;

        if (m_arready) m_arvalid <= 0;
        if (m_rvalid && m_rready) begin
            m_rready <= 0;
            // Write read value to Stats
            vten_write_cmd_stats(current_cmd.cmd_id,
                CMD_COMMITTED, issue_cycle, cycle_count,
                cycle_count, cycle_count, 1, 1, 0);
            finish_cmd(m_rresp != 2'b00);
        end
    endtask

    // ── POLL_REG ──
    task automatic do_poll();
        // Configurable interval polling
        if (poll_count % POLL_INTERVAL == 0) begin
            m_araddr  <= current_cmd.reg_offset[ADDR_W-1:0];
            m_arvalid <= 1;
            m_rready  <= 1;
        end

        if (m_arready) m_arvalid <= 0;

        if (m_rvalid && m_rready) begin
            m_rready <= 0;
            if ((m_rdata & current_cmd.reg_mask[DATA_W-1:0]) ==
                current_cmd.reg_expected[DATA_W-1:0]) begin
                finish_cmd(0);
            end else begin
                poll_count <= poll_count + 1;
                if (poll_count >= POLL_TIMEOUT) begin
                    finish_cmd(1);  // Timeout error
                end
            end
        end
    endtask

    // ── finish_cmd ──
    task automatic finish_cmd(logic error);
        cmd_active <= 0;
        cmd_if.done_valid    <= 1;
        cmd_if.done_cmd_id   <= current_cmd.cmd_id;
        cmd_if.done_error    <= error;
        // v0.4.2: BackendErrorCode (00_data_models.md §11.13)
        if (error && current_cmd.opcode == OP_POLL_REG)
            cmd_if.done_error_code <= ERR_POLL_TIMEOUT;     // 2
        else if (error)
            cmd_if.done_error_code <= ERR_BFM_QUEUE;  // 3
        else
            cmd_if.done_error_code <= ERR_OK;               // 0

        // Write stats for non-poll commands
        if (current_cmd.opcode != OP_READ_REG) begin
            vten_write_cmd_stats(current_cmd.cmd_id,
                error ? CMD_ERROR : CMD_COMMITTED,
                issue_cycle, cycle_count,
                0, 0, 0, 0, 0);
        end
    endtask
endmodule
