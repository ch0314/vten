// vten_bfm_axilite.sv — AXI4-Lite BFM (master)
// Reference: specs/05_bfm_library.md §3
//
// BFM is master — drives AXI-Lite transactions.
// Supports WRITE_REG, READ_REG, POLL_REG.
//
// Diagnostics: +VTEN_VERBOSE (xsim) or +define+VTEN_VERBOSE (verilator)

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

    // Write handshake phase tracking
    logic aw_accepted;
    logic w_accepted;

    // Runtime verbose flag
    bit verbose;
    initial begin
`ifdef VTEN_VERBOSE
        verbose = 1;
`else
  `ifndef VERILATOR
        verbose = $test$plusargs("VTEN_VERBOSE");
  `else
        verbose = 0;
  `endif
`endif
    end

    // v0.4.1: idle signal — registered to avoid xsim continuous-assign issue
    logic idle_r;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            idle_r <= 1'b1;
        else
            idle_r <= !cmd_active && (cmd_queue.size() == 0);
    end
    assign cmd_if.idle = idle_r;

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
            aw_accepted  <= 0;
            w_accepted   <= 0;
        end else begin
            cmd_if.done_valid <= 0;
            // Default: deassert AXI-Lite master outputs when not driven by a command
            m_awvalid <= 0;
            m_wvalid  <= 0;
            m_bready  <= 0;
            m_arvalid <= 0;
            m_rready  <= 0;

            if (!cmd_active && cmd_queue.size() > 0) begin
                current_cmd = cmd_queue.pop_front();
                cmd_active  <= 1;
                poll_count  <= 0;
                issue_cycle <= cycle_count;
                aw_accepted <= 0;
                w_accepted  <= 0;
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
        // Drive AW channel until accepted
        if (!aw_accepted) begin
            m_awaddr  <= current_cmd.reg_offset[ADDR_W-1:0];
            m_awvalid <= 1;
            if (m_awready) begin
                m_awvalid   <= 0;
                aw_accepted <= 1;
            end
        end
        // Drive W channel until accepted
        if (!w_accepted) begin
            m_wdata  <= current_cmd.reg_value[DATA_W-1:0];
            m_wstrb  <= {(DATA_W/8){1'b1}};
            m_wvalid <= 1;
            if (m_wready) begin
                m_wvalid   <= 0;
                w_accepted <= 1;
            end
        end
        // Wait for B response
        m_bready <= 1;
        if (m_bvalid && m_bready) begin
            m_bready <= 0;
            if (verbose)
                $display("[AXILITE  %t] WRITE_REG iface=%0d cmd#%0d done: addr=0x%04h, val=0x%08h, resp=%0b",
                         $time, current_cmd.interface_id, current_cmd.cmd_id,
                         current_cmd.reg_offset[ADDR_W-1:0],
                         current_cmd.reg_value[DATA_W-1:0], m_bresp);
            finish_cmd(m_bresp != 2'b00);
        end
    endtask

    // ── READ_REG ──
    task automatic do_read();
        if (!aw_accepted) begin  // reuse aw_accepted as "ar_accepted"
            m_araddr  <= current_cmd.reg_offset[ADDR_W-1:0];
            m_arvalid <= 1;
            if (m_arready) begin
                m_arvalid   <= 0;
                aw_accepted <= 1;
            end
        end
        m_rready <= 1;
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
        // Always ready to receive read data while polling
        m_rready <= 1;

        // Drive AR on polling interval when no outstanding read
        if (poll_count % POLL_INTERVAL == 0 && !aw_accepted) begin
            m_araddr  <= current_cmd.reg_offset[ADDR_W-1:0];
            m_arvalid <= 1;
        end

        if (m_arvalid && m_arready) begin
            m_arvalid   <= 0;
            aw_accepted <= 1;
        end

        if (m_rvalid && m_rready) begin
            m_rready    <= 0;
            aw_accepted <= 0;  // reset for next poll iteration
            if ((m_rdata & current_cmd.reg_mask[DATA_W-1:0]) ==
                current_cmd.reg_expected[DATA_W-1:0]) begin
                if (verbose)
                    $display("[AXILITE  %t] POLL_REG iface=%0d cmd#%0d: match after %0d polls (addr=0x%04h, got=0x%08h, mask=0x%08h, expected=0x%08h)",
                             $time, current_cmd.interface_id, current_cmd.cmd_id, poll_count,
                             current_cmd.reg_offset[ADDR_W-1:0], m_rdata,
                             current_cmd.reg_mask[DATA_W-1:0],
                             current_cmd.reg_expected[DATA_W-1:0]);
                finish_cmd(0);
            end else begin
                poll_count <= poll_count + 1;
                if (poll_count >= POLL_TIMEOUT) begin
                    if (verbose)
                        $display("[AXILITE  %t] POLL_REG iface=%0d cmd#%0d TIMEOUT: %0d polls exhausted (addr=0x%04h, last=0x%08h, mask=0x%08h, expected=0x%08h)",
                                 $time, current_cmd.interface_id, current_cmd.cmd_id, POLL_TIMEOUT,
                                 current_cmd.reg_offset[ADDR_W-1:0], m_rdata,
                                 current_cmd.reg_mask[DATA_W-1:0],
                                 current_cmd.reg_expected[DATA_W-1:0]);
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
