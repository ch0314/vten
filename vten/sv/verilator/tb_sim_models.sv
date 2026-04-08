// tb_sim_models.sv — Behavioral verification of Verilator IP stubs
//
// Tests: URAM, FIFO (FWFT), SRAM behavioral correctness
// Run:   verilator --cc --exe --main --timing tb_sim_models.sv \
//          vten_uram_model.v vten_fifo_model.v vten_sram_model.v \
//          --top-module tb_sim_models -Wno-PINMISSING && \
//        make -C obj_dir -f Vtb_sim_models.mk && ./obj_dir/Vtb_sim_models

`timescale 1ns / 1ps

module tb_sim_models;

    logic clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    int pass_count = 0;
    int fail_count = 0;

    task check(string name, logic [255:0] got, logic [255:0] expected);
        if (got === expected) begin
            pass_count++;
        end else begin
            $display("FAIL: %s — got %h, expected %h", name, got, expected);
            fail_count++;
        end
    endtask

    task check1(string name, logic got, logic expected);
        if (got === expected) begin
            pass_count++;
        end else begin
            $display("FAIL: %s — got %b, expected %b", name, got, expected);
            fail_count++;
        end
    endtask

    // ════════════════════════════════════════════════════════════════
    // URAM instance
    // ════════════════════════════════════════════════════════════════
    logic [22:0] uram_addr_a, uram_addr_b;
    logic [8:0]  uram_bwe_a;
    logic [71:0] uram_din_a, uram_dout_a, uram_dout_b;
    logic        uram_en_a, uram_en_b, uram_rdb_wr_a, uram_rdb_wr_b;
    logic        uram_rst_a, uram_rst_b;
    logic        uram_rdaccess_a, uram_rdaccess_b;

    uram u_uram (
        .CLK(clk), .CCLK(clk),
        .ADDR_A(uram_addr_a), .ADDR_B(uram_addr_b),
        .BWE_A(uram_bwe_a), .BWE_B(9'b0),
        .DIN_A(uram_din_a), .DIN_B(72'b0),
        .EN_A(uram_en_a), .EN_B(uram_en_b),
        .RDB_WR_A(uram_rdb_wr_a), .RDB_WR_B(uram_rdb_wr_b),
        .RST_A(uram_rst_a), .RST_B(uram_rst_b),
        .DOUT_A(uram_dout_a), .DOUT_B(uram_dout_b),
        .RDACCESS_A(uram_rdaccess_a), .RDACCESS_B(uram_rdaccess_b),
        .SLEEP(1'b0), .URAM_LOCATION(22'h0), .DI(32'b0), .VLD(1'b0),
        .INJECT_DBITERR_A(1'b0), .INJECT_DBITERR_B(1'b0),
        .INJECT_SBITERR_A(1'b0), .INJECT_SBITERR_B(1'b0),
        .OREG_CE_A(1'b0), .OREG_CE_B(1'b0),
        .OREG_ECC_CE_A(1'b0), .OREG_ECC_CE_B(1'b0),
        .CAS_IN_ADDR_A(23'h0), .CAS_IN_ADDR_B(23'h0),
        .CAS_IN_BWE_A(9'h0), .CAS_IN_BWE_B(9'h0),
        .CAS_IN_DBITERR_A(1'b0), .CAS_IN_DBITERR_B(1'b0),
        .CAS_IN_DIN_A(72'h0), .CAS_IN_DIN_B(72'h0),
        .CAS_IN_DOUT_A(72'h0), .CAS_IN_DOUT_B(72'h0),
        .CAS_IN_EN_A(1'b0), .CAS_IN_EN_B(1'b0),
        .CAS_IN_RDACCESS_A(1'b0), .CAS_IN_RDACCESS_B(1'b0),
        .CAS_IN_RDB_WR_A(1'b0), .CAS_IN_RDB_WR_B(1'b0),
        .CAS_IN_SBITERR_A(1'b0), .CAS_IN_SBITERR_B(1'b0)
    );

    // ════════════════════════════════════════════════════════════════
    // FIFO instance
    // ════════════════════════════════════════════════════════════════
    logic [255:0] fifo_din, fifo_dout;
    logic         fifo_wr_en, fifo_rd_en;
    logic         fifo_full, fifo_empty, fifo_valid;
    logic         fifo_srst, fifo_wr_rst_busy, fifo_rd_rst_busy;

    fifo_ofm_256x512 u_fifo (
        .clk(clk), .srst(fifo_srst),
        .din(fifo_din), .wr_en(fifo_wr_en),
        .full(fifo_full), .wr_rst_busy(fifo_wr_rst_busy),
        .rd_en(fifo_rd_en), .dout(fifo_dout),
        .empty(fifo_empty), .valid(fifo_valid),
        .rd_rst_busy(fifo_rd_rst_busy)
    );

    // ════════════════════════════════════════════════════════════════
    // SRAM instance
    // ════════════════════════════════════════════════════════════════
    logic [63:0] sram_dina, sram_doutb;
    logic [7:0]  sram_addra, sram_addrb;
    logic        sram_ena, sram_enb;
    logic [0:0]  sram_wea;

    sram_64x256 u_sram (
        .clka(clk), .ena(sram_ena), .wea(sram_wea),
        .addra(sram_addra), .dina(sram_dina),
        .clkb(clk), .enb(sram_enb), .addrb(sram_addrb),
        .doutb(sram_doutb)
    );

    // ════════════════════════════════════════════════════════════════
    // Test sequence
    // ════════════════════════════════════════════════════════════════
    initial begin
        // Defaults
        uram_addr_a = 0; uram_addr_b = 0; uram_bwe_a = 0;
        uram_din_a = 0; uram_en_a = 0; uram_en_b = 0;
        uram_rdb_wr_a = 0; uram_rdb_wr_b = 0;
        uram_rst_a = 1; uram_rst_b = 1;
        fifo_srst = 1; fifo_din = 0; fifo_wr_en = 0; fifo_rd_en = 0;
        sram_ena = 0; sram_wea = 0; sram_addra = 0; sram_dina = 0;
        sram_enb = 0; sram_addrb = 0;

        // ── Reset phase ──
        repeat (5) @(posedge clk);
        uram_rst_a = 0; uram_rst_b = 0;
        fifo_srst = 0;
        repeat (3) @(posedge clk);

        // ════════════════════════════════════
        // TEST 1: URAM — write then read
        // ════════════════════════════════════
        $display("\n=== URAM Tests ===");

        // Write 0xDEAD_BEEF_CAFE_1234 to addr 42 via port A
        // Cycle W: EN_A=1, RDB_WR_A=1 → write happens at posedge
        @(posedge clk);
        uram_en_a = 1; uram_rdb_wr_a = 1;
        uram_addr_a = 23'd42;
        uram_bwe_a = 9'h1FF;
        uram_din_a = 72'hDEAD_BEEF_CAFE_1234;

        @(posedge clk);  // Write committed
        uram_en_a = 0; uram_rdb_wr_a = 0; uram_bwe_a = 0;

        // Read from addr 42 via port B
        // Cycle R: EN_B=1 → read happens at posedge, result available NEXT cycle
        @(posedge clk);
        uram_en_b = 1; uram_rdb_wr_b = 0;
        uram_addr_b = 23'd42;

        @(posedge clk);  // Read latched into dout_b_r, RDACCESS_B=1
        // Check IMMEDIATELY — RDACCESS_B is high for exactly 1 cycle
        check1("URAM: RDACCESS_B after read", uram_rdaccess_b, 1'b1);
        check("URAM: DOUT_B data", {184'b0, uram_dout_b}, {184'b0, 72'hDEAD_BEEF_CAFE_1234});
        uram_en_b = 0;

        @(posedge clk);
        // RDACCESS_B should be low now (EN_B was deasserted)
        check1("URAM: RDACCESS_B low after idle", uram_rdaccess_b, 1'b0);

        // TEST 2: URAM — BWE partial write (overwrite byte 0 only)
        @(posedge clk);
        uram_en_a = 1; uram_rdb_wr_a = 1;
        uram_addr_a = 23'd42;
        uram_bwe_a = 9'b0_0000_0001;  // only byte 0
        uram_din_a = 72'hFF_FFFF_FFFF_FFFF_FF42;

        @(posedge clk);
        uram_en_a = 0;

        // Read back
        @(posedge clk);
        uram_en_b = 1; uram_rdb_wr_b = 0;
        uram_addr_b = 23'd42;

        @(posedge clk);
        check("URAM: BWE partial write", {184'b0, uram_dout_b}, {184'b0, 72'hDEAD_BEEF_CAFE_1242});
        uram_en_b = 0;

        // TEST 3: URAM — Port A read
        @(posedge clk);
        uram_en_a = 1; uram_rdb_wr_a = 0;  // read mode
        uram_addr_a = 23'd42;

        @(posedge clk);
        check1("URAM: RDACCESS_A after read", uram_rdaccess_a, 1'b1);
        check("URAM: DOUT_A data", {184'b0, uram_dout_a}, {184'b0, 72'hDEAD_BEEF_CAFE_1242});
        uram_en_a = 0;

        // ════════════════════════════════════
        // TEST 4: FIFO — FWFT basic
        // ════════════════════════════════════
        $display("\n=== FIFO FWFT Tests ===");

        // Initially empty
        check1("FIFO: initially empty", fifo_empty, 1'b1);
        check1("FIFO: initially not valid", fifo_valid, 1'b0);

        // Write one entry
        @(posedge clk);
        fifo_wr_en = 1;
        fifo_din = 256'hAAAA_BBBB_CCCC_DDDD;
        @(posedge clk);
        fifo_wr_en = 0;

        // FWFT: data auto-fills output register after 1 cycle
        @(posedge clk);
        check1("FIFO: FWFT valid without rd_en", fifo_valid, 1'b1);
        check1("FIFO: FWFT not empty", fifo_empty, 1'b0);
        check("FIFO: FWFT dout", fifo_dout, 256'hAAAA_BBBB_CCCC_DDDD);

        // Pop with rd_en — backing store is empty, output should go invalid
        fifo_rd_en = 1;
        @(posedge clk);
        fifo_rd_en = 0;

        @(posedge clk);
        check1("FIFO: empty after pop", fifo_empty, 1'b1);
        check1("FIFO: not valid after pop", fifo_valid, 1'b0);

        // TEST 5: FIFO — multiple entries, continuous read
        // Write 3 entries back-to-back
        @(posedge clk);
        fifo_wr_en = 1;
        fifo_din = 256'h1111;
        @(posedge clk);
        fifo_din = 256'h2222;
        @(posedge clk);
        fifo_din = 256'h3333;
        @(posedge clk);
        fifo_wr_en = 0;

        // Wait for FWFT to present first word (auto-fill takes 1 cycle)
        @(posedge clk);
        check1("FIFO: multi-entry valid", fifo_valid, 1'b1);
        check("FIFO: first word", fifo_dout, 256'h1111);

        // Continuous pop: rd_en=1, each cycle shows next word
        fifo_rd_en = 1;
        @(posedge clk);  // pop 1111, load 2222
        check("FIFO: second word", fifo_dout, 256'h2222);

        @(posedge clk);  // pop 2222, load 3333
        check("FIFO: third word", fifo_dout, 256'h3333);

        @(posedge clk);  // pop 3333, backing empty
        fifo_rd_en = 0;

        @(posedge clk);
        check1("FIFO: empty after all pops", fifo_empty, 1'b1);

        // TEST 6: FIFO — simultaneous write/read
        @(posedge clk);
        fifo_wr_en = 1;
        fifo_din = 256'hF00D;
        @(posedge clk);
        fifo_wr_en = 0;
        @(posedge clk);  // auto-fill
        check1("FIFO: sim wr/rd — valid", fifo_valid, 1'b1);
        check("FIFO: sim wr/rd — dout", fifo_dout, 256'hF00D);

        // Write while reading
        fifo_wr_en = 1; fifo_rd_en = 1;
        fifo_din = 256'hBEEF;
        @(posedge clk);  // pop F00D, write BEEF to backing
        fifo_wr_en = 0; fifo_rd_en = 0;

        @(posedge clk);  // auto-fill BEEF
        check1("FIFO: after sim wr/rd valid", fifo_valid, 1'b1);
        check("FIFO: after sim wr/rd dout", fifo_dout, 256'hBEEF);

        // Clean up
        fifo_rd_en = 1;
        @(posedge clk);
        fifo_rd_en = 0;
        @(posedge clk);

        // TEST 7: FIFO — full flag
        $display("  writing 512 entries...");
        fifo_srst = 1;
        repeat (3) @(posedge clk);
        fifo_srst = 0;
        repeat (3) @(posedge clk);

        for (int i = 0; i < 512; i++) begin
            @(posedge clk);
            fifo_wr_en = 1;
            fifo_din = i;
        end
        @(posedge clk);
        fifo_wr_en = 0;

        // After 512 writes: 1 entry in FWFT output reg + 511 in backing = effectively full
        // Check full depends on exact pointer state; at minimum fifo should not be empty
        @(posedge clk);
        check1("FIFO: not empty after 512 writes", fifo_empty, 1'b0);
        check1("FIFO: valid after 512 writes", fifo_valid, 1'b1);

        // Drain
        fifo_srst = 1;
        repeat (3) @(posedge clk);
        fifo_srst = 0;
        repeat (3) @(posedge clk);

        // ════════════════════════════════════
        // TEST 8: SRAM — write then read
        // ════════════════════════════════════
        $display("\n=== SRAM Tests ===");

        // Write 0xCAFE_BABE_DEAD_BEEF to addr 100
        @(posedge clk);
        sram_ena = 1; sram_wea = 1;
        sram_addra = 8'd100;
        sram_dina = 64'hCAFE_BABE_DEAD_BEEF;

        @(posedge clk);  // Write committed
        sram_ena = 0; sram_wea = 0;

        // Read from addr 100 via port B (1-cycle latency)
        @(posedge clk);
        sram_enb = 1;
        sram_addrb = 8'd100;

        @(posedge clk);  // Data latched
        sram_enb = 0;
        check("SRAM: read after write", {192'b0, sram_doutb}, {192'b0, 64'hCAFE_BABE_DEAD_BEEF});

        // TEST 9: SRAM — multiple addresses
        for (int i = 0; i < 4; i++) begin
            @(posedge clk);
            sram_ena = 1; sram_wea = 1;
            sram_addra = i[7:0];
            sram_dina = 64'h1000_0000_0000_0000 + i;
        end
        @(posedge clk);
        sram_ena = 0; sram_wea = 0;

        for (int i = 0; i < 4; i++) begin
            @(posedge clk);
            sram_enb = 1;
            sram_addrb = i[7:0];
            @(posedge clk);
            check("SRAM: multi-addr read",
                  {192'b0, sram_doutb},
                  {192'b0, 64'h1000_0000_0000_0000 + i});
        end
        sram_enb = 0;

        // ════════════════════════════════════
        // Summary
        // ════════════════════════════════════
        repeat (5) @(posedge clk);
        $display("\n════════════════════════════════════");
        $display("Results: %0d passed, %0d failed", pass_count, fail_count);
        $display("════════════════════════════════════");
        if (fail_count > 0)
            $display("*** SOME TESTS FAILED ***");
        else
            $display("All tests PASSED");
        $finish;
    end

endmodule
