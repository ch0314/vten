// vten_aximm_if.sv — Parameterized AXI4 Memory-Mapped interface
// Naming follows Xilinx VLNV: xilinx.com:interface:aximm_rtl:1.0

interface vten_aximm_if #(
    parameter int DATA_W = 256,
    parameter int ADDR_W = 64
);
    // AW channel
    logic [ADDR_W-1:0]   awaddr;
    logic [7:0]          awlen;
    logic [2:0]          awsize;
    logic [1:0]          awburst;
    logic                awvalid;
    logic                awready;
    // W channel
    logic [DATA_W-1:0]   wdata;
    logic [DATA_W/8-1:0] wstrb;
    logic                wlast;
    logic                wvalid;
    logic                wready;
    // B channel
    logic [1:0]          bresp;
    logic                bvalid;
    logic                bready;
    // AR channel
    logic [ADDR_W-1:0]   araddr;
    logic [7:0]          arlen;
    logic [2:0]          arsize;
    logic [1:0]          arburst;
    logic                arvalid;
    logic                arready;
    // R channel
    logic [DATA_W-1:0]   rdata;
    logic [1:0]          rresp;
    logic                rlast;
    logic                rvalid;
    logic                rready;

    modport master (
        output awaddr, awlen, awsize, awburst, awvalid, input awready,
        output wdata, wstrb, wlast, wvalid, input wready,
        input  bresp, bvalid, output bready,
        output araddr, arlen, arsize, arburst, arvalid, input arready,
        input  rdata, rresp, rlast, rvalid, output rready
    );
    modport slave (
        input  awaddr, awlen, awsize, awburst, awvalid, output awready,
        input  wdata, wstrb, wlast, wvalid, output wready,
        output bresp, bvalid, input bready,
        input  araddr, arlen, arsize, arburst, arvalid, output arready,
        output rdata, rresp, rlast, rvalid, input rready
    );
endinterface
