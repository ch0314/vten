// vten_axilite_if.sv — Parameterized AXI4-Lite interface
// Naming follows Xilinx VLNV: xilinx.com:interface:aximm_rtl:1.0 (lite subset)

interface vten_axilite_if #(
    parameter int ADDR_W = 32,
    parameter int DATA_W = 32
);
    // Write address
    logic [ADDR_W-1:0]   awaddr;
    logic                awvalid;
    logic                awready;
    // Write data
    logic [DATA_W-1:0]   wdata;
    logic [DATA_W/8-1:0] wstrb;
    logic                wvalid;
    logic                wready;
    // Write response
    logic [1:0]          bresp;
    logic                bvalid;
    logic                bready;
    // Read address
    logic [ADDR_W-1:0]   araddr;
    logic                arvalid;
    logic                arready;
    // Read data
    logic [DATA_W-1:0]   rdata;
    logic [1:0]          rresp;
    logic                rvalid;
    logic                rready;

    modport master (
        output awaddr, awvalid, input awready,
        output wdata, wstrb, wvalid, input wready,
        input  bresp, bvalid, output bready,
        output araddr, arvalid, input arready,
        input  rdata, rresp, rvalid, output rready
    );
    modport slave (
        input  awaddr, awvalid, output awready,
        input  wdata, wstrb, wvalid, output wready,
        output bresp, bvalid, input bready,
        input  araddr, arvalid, output arready,
        output rdata, rresp, rvalid, input rready
    );
endinterface
