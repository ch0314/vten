// vten_axis_if.sv — Parameterized AXI4-Stream interface
// Naming follows Xilinx VLNV: xilinx.com:interface:axis_rtl:1.0

interface vten_axis_if #(
    parameter int DATA_W = 256
);
    logic [DATA_W-1:0] tdata;
    logic              tvalid;
    logic              tready;
    logic              tlast;

    modport master (
        output tdata, tvalid, tlast,
        input  tready
    );
    modport slave (
        input  tdata, tvalid, tlast,
        output tready
    );
endinterface
