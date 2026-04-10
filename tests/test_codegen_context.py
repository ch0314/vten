"""Phase 4 tests: Codegen context — dataclasses and _build_context().

Spec references:
- 06_codegen_and_cli.md §2 (Template Context Schema)
- 06_codegen_and_cli.md §3 (Code Generator)
- specs/npu_3d_analysis.md (NPU 3D realistic patterns)
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from vten.runtime.ir import BFMConfig
from vten.spec.models import (
    InterfaceSpec,
    KernelSpec,
    MemoryRegion,
    PackingScheme,
    Protocol,
    RegisterSpec,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _passthrough_spec() -> KernelSpec:
    """Minimal AXI4-Stream passthrough: 2 interfaces."""
    return KernelSpec(
        kernel_name="passthrough",
        rtl_top="rtl/passthrough.sv",
        parameters={"SIZE": "${SIZE}"},
        interfaces={
            "axi_stream_in": InterfaceSpec(
                name="axi_stream_in",
                rtl_port="s_axis_in",
                protocol=Protocol.AXI4S,
                tensor="data_in",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "axi_stream_out": InterfaceSpec(
                name="axi_stream_out",
                rtl_port="m_axis_out",
                protocol=Protocol.AXI4S,
                tensor="data_out",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
        },
    )


def _passthrough_bfm_configs() -> list[BFMConfig]:
    return [
        BFMConfig(
            interface_name="axi_stream_in",
            protocol=Protocol.AXI4S,
            data_width=32,
            role="master",
        ),
        BFMConfig(
            interface_name="axi_stream_out",
            protocol=Protocol.AXI4S,
            data_width=32,
            role="slave",
        ),
    ]


def _fmapio_spec() -> KernelSpec:
    """fmapIO: AXI4-Lite ctrl + AXI4 DDR + 2 AXIS internal."""
    return KernelSpec(
        kernel_name="fmapIO",
        rtl_top="design/fmapIO/rtl/fmapIO_top.sv",
        parameters={"IN_CH": "${IN_CH}"},
        memory_regions={
            "ddr": MemoryRegion(name="ddr", base=0, size=0x1_0000_0000),
        },
        interfaces={
            "ctrl": InterfaceSpec(
                name="ctrl",
                rtl_port="s_axilite_ctrl",
                protocol=Protocol.AXI4L,
                addr_width=16,
                registers=[
                    RegisterSpec(name="in_depth", offset=0x014),
                    RegisterSpec(name="vsync", offset=0x050),
                    RegisterSpec(name="layer_done", offset=0x054),
                ],
            ),
            "ddr": InterfaceSpec(
                name="ddr",
                rtl_port="m_axi_ddr",
                protocol=Protocol.AXI4,
                data_width=256,
                addr_width=64,
                memory_region="ddr",
                tensors=["ifm", "ofm", "concat"],
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
            "ifm_out": InterfaceSpec(
                name="ifm_out",
                rtl_port="m_axis_ifm",
                protocol=Protocol.AXI4S,
                tensor="ifm",
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
            "ofm_in": InterfaceSpec(
                name="ofm_in",
                rtl_port="s_axis_ofm",
                protocol=Protocol.AXI4S,
                tensor="ofm",
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
        },
    )


def _fmapio_bfm_configs() -> list[BFMConfig]:
    """fmapIO external BFMs: 1 AXI4-Lite + 1 AXI4 + 2 AXI4-Stream."""
    return [
        BFMConfig(interface_name="ctrl", protocol=Protocol.AXI4L, data_width=32, addr_width=16, role="master"),
        BFMConfig(interface_name="ddr", protocol=Protocol.AXI4, data_width=256, addr_width=64, role="slave"),
        BFMConfig(interface_name="ifm_out", protocol=Protocol.AXI4S, data_width=256, role="master"),
        BFMConfig(interface_name="ofm_in", protocol=Protocol.AXI4S, data_width=256, role="slave"),
    ]


def _npu_40_bfm_configs() -> list[BFMConfig]:
    """NPU 3D: 40 BFMs (6 AXI4-Lite + 2 DDR AXI4 + 32 HBM AXI4)."""
    cfgs: list[BFMConfig] = []

    # 6 AXI4-Lite master (per-IP control)
    for name in ["ctrl_fmapio", "ctrl_wgt", "ctrl_mac", "ctrl_psum", "ctrl_bias", "ctrl_act"]:
        cfgs.append(BFMConfig(
            interface_name=name, protocol=Protocol.AXI4L,
            data_width=32, addr_width=16, role="master",
        ))

    # 2 DDR AXI4 slave
    for name in ["ddr_fmap", "ddr_bias"]:
        cfgs.append(BFMConfig(
            interface_name=name, protocol=Protocol.AXI4,
            data_width=256, addr_width=64, role="slave",
        ))

    # 32 HBM AXI4 slave
    for i in range(32):
        cfgs.append(BFMConfig(
            interface_name=f"hbm_{i:02d}", protocol=Protocol.AXI4,
            data_width=256, addr_width=64, role="slave",
        ))

    assert len(cfgs) == 40
    return cfgs


def _minimal_project_config() -> dict:
    return {
        "project": {"name": "test_proj", "version": "0.1.0"},
        "rtl": {
            "sources": ["rtl/**/*.sv"],
            "top_module": "passthrough",
            "include_dirs": [],
        },
        "backend": {
            "xsim": {
                "vivado_path": "/tools/Xilinx/Vivado/2023.2",
                "compile_options": ["-timescale", "1ns/1ps"],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════
# §1  TestbenchContext dataclass
# ═══════════════════════════════════════════════════════════════════


class TestTestbenchContext:
    """TestbenchContext: 06_codegen_and_cli.md §2.1."""

    def test_construction_all_fields(self):
        from vten.codegen.sv_generator import BFMInstance, DUTPort, TestbenchContext

        ctx = TestbenchContext(
            project_name="my_npu",
            top_module="npu_top",
            session_id="test123",
            dut_ports=[DUTPort(name="clk", direction="input", width=1, connected_to="clk")],
            bfms=[],
        )
        assert ctx.project_name == "my_npu"
        assert ctx.top_module == "npu_top"
        assert ctx.session_id == "test123"

    def test_defaults(self):
        from vten.codegen.sv_generator import TestbenchContext

        ctx = TestbenchContext(
            project_name="p", top_module="m", session_id="s",
            dut_ports=[], bfms=[],
        )
        assert ctx.clock_name == "clk"
        assert ctx.reset_name == "rst_n"
        assert ctx.reset_active_low is True
        assert ctx.clock_period_ns == 10.0

    def test_custom_clock_reset(self):
        from vten.codegen.sv_generator import TestbenchContext

        ctx = TestbenchContext(
            project_name="p", top_module="m", session_id="s",
            dut_ports=[], bfms=[],
            clock_name="ap_clk", reset_name="ap_aresetn",
            reset_active_low=True,
        )
        assert ctx.clock_name == "ap_clk"
        assert ctx.reset_name == "ap_aresetn"

    def test_active_high_reset(self):
        from vten.codegen.sv_generator import TestbenchContext

        ctx = TestbenchContext(
            project_name="p", top_module="m", session_id="s",
            dut_ports=[], bfms=[],
            reset_active_low=False,
        )
        assert ctx.reset_active_low is False

    def test_is_dataclass(self):
        from vten.codegen.sv_generator import TestbenchContext

        assert hasattr(TestbenchContext, "__dataclass_fields__")


# ═══════════════════════════════════════════════════════════════════
# §2  DUTPort dataclass
# ═══════════════════════════════════════════════════════════════════


class TestDUTPort:
    """DUTPort: 06_codegen_and_cli.md §2.1."""

    def test_construction(self):
        from vten.codegen.sv_generator import DUTPort

        p = DUTPort(name="clk", direction="input", width=1, connected_to="clk")
        assert p.name == "clk"
        assert p.direction == "input"
        assert p.width == 1

    def test_direction_input(self):
        from vten.codegen.sv_generator import DUTPort

        p = DUTPort(name="rst_n", direction="input", width=1, connected_to="rst_n")
        assert p.direction == "input"

    def test_direction_output(self):
        from vten.codegen.sv_generator import DUTPort

        p = DUTPort(name="m_axis_tdata", direction="output", width=256, connected_to="bfm_wire")
        assert p.direction == "output"

    def test_direction_inout(self):
        from vten.codegen.sv_generator import DUTPort

        p = DUTPort(name="sda", direction="inout", width=1, connected_to="i2c_sda")
        assert p.direction == "inout"

    def test_wide_bus(self):
        from vten.codegen.sv_generator import DUTPort

        p = DUTPort(name="m_axi_rdata", direction="input", width=256, connected_to="bfm_rdata")
        assert p.width == 256

    def test_is_dataclass(self):
        from vten.codegen.sv_generator import DUTPort

        assert hasattr(DUTPort, "__dataclass_fields__")


# ═══════════════════════════════════════════════════════════════════
# §3  BFMInstance dataclass
# ═══════════════════════════════════════════════════════════════════


class TestBFMInstance:
    """BFMInstance: 06_codegen_and_cli.md §2.1."""

    def test_axi4s_instance(self):
        from vten.codegen.sv_generator import BFMInstance

        b = BFMInstance(
            name="bfm_axi_stream_in",
            module_name="vten_bfm_axi4s",
            protocol="axi4_stream",
            data_width=32,
            role="master",
            rtl_port_prefix="s_axis_in",
            parameters={"DATA_W": 32},
            interface_id=0,
        )
        assert b.module_name == "vten_bfm_axi4s"
        assert b.protocol == "axi4_stream"
        assert b.role == "master"

    def test_axi4_instance(self):
        from vten.codegen.sv_generator import BFMInstance

        b = BFMInstance(
            name="bfm_ddr",
            module_name="vten_bfm_axi4",
            protocol="axi4",
            data_width=256,
            role="slave",
            rtl_port_prefix="m_axi_ddr",
            parameters={"DATA_W": 256, "ADDR_W": 64},
            interface_id=1,
        )
        assert b.module_name == "vten_bfm_axi4"
        assert b.data_width == 256
        assert b.role == "slave"

    def test_axilite_instance(self):
        from vten.codegen.sv_generator import BFMInstance

        b = BFMInstance(
            name="bfm_ctrl",
            module_name="vten_bfm_axilite",
            protocol="axi4_lite",
            data_width=32,
            role="master",
            rtl_port_prefix="s_axilite_ctrl",
            parameters={"DATA_W": 32},
            interface_id=2,
        )
        assert b.module_name == "vten_bfm_axilite"
        assert b.role == "master"

    def test_parameters_dict(self):
        from vten.codegen.sv_generator import BFMInstance

        b = BFMInstance(
            name="bfm_test", module_name="vten_bfm_axi4",
            protocol="axi4", data_width=256, role="slave",
            rtl_port_prefix="m_axi", parameters={"DATA_W": 256, "ADDR_W": 64},
            interface_id=0,
        )
        assert b.parameters["DATA_W"] == 256
        assert b.parameters["ADDR_W"] == 64

    def test_interface_id_is_int(self):
        from vten.codegen.sv_generator import BFMInstance

        b = BFMInstance(
            name="bfm_test", module_name="vten_bfm_axi4s",
            protocol="axi4_stream", data_width=32, role="master",
            rtl_port_prefix="s_axis", parameters={}, interface_id=5,
        )
        assert isinstance(b.interface_id, int)
        assert b.interface_id == 5

    def test_is_dataclass(self):
        from vten.codegen.sv_generator import BFMInstance

        assert hasattr(BFMInstance, "__dataclass_fields__")


# ═══════════════════════════════════════════════════════════════════
# §4  BuildContext dataclass
# ═══════════════════════════════════════════════════════════════════


class TestBuildContext:
    """BuildContext: 06_codegen_and_cli.md §2.2."""

    def test_construction(self):
        from vten.codegen.sv_generator import BuildContext

        ctx = BuildContext(
            vivado_path="/tools/Xilinx/Vivado/2023.2",
            rtl_sources=["rtl/passthrough.sv"],
            include_dirs=[],
            generated_sv=["build/generated/tb_top.sv"],
            vten_sv_dir="/home/user/vten/vten_sv",
            dpi_c_source="/home/user/vten/vten_sv/vten_shm_bridge.c",
            compile_options=["-timescale", "1ns/1ps"],
        )
        assert ctx.vivado_path == "/tools/Xilinx/Vivado/2023.2"
        assert len(ctx.rtl_sources) == 1

    def test_defaults(self):
        from vten.codegen.sv_generator import BuildContext

        ctx = BuildContext(
            vivado_path="/v", rtl_sources=[], include_dirs=[],
            generated_sv=[], vten_sv_dir="/sv", dpi_c_source="/c.c",
            compile_options=[],
        )
        assert ctx.timescale == "1ns/1ps"
        assert ctx.top_module == "tb_top"

    def test_custom_top_module(self):
        from vten.codegen.sv_generator import BuildContext

        ctx = BuildContext(
            vivado_path="/v", rtl_sources=[], include_dirs=[],
            generated_sv=[], vten_sv_dir="/sv", dpi_c_source="/c.c",
            compile_options=[], top_module="my_tb",
        )
        assert ctx.top_module == "my_tb"

    def test_multiple_rtl_sources(self):
        from vten.codegen.sv_generator import BuildContext

        ctx = BuildContext(
            vivado_path="/v",
            rtl_sources=["rtl/a.sv", "rtl/b.sv", "rtl/c.v"],
            include_dirs=["rtl/include"],
            generated_sv=[], vten_sv_dir="/sv", dpi_c_source="/c.c",
            compile_options=[],
        )
        assert len(ctx.rtl_sources) == 3
        assert len(ctx.include_dirs) == 1

    def test_is_dataclass(self):
        from vten.codegen.sv_generator import BuildContext

        assert hasattr(BuildContext, "__dataclass_fields__")


# ═══════════════════════════════════════════════════════════════════
# §5  SVGenerator._build_context() — KernelSpec + BFMConfig[] → Context
# ═══════════════════════════════════════════════════════════════════


class TestSVGeneratorBuildContext:
    """SVGenerator._build_context() transforms spec + BFMs into template context."""

    def test_passthrough_2bfm_instance_count(self):
        """Passthrough: 2 AXI4-Stream BFMs."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        assert len(ctx.tb.bfms) == 2

    def test_passthrough_bfm_module_names(self):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        module_names = {b.module_name for b in ctx.tb.bfms}
        assert module_names == {"vten_bfm_axi4s"}

    def test_fmapio_4bfm_instance_count(self):
        """fmapIO: 1 AXI4-Lite + 1 AXI4 + 2 AXI4-Stream = 4 BFMs."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_fmapio_spec(),
            bfm_configs=_fmapio_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        assert len(ctx.tb.bfms) == 4

    def test_fmapio_module_names(self):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_fmapio_spec(),
            bfm_configs=_fmapio_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        module_names = {b.module_name for b in ctx.tb.bfms}
        assert "vten_bfm_axilite" in module_names
        assert "vten_bfm_axi4" in module_names
        assert "vten_bfm_axi4s" in module_names

    def test_fmapio_rtl_port_prefix(self):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_fmapio_spec(),
            bfm_configs=_fmapio_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        prefixes = {b.rtl_port_prefix for b in ctx.tb.bfms}
        # No generate_controller → uses rtl_port names directly
        assert "s_axilite_ctrl" in prefixes
        assert "m_axi_ddr" in prefixes

    def test_module_for_protocol_axi4s(self):
        """Protocol.AXI4S → 'vten_bfm_axi4s'."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=[], project_config=_minimal_project_config(),
        )
        assert gen._module_for_protocol(Protocol.AXI4S) == "vten_bfm_axi4s"

    def test_module_for_protocol_axi4(self):
        """Protocol.AXI4 → 'vten_bfm_axi4'."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=[], project_config=_minimal_project_config(),
        )
        assert gen._module_for_protocol(Protocol.AXI4) == "vten_bfm_axi4"

    def test_module_for_protocol_axilite(self):
        """Protocol.AXI4L → 'vten_bfm_axilite'."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=[], project_config=_minimal_project_config(),
        )
        assert gen._module_for_protocol(Protocol.AXI4L) == "vten_bfm_axilite"

    def test_bfm_interface_id_sequential(self):
        """BFMInstance.interface_id follows BFMConfig list ordering."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        ids = [b.interface_id for b in ctx.tb.bfms]
        assert ids == [0, 1]

    def test_bfm_name_format(self):
        """BFM instance name: 'bfm_{interface_name}'."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        names = [b.name for b in ctx.tb.bfms]
        assert "bfm_axi_stream_in" in names
        assert "bfm_axi_stream_out" in names

    def test_bfm_data_width_parameter(self):
        """BFMInstance.parameters includes DATA_W from BFMConfig.data_width."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_fmapio_spec(),
            bfm_configs=_fmapio_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        # Find the AXI4 DDR BFM
        ddr_bfm = next(b for b in ctx.tb.bfms if b.name == "bfm_ddr")
        assert ddr_bfm.parameters["DATA_W"] == 256

    def test_context_has_tb_and_build(self):
        """_build_context returns object with .tb (TestbenchContext) and .build (BuildContext)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        assert hasattr(ctx, "tb")
        assert hasattr(ctx, "build")

    def test_context_top_module_from_spec(self):
        """TestbenchContext.top_module comes from KernelSpec or config."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()
        # top_module should match the RTL top from config or spec
        assert ctx.tb.top_module is not None

    def test_npu_40bfm_instance_count(self):
        """NPU 3D: 40 BFMs correctly instantiated."""
        from vten.codegen.sv_generator import SVGenerator

        # Create a minimal spec with all 40 interfaces
        interfaces = {}
        for cfg in _npu_40_bfm_configs():
            proto = cfg.protocol
            interfaces[cfg.interface_name] = InterfaceSpec(
                name=cfg.interface_name,
                rtl_port=f"port_{cfg.interface_name}",
                protocol=proto,
                data_width=cfg.data_width,
                addr_width=cfg.addr_width,
            )

        spec = KernelSpec(
            kernel_name="npu_3d",
            rtl_top="design/NPU_3D_top.sv",
            interfaces=interfaces,
        )
        config = _minimal_project_config()
        config["rtl"]["top_module"] = "NPU_3D_top"

        gen = SVGenerator(
            kernel_spec=spec,
            bfm_configs=_npu_40_bfm_configs(),
            project_config=config,
        )
        ctx = gen._build_context()
        assert len(ctx.tb.bfms) == 40

    def test_npu_40bfm_module_distribution(self):
        """NPU 3D: 6 axilite + 34 axi4 modules."""
        from vten.codegen.sv_generator import SVGenerator

        interfaces = {}
        for cfg in _npu_40_bfm_configs():
            interfaces[cfg.interface_name] = InterfaceSpec(
                name=cfg.interface_name,
                rtl_port=f"port_{cfg.interface_name}",
                protocol=cfg.protocol,
                data_width=cfg.data_width,
                addr_width=cfg.addr_width,
            )

        spec = KernelSpec(
            kernel_name="npu_3d",
            rtl_top="design/NPU_3D_top.sv",
            interfaces=interfaces,
        )

        gen = SVGenerator(
            kernel_spec=spec,
            bfm_configs=_npu_40_bfm_configs(),
            project_config=_minimal_project_config(),
        )
        ctx = gen._build_context()

        axilite_count = sum(1 for b in ctx.tb.bfms if b.module_name == "vten_bfm_axilite")
        axi4_count = sum(1 for b in ctx.tb.bfms if b.module_name == "vten_bfm_axi4")
        assert axilite_count == 6
        assert axi4_count == 34  # 2 DDR + 32 HBM
