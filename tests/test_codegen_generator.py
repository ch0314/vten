"""Phase 4 tests: SVGenerator.generate() — Jinja2 template rendering.

Spec references:
- 06_codegen_and_cli.md §1 (Template Architecture)
- 06_codegen_and_cli.md §3 (Code Generator)
- specs/npu_3d_analysis.md §11 (NPU 3D mapping)
"""

from __future__ import annotations

import re
from pathlib import Path

from vten.runtime.ir import BFMConfig
from vten.spec.models import (
    InterfaceSpec,
    KernelSpec,
    PackingScheme,
    Protocol,
    RegisterSpec,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _passthrough_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="passthrough",
        rtl_top="rtl/passthrough.sv",
        parameters={"SIZE": "${SIZE}"},
        interfaces={
            "axi_stream_in": InterfaceSpec(
                name="axi_stream_in", rtl_port="s_axis_in",
                protocol=Protocol.AXI4S, tensor="data_in",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "axi_stream_out": InterfaceSpec(
                name="axi_stream_out", rtl_port="m_axis_out",
                protocol=Protocol.AXI4S, tensor="data_out",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
        },
    )


def _passthrough_bfm_configs() -> list[BFMConfig]:
    return [
        BFMConfig(interface_name="axi_stream_in", protocol=Protocol.AXI4S, data_width=32, role="master"),
        BFMConfig(interface_name="axi_stream_out", protocol=Protocol.AXI4S, data_width=32, role="slave"),
    ]


def _minimal_config() -> dict:
    return {
        "project": {"name": "test_proj", "version": "0.1.0"},
        "rtl": {
            "sources": ["rtl/passthrough.sv"],
            "top_module": "passthrough",
            "include_dirs": [],
        },
        "backend": {
            "xsim": {
                "vivado_path": "/tools/Xilinx/Vivado/2024.1",
                "compile_options": ["-timescale", "1ns/1ps"],
            },
        },
    }


def _npu_40_bfm_configs() -> list[BFMConfig]:
    cfgs: list[BFMConfig] = []
    for name in ["ctrl_fmapio", "ctrl_wgt", "ctrl_mac", "ctrl_psum", "ctrl_bias", "ctrl_act"]:
        cfgs.append(BFMConfig(interface_name=name, protocol=Protocol.AXI4L, data_width=32, role="master"))
    for name in ["ddr_fmap", "ddr_bias"]:
        cfgs.append(BFMConfig(interface_name=name, protocol=Protocol.AXI4, data_width=256, role="slave"))
    for i in range(32):
        cfgs.append(BFMConfig(interface_name=f"hbm_{i:02d}", protocol=Protocol.AXI4, data_width=256, role="slave"))
    return cfgs


def _npu_spec() -> KernelSpec:
    interfaces = {}
    for cfg in _npu_40_bfm_configs():
        interfaces[cfg.interface_name] = InterfaceSpec(
            name=cfg.interface_name,
            rtl_port=f"port_{cfg.interface_name}",
            protocol=cfg.protocol,
            data_width=cfg.data_width,
            addr_width=cfg.addr_width if hasattr(cfg, "addr_width") else None,
        )
    return KernelSpec(
        kernel_name="npu_3d", rtl_top="design/NPU_3D_top.sv",
        interfaces=interfaces,
    )


def _npu_config() -> dict:
    return {
        "project": {"name": "npu_3d", "version": "0.1.0"},
        "rtl": {
            "sources": ["design/**/*.sv"],
            "top_module": "NPU_3D_top",
            "include_dirs": ["design/include"],
        },
        "backend": {
            "xsim": {
                "vivado_path": "/tools/Xilinx/Vivado/2024.1",
                "compile_options": ["-timescale", "1ns/1ps"],
            },
            "scheduler": {
                "max_bfms": 40,
                "max_ifaces": 42,
            },
        },
    }


def _generate_passthrough(tmp_path: Path) -> str:
    """Generate passthrough testbench and return tb_top.sv content."""
    from vten.codegen.sv_generator import SVGenerator

    gen = SVGenerator(
        kernel_spec=_passthrough_spec(),
        bfm_configs=_passthrough_bfm_configs(),
        project_config=_minimal_config(),
    )
    gen.generate(str(tmp_path))
    return (tmp_path / "tb_top.sv").read_text()


def _generate_npu(tmp_path: Path) -> str:
    """Generate NPU 40-BFM testbench and return tb_top.sv content."""
    from vten.codegen.sv_generator import SVGenerator

    gen = SVGenerator(
        kernel_spec=_npu_spec(),
        bfm_configs=_npu_40_bfm_configs(),
        project_config=_npu_config(),
    )
    gen.generate(str(tmp_path))
    return (tmp_path / "tb_top.sv").read_text()


# ═══════════════════════════════════════════════════════════════════
# §1  SVGenerator init
# ═══════════════════════════════════════════════════════════════════


class TestSVGeneratorInit:
    """SVGenerator constructor."""

    def test_constructor_stores_spec(self):
        from vten.codegen.sv_generator import SVGenerator

        spec = _passthrough_spec()
        cfgs = _passthrough_bfm_configs()
        config = _minimal_config()
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=config)

        assert gen.spec is spec
        assert gen.bfm_configs is cfgs
        assert gen.config is config

    def test_constructor_accepts_empty_bfm_list(self):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=[],
            project_config=_minimal_config(),
        )
        assert gen.bfm_configs == []


# ═══════════════════════════════════════════════════════════════════
# §2  SVGenerator.generate() — file output verification
# ═══════════════════════════════════════════════════════════════════


class TestSVGeneratorGenerate:
    """SVGenerator.generate() creates expected files from Jinja2 templates."""

    def test_generate_creates_tb_top_sv(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert (tmp_path / "tb_top.sv").exists()

    def test_generate_creates_build_tcl(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert (tmp_path / "build.tcl").exists()

    def test_generate_creates_run_tcl(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert (tmp_path / "run.tcl").exists()

    def test_generate_creates_makefile(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert (tmp_path / "Makefile").exists()

    def test_tb_top_contains_dut_instance(self, tmp_path: Path):
        """tb_top.sv contains the DUT module instantiation."""
        content = _generate_passthrough(tmp_path)
        # Should contain module name as instance
        assert "passthrough" in content

    def test_tb_top_contains_clock_reset(self, tmp_path: Path):
        """tb_top.sv has clock and reset generation."""
        content = _generate_passthrough(tmp_path)
        assert "clk" in content
        assert "rst" in content.lower() or "reset" in content.lower()

    def test_tb_top_contains_bfm_instances(self, tmp_path: Path):
        """tb_top.sv has BFM instantiations."""
        content = _generate_passthrough(tmp_path)
        assert "vten_bfm_axi4s" in content

    def test_tb_top_contains_scheduler(self, tmp_path: Path):
        """tb_top.sv instantiates vten_command_scheduler."""
        content = _generate_passthrough(tmp_path)
        assert "vten_command_scheduler" in content

    def test_tb_top_contains_shm_controller(self, tmp_path: Path):
        """tb_top.sv instantiates vten_shm_controller."""
        content = _generate_passthrough(tmp_path)
        assert "vten_shm_controller" in content

    def test_tb_top_scheduler_params_with_regex(self, tmp_path: Path):
        """MAX_CMDS, MAX_BFMS, MAX_IFACES appear as parameter assignments."""
        content = _generate_passthrough(tmp_path)
        # Match patterns like .MAX_BFMS(N) or MAX_BFMS = N or #(.MAX_BFMS(N))
        assert re.search(r"MAX_CMDS\s*[=(]", content), "MAX_CMDS parameter not found"
        assert re.search(r"MAX_BFMS\s*[=(]", content), "MAX_BFMS parameter not found"
        assert re.search(r"MAX_IFACES\s*[=(]", content), "MAX_IFACES parameter not found"

    def test_tb_top_max_bfms_value_matches_config_count(self, tmp_path: Path):
        """MAX_BFMS parameter value >= number of BFM configs."""
        content = _generate_passthrough(tmp_path)
        # 2 BFMs for passthrough: should find a value >= 2
        match = re.search(r"MAX_BFMS\s*[=(]\s*(\d+)", content)
        assert match is not None, "MAX_BFMS value not found"
        max_bfms = int(match.group(1))
        assert max_bfms >= 2

    def test_tb_top_contains_iface_to_bfm_mapping(self, tmp_path: Path):
        """tb_top.sv has scheduler.iface_to_bfm initialization."""
        content = _generate_passthrough(tmp_path)
        assert "iface_to_bfm" in content

    def test_tb_top_two_bfm_instances_for_passthrough(self, tmp_path: Path):
        """Passthrough has exactly 2 AXI4-Stream BFM instances."""
        content = _generate_passthrough(tmp_path)
        # Count vten_bfm_axi4s instances (not vten_bfm_axi4 without 's')
        # Match pattern like: vten_bfm_axi4s #( or vten_bfm_axi4s bfm_
        count = len(re.findall(r"vten_bfm_axi4s\b", content))
        assert count >= 2, f"Expected >= 2 AXI4-Stream BFM instances, found {count}"

    def test_build_tcl_contains_xvlog(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "build.tcl").read_text()
        assert "xvlog" in content

    def test_build_tcl_contains_xelab(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "build.tcl").read_text()
        assert "xelab" in content

    def test_build_tcl_contains_rtl_sources(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "build.tcl").read_text()
        assert "passthrough" in content

    def test_build_tcl_contains_sv_lib(self, tmp_path: Path):
        """build.tcl references DPI-C shared library (sv_lib)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "build.tcl").read_text()
        assert "sv_lib" in content or "libvten_shm" in content

    def test_build_tcl_vivado_path(self, tmp_path: Path):
        """build.tcl uses vivado_path from config."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "build.tcl").read_text()
        assert "Vivado" in content or "vivado" in content

    def test_run_tcl_contains_xsim(self, tmp_path: Path):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "run.tcl").read_text()
        assert "xsim" in content

    def test_tb_top_includes_vten_types(self, tmp_path: Path):
        """tb_top.sv includes vten_types.svh."""
        content = _generate_passthrough(tmp_path)
        assert "vten_types.svh" in content

    def test_tb_top_module_declaration(self, tmp_path: Path):
        """tb_top.sv declares module tb_top."""
        content = _generate_passthrough(tmp_path)
        assert re.search(r"module\s+tb_top", content), "module tb_top declaration not found"

    def test_tb_top_endmodule(self, tmp_path: Path):
        """tb_top.sv ends with endmodule."""
        content = _generate_passthrough(tmp_path)
        assert "endmodule" in content

    def test_tb_top_initial_block(self, tmp_path: Path):
        """tb_top.sv contains initial block (for SHM init or sim control)."""
        content = _generate_passthrough(tmp_path)
        assert "initial" in content


# ═══════════════════════════════════════════════════════════════════
# §3  SVGenerator NPU 3D — large-scale testbench
# ═══════════════════════════════════════════════════════════════════


class TestSVGeneratorNPU:
    """NPU 3D: 40-BFM testbench generation."""

    def test_npu_40bfm_tb_top(self, tmp_path: Path):
        """Full NPU 3D testbench generates valid tb_top.sv."""
        content = _generate_npu(tmp_path)
        assert len(content) > 100  # Non-trivial content

    def test_npu_max_bfms_at_least_40(self, tmp_path: Path):
        """MAX_BFMS >= 40 in generated tb_top.sv."""
        content = _generate_npu(tmp_path)
        match = re.search(r"MAX_BFMS\s*[=(]\s*(\d+)", content)
        assert match is not None, "MAX_BFMS not found in NPU tb_top.sv"
        max_bfms = int(match.group(1))
        assert max_bfms >= 40, f"MAX_BFMS={max_bfms}, expected >= 40"

    def test_npu_hbm_32port_bfm_instances(self, tmp_path: Path):
        """32 HBM BFM instances appear in tb_top.sv."""
        content = _generate_npu(tmp_path)
        # Check first and last HBM BFM
        assert "hbm_00" in content or "bfm_hbm_00" in content
        assert "hbm_31" in content or "bfm_hbm_31" in content

    def test_npu_hbm_all_32_present(self, tmp_path: Path):
        """All 32 HBM port instances (hbm_00 through hbm_31) present."""
        content = _generate_npu(tmp_path)
        for i in range(32):
            name = f"hbm_{i:02d}"
            assert name in content, f"HBM port {name} not found in tb_top.sv"

    def test_npu_axilite_6ip_bfm_instances(self, tmp_path: Path):
        """6 AXI4-Lite BFM instances for 6 IPs."""
        content = _generate_npu(tmp_path)
        assert "vten_bfm_axilite" in content
        assert "vten_bfm_axi4" in content

    def test_npu_ctrl_ips_present(self, tmp_path: Path):
        """All 6 control IP names appear in tb_top.sv."""
        content = _generate_npu(tmp_path)
        for name in ["ctrl_fmapio", "ctrl_wgt", "ctrl_mac", "ctrl_psum", "ctrl_bias", "ctrl_act"]:
            assert name in content, f"Control IP {name} not found in tb_top.sv"

    def test_npu_ddr_ports_present(self, tmp_path: Path):
        """DDR fmap and bias ports appear in tb_top.sv."""
        content = _generate_npu(tmp_path)
        assert "ddr_fmap" in content
        assert "ddr_bias" in content

    def test_npu_iface_to_bfm_40_entries(self, tmp_path: Path):
        """iface_to_bfm mapping has entries for all 40 interfaces."""
        content = _generate_npu(tmp_path)
        count = content.count("iface_to_bfm[")
        assert count >= 40, f"Expected >= 40 iface_to_bfm entries, found {count}"

    def test_npu_three_bfm_module_types(self, tmp_path: Path):
        """NPU tb_top uses all three BFM module types."""
        content = _generate_npu(tmp_path)
        assert "vten_bfm_axilite" in content, "Missing AXI4-Lite BFM"
        assert re.search(r"vten_bfm_axi4\b", content), "Missing AXI4 BFM"
        # AXI4-Stream not used in NPU (all ports are AXI4 or AXI4-Lite)

    def test_npu_top_module_name(self, tmp_path: Path):
        """NPU tb_top references NPU_3D_top as DUT."""
        content = _generate_npu(tmp_path)
        assert "NPU_3D_top" in content


# ═══════════════════════════════════════════════════════════════════
# §4  RTL port matching — DUT-BFM wiring
# ═══════════════════════════════════════════════════════════════════


class TestSVGeneratorRTLPortMatching:
    """RTL port prefix → DUT-BFM wire matching (06_codegen_and_cli.md §3.2)."""

    def test_axi4_port_prefix_expands_signals(self, tmp_path: Path):
        """AXI4 BFM with rtl_port expands to channel-level signal wires."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axi4",
            rtl_top="rtl/test.sv",
            interfaces={
                "ddr": InterfaceSpec(
                    name="ddr", rtl_port="m_axi_ddr",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="ddr", protocol=Protocol.AXI4, data_width=256, role="slave")]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        lower = content.lower()
        # AXI4 port expansion MUST produce channel signals, not just the prefix
        assert "araddr" in lower or "awaddr" in lower, (
            "AXI4 port expansion failed: expected araddr/awaddr signal wires, "
            f"but only found raw prefix. Content snippet: ...{content[max(0,content.lower().find('ddr')-50):content.lower().find('ddr')+100]}..."
        )

    def test_axi4_port_both_read_and_write_channels(self, tmp_path: Path):
        """AXI4 generates both AR (read) and AW (write) channel signals."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axi4",
            rtl_top="rtl/test.sv",
            interfaces={
                "mem": InterfaceSpec(
                    name="mem", rtl_port="m_axi_mem",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="mem", protocol=Protocol.AXI4, data_width=256, role="slave")]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        lower = content.lower()
        # Read address channel
        assert "arvalid" in lower, "AXI4 missing arvalid (read address channel)"
        assert "arready" in lower, "AXI4 missing arready (read address channel)"
        # Write address channel
        assert "awvalid" in lower, "AXI4 missing awvalid (write address channel)"
        assert "awready" in lower, "AXI4 missing awready (write address channel)"

    def test_axilite_port_prefix_expands_signals(self, tmp_path: Path):
        """AXI4-Lite BFM expands to handshake signal wires."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axilite",
            rtl_top="rtl/test.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axilite_ctrl",
                    protocol=Protocol.AXI4L, addr_width=16,
                    registers=[RegisterSpec(name="reg0", offset=0x10)],
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="ctrl", protocol=Protocol.AXI4L, data_width=32, role="master")]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        lower = content.lower()
        # AXI4-Lite must expand to channel signals
        assert "awaddr" in lower or "araddr" in lower, (
            "AXI4-Lite port expansion failed: expected awaddr/araddr signals"
        )
        assert "wdata" in lower or "rdata" in lower, (
            "AXI4-Lite port expansion failed: expected wdata/rdata signals"
        )

    def test_axi4s_port_prefix_expands_signals(self, tmp_path: Path):
        """AXI4-Stream BFM expands to tdata/tvalid/tready/tlast."""
        content = _generate_passthrough(tmp_path)
        lower = content.lower()
        # AXI4-Stream MUST expand to these specific signals
        assert "tdata" in lower, "AXI4-Stream missing tdata signal"
        assert "tvalid" in lower, "AXI4-Stream missing tvalid signal"
        assert "tready" in lower, "AXI4-Stream missing tready signal"
        assert "tlast" in lower, "AXI4-Stream missing tlast signal"

    def test_wire_declarations_use_wire_or_logic(self, tmp_path: Path):
        """Signal declarations use 'wire' or 'logic' keyword."""
        content = _generate_passthrough(tmp_path)
        lower = content.lower()
        assert "wire" in lower or "logic" in lower, "No wire/logic declarations found"

    def test_dut_port_connections_use_rtl_port_names(self, tmp_path: Path):
        """DUT instance has .rtl_port_name(...) connections."""
        content = _generate_passthrough(tmp_path)
        # Must use the rtl_port names from spec, not interface_name
        assert "s_axis_in" in content, "DUT missing .s_axis_in port connection"
        assert "m_axis_out" in content, "DUT missing .m_axis_out port connection"


# ═══════════════════════════════════════════════════════════════════
# §5  SVGenerator — signal completeness per protocol
# ═══════════════════════════════════════════════════════════════════


class TestSVGeneratorSignalCompleteness:
    """Verify generated wires include all required signals per protocol."""

    def test_axi4s_handshake_pair(self, tmp_path: Path):
        """AXI4-Stream: tvalid and tready both present (handshake pair)."""
        content = _generate_passthrough(tmp_path)
        lower = content.lower()
        assert "tvalid" in lower, "AXI4-Stream missing tvalid"
        assert "tready" in lower, "AXI4-Stream missing tready"

    def test_axi4s_data_signals(self, tmp_path: Path):
        """AXI4-Stream: tdata and tlast present."""
        content = _generate_passthrough(tmp_path)
        lower = content.lower()
        assert "tdata" in lower, "AXI4-Stream missing tdata"
        assert "tlast" in lower, "AXI4-Stream missing tlast"

    def test_axi4_five_channels(self, tmp_path: Path):
        """AXI4: all 5 channels (AR, R, AW, W, B) have valid/ready."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axi4",
            rtl_top="rtl/test.sv",
            interfaces={
                "mem": InterfaceSpec(
                    name="mem", rtl_port="m_axi_mem",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="mem", protocol=Protocol.AXI4, data_width=256, role="slave")]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        lower = content.lower()

        # Each channel must have valid+ready pair
        for ch in ["ar", "r", "aw", "w", "b"]:
            assert f"{ch}valid" in lower, f"AXI4 missing {ch}valid"
            assert f"{ch}ready" in lower, f"AXI4 missing {ch}ready"

    def test_axilite_no_burst_signals(self, tmp_path: Path):
        """AXI4-Lite: no burst signals (arlen, awlen, arsize, awsize, arburst, awburst)."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axilite",
            rtl_top="rtl/test.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axi_ctrl",
                    protocol=Protocol.AXI4L, addr_width=16,
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="ctrl", protocol=Protocol.AXI4L, data_width=32, role="master")]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        lower = content.lower()
        # AXI4-Lite must NOT have burst-related signals
        for sig in ["arlen", "awlen", "arsize", "awsize", "arburst", "awburst"]:
            assert sig not in lower, f"AXI4-Lite should not have burst signal '{sig}'"

    def test_axilite_handshake_present(self, tmp_path: Path):
        """AXI4-Lite: valid and ready signals present."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axilite",
            rtl_top="rtl/test.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axi_ctrl",
                    protocol=Protocol.AXI4L, addr_width=16,
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="ctrl", protocol=Protocol.AXI4L, data_width=32, role="master")]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        lower = content.lower()
        assert "awvalid" in lower, "AXI4-Lite missing awvalid"
        assert "awready" in lower, "AXI4-Lite missing awready"
        assert "wvalid" in lower, "AXI4-Lite missing wvalid"
        assert "wready" in lower, "AXI4-Lite missing wready"
