"""Tests for SVGenerator.generate() — Jinja2 template rendering.

Spec references:
- 06_codegen_and_cli.md §1 (Template Architecture)
- 06_codegen_and_cli.md §3 (Code Generator)
- NPU 3D accelerator mapping
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
                "vivado_path": "/tools/Xilinx/Vivado/2023.2",
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
                "vivado_path": "/tools/Xilinx/Vivado/2023.2",
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

    def test_generate_does_not_create_build_tcl(self, tmp_path: Path):
        """build.tcl is NOT generated (§8.9: removed in v0.5.0)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert not (tmp_path / "build.tcl").exists()

    def test_generate_does_not_create_run_tcl(self, tmp_path: Path):
        """run.tcl is NOT generated (§8.9: removed in v0.5.0)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert not (tmp_path / "run.tcl").exists()

    def test_generate_does_not_create_makefile(self, tmp_path: Path):
        """Makefile is NOT generated (§8.9: removed in v0.5.0)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert not (tmp_path / "Makefile").exists()

    def test_generate_only_produces_tb_top(self, tmp_path: Path):
        """generate() only produces tb_top.sv — no scripts (§8.9)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        generated_files = [f.name for f in tmp_path.iterdir() if f.is_file()]
        assert "tb_top.sv" in generated_files
        assert "build.tcl" not in generated_files
        assert "run.tcl" not in generated_files
        assert "Makefile" not in generated_files

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
        """DUT instance without wrapper uses rtl_port names directly."""
        content = _generate_passthrough(tmp_path)
        # No generate_controller → uses rtl_port names (not ext_port)
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


# ═══════════════════════════════════════════════════════════════════
# §6  BFM port connections — regression tests for C1–C6 bugs
# ═══════════════════════════════════════════════════════════════════


def _generate_axi4_only(tmp_path: Path) -> str:
    """Generate tb_top.sv with a single AXI4 BFM and return content."""
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
    cfgs = [BFMConfig(interface_name="ddr", protocol=Protocol.AXI4, data_width=256,
                       addr_width=64, role="slave")]
    gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
    gen.generate(str(tmp_path))
    return (tmp_path / "tb_top.sv").read_text()


def _generate_axilite_only(tmp_path: Path) -> str:
    """Generate tb_top.sv with a single AXI4-Lite BFM and return content."""
    from vten.codegen.sv_generator import SVGenerator

    spec = KernelSpec(
        kernel_name="test_axilite",
        rtl_top="rtl/test.sv",
        interfaces={
            "ctrl": InterfaceSpec(
                name="ctrl", rtl_port="s_axi_ctrl",
                protocol=Protocol.AXI4L, addr_width=16,
                registers=[RegisterSpec(name="reg0", offset=0x10)],
            ),
        },
    )
    cfgs = [BFMConfig(interface_name="ctrl", protocol=Protocol.AXI4L, data_width=32,
                       addr_width=16, role="master")]
    gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
    gen.generate(str(tmp_path))
    return (tmp_path / "tb_top.sv").read_text()


class TestBFMPortConnections:
    """Regression: BFM instance blocks must have actual AXI port connections.

    Verifies C1–C6 bug fixes: BFM instantiations in tb_top.sv must contain
    port connections like .s_araddr(wire), not just module-level wire declarations.
    """

    # ── AXI4 BFM (s_ prefix ports per vten_bfm_axi4.sv) ──

    def test_axi4_bfm_instance_has_s_araddr(self, tmp_path: Path):
        """AXI4 BFM instance has .s_araddr( port connection."""
        content = _generate_axi4_only(tmp_path)
        assert ".s_araddr(" in content, "AXI4 BFM missing .s_araddr( port"

    def test_axi4_bfm_instance_has_s_awaddr(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_awaddr(" in content, "AXI4 BFM missing .s_awaddr( port"

    def test_axi4_bfm_instance_has_s_rdata(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_rdata(" in content, "AXI4 BFM missing .s_rdata( port"

    def test_axi4_bfm_instance_has_s_wdata(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_wdata(" in content, "AXI4 BFM missing .s_wdata( port"

    def test_axi4_bfm_instance_has_s_wstrb(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_wstrb(" in content, "AXI4 BFM missing .s_wstrb( port"

    def test_axi4_bfm_instance_has_s_bresp(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_bresp(" in content, "AXI4 BFM missing .s_bresp( port"

    def test_axi4_bfm_instance_has_s_rresp(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_rresp(" in content, "AXI4 BFM missing .s_rresp( port"

    def test_axi4_bfm_instance_has_s_rlast(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_rlast(" in content, "AXI4 BFM missing .s_rlast( port"

    def test_axi4_bfm_instance_has_s_arlen(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_arlen(" in content, "AXI4 BFM missing .s_arlen( port"

    def test_axi4_bfm_instance_has_s_arsize(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_arsize(" in content, "AXI4 BFM missing .s_arsize( port"

    def test_axi4_bfm_instance_has_s_arburst(self, tmp_path: Path):
        content = _generate_axi4_only(tmp_path)
        assert ".s_arburst(" in content, "AXI4 BFM missing .s_arburst( port"

    def test_axi4_bfm_all_s_prefix_ports(self, tmp_path: Path):
        """AXI4 BFM has all 24 s_-prefix port connections."""
        content = _generate_axi4_only(tmp_path)
        expected = [
            ".s_araddr(", ".s_arlen(", ".s_arsize(", ".s_arburst(",
            ".s_arvalid(", ".s_arready(",
            ".s_rdata(", ".s_rresp(", ".s_rlast(", ".s_rvalid(", ".s_rready(",
            ".s_awaddr(", ".s_awlen(", ".s_awsize(", ".s_awburst(",
            ".s_awvalid(", ".s_awready(",
            ".s_wdata(", ".s_wstrb(", ".s_wlast(", ".s_wvalid(", ".s_wready(",
            ".s_bresp(", ".s_bvalid(", ".s_bready(",
        ]
        for port in expected:
            assert port in content, f"AXI4 BFM missing port {port}"

    # ── AXI4-Lite BFM (m_ prefix ports per vten_bfm_axilite.sv) ──

    def test_axilite_bfm_instance_has_m_awaddr(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert ".m_awaddr(" in content, "AXI4-Lite BFM missing .m_awaddr( port"

    def test_axilite_bfm_instance_has_m_wdata(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert ".m_wdata(" in content, "AXI4-Lite BFM missing .m_wdata( port"

    def test_axilite_bfm_instance_has_m_wstrb(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert ".m_wstrb(" in content, "AXI4-Lite BFM missing .m_wstrb( port"

    def test_axilite_bfm_instance_has_m_bresp(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert ".m_bresp(" in content, "AXI4-Lite BFM missing .m_bresp( port"

    def test_axilite_bfm_instance_has_m_rresp(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert ".m_rresp(" in content, "AXI4-Lite BFM missing .m_rresp( port"

    def test_axilite_bfm_instance_has_m_araddr(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert ".m_araddr(" in content, "AXI4-Lite BFM missing .m_araddr( port"

    def test_axilite_bfm_instance_has_m_rdata(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert ".m_rdata(" in content, "AXI4-Lite BFM missing .m_rdata( port"

    def test_axilite_bfm_all_m_prefix_ports(self, tmp_path: Path):
        """AXI4-Lite BFM has all 17 m_-prefix port connections."""
        content = _generate_axilite_only(tmp_path)
        expected = [
            ".m_awaddr(", ".m_awvalid(", ".m_awready(",
            ".m_wdata(", ".m_wstrb(", ".m_wvalid(", ".m_wready(",
            ".m_bresp(", ".m_bvalid(", ".m_bready(",
            ".m_araddr(", ".m_arvalid(", ".m_arready(",
            ".m_rdata(", ".m_rresp(", ".m_rvalid(", ".m_rready(",
        ]
        for port in expected:
            assert port in content, f"AXI4-Lite BFM missing port {port}"

    # ── AXI4-Stream BFM (master vs slave role) ──

    def test_axi4s_master_bfm_m_ports_connected(self, tmp_path: Path):
        """AXI4-Stream master BFM: .m_tdata(wire), .m_tvalid(wire), etc."""
        content = _generate_passthrough(tmp_path)
        # s_axis_in is master role BFM (drives data into DUT)
        assert ".m_tdata(s_axis_in_tdata)" in content, \
            "AXI4S master BFM missing .m_tdata(wire) connection"
        assert ".m_tvalid(s_axis_in_tvalid)" in content
        assert ".m_tready(s_axis_in_tready)" in content
        assert ".m_tlast(s_axis_in_tlast)" in content

    def test_axi4s_master_bfm_s_ports_tied_off(self, tmp_path: Path):
        """AXI4-Stream master BFM: .s_tdata('0), .s_tvalid(1'b0) tie-off."""
        content = _generate_passthrough(tmp_path)
        assert ".s_tdata('0)" in content, "AXI4S master BFM missing .s_tdata('0) tie-off"
        assert ".s_tvalid(1'b0)" in content, "AXI4S master BFM missing .s_tvalid(1'b0) tie-off"

    def test_axi4s_slave_bfm_s_ports_connected(self, tmp_path: Path):
        """AXI4-Stream slave BFM: .s_tdata(wire), .s_tvalid(wire), etc."""
        content = _generate_passthrough(tmp_path)
        # m_axis_out is slave role BFM (receives data from DUT)
        assert ".s_tdata(m_axis_out_tdata)" in content, \
            "AXI4S slave BFM missing .s_tdata(wire) connection"
        assert ".s_tvalid(m_axis_out_tvalid)" in content
        assert ".s_tready(m_axis_out_tready)" in content
        assert ".s_tlast(m_axis_out_tlast)" in content

    def test_axi4s_slave_bfm_m_tready_tied_high(self, tmp_path: Path):
        """AXI4-Stream slave BFM: .m_tready(1'b1) tie-off."""
        content = _generate_passthrough(tmp_path)
        assert ".m_tready(1'b1)" in content, "AXI4S slave BFM missing .m_tready(1'b1) tie-off"


class TestBFMCmdIfPortName:
    """Regression: BFM uses .cmd_if(bfm_cmd[N]), not .cmd(bfm_cmd[N])."""

    def test_cmd_if_port_name(self, tmp_path: Path):
        """BFM instance has .cmd_if(bfm_cmd[N]) port."""
        content = _generate_passthrough(tmp_path)
        assert re.search(r"\.cmd_if\(bfm_cmd\[", content), \
            "BFM missing .cmd_if(bfm_cmd[N]) — found .cmd( instead?"

    def test_no_dot_cmd_port(self, tmp_path: Path):
        """BFM instance does NOT have .cmd(bfm_cmd[N])."""
        content = _generate_passthrough(tmp_path)
        # .cmd( without _if is the old buggy pattern
        assert not re.search(r"\.cmd\(bfm_cmd\[", content), \
            "BFM has old .cmd(bfm_cmd[N]) pattern — should be .cmd_if(..."

    def test_cmd_if_npu_40bfm(self, tmp_path: Path):
        """All 40 NPU BFMs have .cmd_if(bfm_cmd[N])."""
        content = _generate_npu(tmp_path)
        matches = re.findall(r"\.cmd_if\(bfm_cmd\[\d+\]\)", content)
        assert len(matches) >= 40, (
            f"Expected >= 40 .cmd_if(bfm_cmd[N]) ports, found {len(matches)}"
        )


class TestCycleCounter:
    """Regression: tb_top.sv has cycle_count declaration, increment, BFM connection."""

    def test_cycle_count_declaration(self, tmp_path: Path):
        """int cycle_count; declared in tb_top.sv."""
        content = _generate_passthrough(tmp_path)
        assert "int cycle_count" in content, "Missing 'int cycle_count' declaration"

    def test_cycle_count_always_ff_increment(self, tmp_path: Path):
        """cycle_count incremented in always_ff block."""
        content = _generate_passthrough(tmp_path)
        assert "cycle_count <= cycle_count + 1" in content, \
            "Missing cycle_count increment in always_ff"

    def test_cycle_count_connected_to_all_bfms(self, tmp_path: Path):
        """All BFMs have .cycle_count(cycle_count) connection."""
        content = _generate_passthrough(tmp_path)
        count = content.count(".cycle_count(cycle_count)")
        # passthrough has 2 BFMs
        assert count >= 2, f"Expected >= 2 .cycle_count(cycle_count), found {count}"

    def test_cycle_count_npu_40bfm(self, tmp_path: Path):
        """All 40 NPU BFMs have .cycle_count(cycle_count)."""
        content = _generate_npu(tmp_path)
        count = content.count(".cycle_count(cycle_count)")
        assert count >= 40, f"Expected >= 40 .cycle_count(cycle_count), found {count}"


class TestAXI4LiteSignalCompleteness:
    """Regression: AXI4-Lite wstrb/bresp/rresp in wire decl, DUT, BFM."""

    def test_wire_declarations_have_wstrb(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert "_wstrb" in content, "Wire declaration missing _wstrb"

    def test_wire_declarations_have_bresp(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert "_bresp" in content, "Wire declaration missing _bresp"

    def test_wire_declarations_have_rresp(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert "_rresp" in content, "Wire declaration missing _rresp"

    def test_dut_connections_have_wstrb(self, tmp_path: Path):
        """DUT instance has .xxx_wstrb(xxx_wstrb) connection."""
        content = _generate_axilite_only(tmp_path)
        assert re.search(r"\.\w+_wstrb\(\w+_wstrb\)", content), \
            "DUT missing wstrb port connection"

    def test_dut_connections_have_bresp(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert re.search(r"\.\w+_bresp\(\w+_bresp\)", content), \
            "DUT missing bresp port connection"

    def test_dut_connections_have_rresp(self, tmp_path: Path):
        content = _generate_axilite_only(tmp_path)
        assert re.search(r"\.\w+_rresp\(\w+_rresp\)", content), \
            "DUT missing rresp port connection"


class TestAXI4StreamMODEParameter:
    """Regression: AXI4-Stream BFM has .MODE("MASTER"/"SLAVE") parameter."""

    def test_mode_parameter_exists(self, tmp_path: Path):
        """Rendered tb_top.sv has .MODE( parameter for AXI4S BFMs."""
        content = _generate_passthrough(tmp_path)
        assert ".MODE(" in content, "AXI4S BFM missing .MODE( parameter"

    def test_master_role_mode_value(self, tmp_path: Path):
        """Master role BFM has .MODE("MASTER")."""
        content = _generate_passthrough(tmp_path)
        assert '.MODE("MASTER")' in content, \
            "AXI4S master BFM missing .MODE(\"MASTER\")"

    def test_slave_role_mode_value(self, tmp_path: Path):
        """Slave role BFM has .MODE("SLAVE")."""
        content = _generate_passthrough(tmp_path)
        assert '.MODE("SLAVE")' in content, \
            "AXI4S slave BFM missing .MODE(\"SLAVE\")"

    def test_svgenerator_builds_mode_parameter(self):
        """SVGenerator._build_context() sets MODE in AXI4S BFM parameters."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=_passthrough_bfm_configs(),
            project_config=_minimal_config(),
        )
        ctx = gen._build_context()
        for bfm in ctx.tb.bfms:
            if bfm.protocol == "axi4_stream":
                assert "MODE" in bfm.parameters, f"BFM {bfm.name} missing MODE parameter"
                if bfm.role == "master":
                    assert "MASTER" in bfm.parameters["MODE"]
                else:
                    assert "SLAVE" in bfm.parameters["MODE"]


class TestADDR_WParameterization:
    """Regression: Wire widths use ADDR_W parameter, not hardcoded [63:0]."""

    def test_axi4_wire_uses_parameterized_addr_width(self, tmp_path: Path):
        """AXI4 araddr/awaddr wire width from ADDR_W, not [63:0] hardcode."""
        content = _generate_axi4_only(tmp_path)
        # With ADDR_W=64, wire should be [63:0] — but it should come from
        # bfm.parameters.ADDR_W in template, not hardcoded.
        # Verify the template rendered correctly for ADDR_W=64:
        assert re.search(r"logic\s+\[63:0\]\s+\w+_araddr", content), \
            "AXI4 araddr wire width incorrect for ADDR_W=64"
        assert re.search(r"logic\s+\[63:0\]\s+\w+_awaddr", content), \
            "AXI4 awaddr wire width incorrect for ADDR_W=64"

    def test_axi4_different_addr_width(self, tmp_path: Path):
        """AXI4 with ADDR_W=48 produces [47:0] wire width."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axi4_48",
            rtl_top="rtl/test.sv",
            interfaces={
                "mem": InterfaceSpec(
                    name="mem", rtl_port="m_axi_mem",
                    protocol=Protocol.AXI4, data_width=256, addr_width=48,
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="mem", protocol=Protocol.AXI4, data_width=256,
                           addr_width=48, role="slave")]
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        assert re.search(r"logic\s+\[47:0\]\s+\w+_araddr", content), \
            "AXI4 araddr should be [47:0] for ADDR_W=48"
        assert "[63:0]" not in content or "araddr" not in content.split("[63:0]")[0][-20:], \
            "AXI4 wire still using hardcoded [63:0] with ADDR_W=48"

    def test_axilite_wire_uses_parameterized_addr_width(self, tmp_path: Path):
        """AXI4-Lite with ADDR_W=16 produces [15:0] wire width."""
        content = _generate_axilite_only(tmp_path)
        assert re.search(r"logic\s+\[15:0\]\s+\w+_araddr", content), \
            "AXI4-Lite araddr should be [15:0] for ADDR_W=16"
        assert re.search(r"logic\s+\[15:0\]\s+\w+_awaddr", content), \
            "AXI4-Lite awaddr should be [15:0] for ADDR_W=16"

    def test_axilite_different_addr_width(self, tmp_path: Path):
        """AXI4-Lite with ADDR_W=12 produces [11:0] wire width."""
        from vten.codegen.sv_generator import SVGenerator

        spec = KernelSpec(
            kernel_name="test_axilite_12",
            rtl_top="rtl/test.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axi_ctrl",
                    protocol=Protocol.AXI4L, addr_width=12,
                ),
            },
        )
        cfgs = [BFMConfig(interface_name="ctrl", protocol=Protocol.AXI4L, data_width=32,
                           addr_width=12, role="master")]
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        assert re.search(r"logic\s+\[11:0\]\s+\w+_araddr", content), \
            "AXI4-Lite araddr should be [11:0] for ADDR_W=12"


class TestBFMPortConnectionsNPU:
    """NPU 40-BFM scale: verify port connections for all 3 protocol types."""

    def test_npu_axi4_bfms_have_s_prefix_ports(self, tmp_path: Path):
        """NPU AXI4 BFMs (DDR+HBM) have .s_araddr(, .s_awaddr( etc."""
        content = _generate_npu(tmp_path)
        # 34 AXI4 BFMs → should have 34 .s_araddr( occurrences
        s_araddr_count = content.count(".s_araddr(")
        assert s_araddr_count >= 34, (
            f"Expected >= 34 .s_araddr( for NPU AXI4 BFMs, found {s_araddr_count}"
        )

    def test_npu_axilite_bfms_have_m_prefix_ports(self, tmp_path: Path):
        """NPU AXI4-Lite BFMs (6 ctrl IPs) have .m_awaddr( etc."""
        content = _generate_npu(tmp_path)
        m_awaddr_count = content.count(".m_awaddr(")
        assert m_awaddr_count >= 6, (
            f"Expected >= 6 .m_awaddr( for NPU AXI4-Lite BFMs, found {m_awaddr_count}"
        )

    def test_npu_all_bfms_have_cmd_if(self, tmp_path: Path):
        """All 40 NPU BFMs have .cmd_if(bfm_cmd[N])."""
        content = _generate_npu(tmp_path)
        matches = re.findall(r"\.cmd_if\(bfm_cmd\[\d+\]\)", content)
        assert len(matches) >= 40

    def test_npu_all_bfms_have_cycle_count(self, tmp_path: Path):
        """All 40 NPU BFMs have .cycle_count(cycle_count)."""
        content = _generate_npu(tmp_path)
        count = content.count(".cycle_count(cycle_count)")
        assert count >= 40
