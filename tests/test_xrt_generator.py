"""Phase B tests for XrtGenerator.

Tests cover:
  S1: Initialization — constructor stores spec and config
  S2: File generation — generate() creates files in output dir
  S3: Template rendering — protocol-specific content in generated files
"""

from __future__ import annotations

import pytest

from vten.codegen.xrt_generator import XrtGenerator
from vten.spec.models import (
    AutoBindSpec,
    InterfaceSpec,
    KernelSpec,
    Protocol,
    RegisterSpec,
    XrtInterfaceConfig,
)


# ── Helper ──


def _make_spec() -> KernelSpec:
    """Build a minimal KernelSpec with AXI4 + AXI4-Lite interfaces."""
    return KernelSpec(
        kernel_name="test_kern",
        rtl_top="test_kern_top",
        interfaces={
            "data_in": InterfaceSpec(
                name="data_in",
                rtl_port="m_axi_data_in",
                protocol=Protocol.AXI4,
                data_width=256,
                addr_width=64,
                xrt=XrtInterfaceConfig(
                    memory_bank="HBM[0]",
                    arg_index=0,
                ),
                tensor="input_data",
            ),
            "ctrl": InterfaceSpec(
                name="ctrl",
                rtl_port="s_axi_ctrl",
                protocol=Protocol.AXI4L,
                data_width=32,
                addr_width=12,
                registers=[
                    RegisterSpec(
                        name="CONTROL",
                        offset=0x00,
                        fields={"start": "0:0", "done": "1:1"},
                    ),
                    RegisterSpec(
                        name="addr_lo",
                        offset=0x10,
                        auto_bind=AutoBindSpec(
                            tensor="input_data", value="address", bits="31:0"
                        ),
                    ),
                ],
            ),
        },
    )


# ============================================================
# S1: XrtGenerator initialization
# ============================================================


class TestXrtGeneratorInit:
    """Constructor stores spec and config."""

    def test_stores_spec(self):
        spec = _make_spec()
        gen = XrtGenerator(spec)
        assert gen._spec is spec

    def test_stores_config(self):
        spec = _make_spec()
        cfg = {"rtl": {"sources": ["rtl/top.sv"]}}
        gen = XrtGenerator(spec, project_config=cfg)
        assert gen._config is cfg

    def test_default_config_is_empty_dict(self):
        gen = XrtGenerator(_make_spec())
        assert gen._config == {}

    def test_none_config_becomes_empty_dict(self):
        gen = XrtGenerator(_make_spec(), project_config=None)
        assert gen._config == {}


# ============================================================
# S2: File generation
# ============================================================


class TestFileGeneration:
    """generate() creates the expected files in output dir."""

    @pytest.fixture()
    def result(self, tmp_path):
        gen = XrtGenerator(_make_spec())
        return gen.generate(str(tmp_path))

    def test_returns_expected_keys(self, result):
        assert set(result.keys()) == {
            "package_ip.tcl",
            "kernel.xml",
            "gen_xo.tcl",
            "connectivity.cfg",
            "build_hw_emu.sh",
        }

    def test_all_files_exist(self, result):
        for path in result.values():
            assert path.exists(), f"{path} does not exist"

    def test_files_in_flat_output_dir(self, result):
        """All files should be in the same directory."""
        parents = {path.parent for path in result.values()}
        assert len(parents) == 1

    def test_kernel_xml_contains_kernel_name(self, result):
        content = result["kernel.xml"].read_text()
        assert 'name="test_kern"' in content

    def test_gen_xo_contains_kernel_name(self, result):
        content = result["gen_xo.tcl"].read_text()
        assert "test_kern" in content

    def test_connectivity_contains_kernel_name(self, result):
        content = result["connectivity.cfg"].read_text()
        assert "test_kern" in content

    def test_kernel_xml_contains_ext_port_names(self, result):
        content = result["kernel.xml"].read_text()
        assert "m_axi_data_in" in content
        assert "s_axi_ctrl" in content

    def test_package_ip_contains_ext_port_names(self, result):
        content = result["package_ip.tcl"].read_text()
        assert "m_axi_data_in" in content
        assert "s_axi_ctrl" in content

    def test_output_dir_created_if_missing(self, tmp_path):
        out = tmp_path / "nested" / "deep"
        gen = XrtGenerator(_make_spec())
        result = gen.generate(str(out))
        assert out.exists()
        for path in result.values():
            assert path.exists()

    def test_build_script_is_executable(self, result):
        import stat

        path = result["build_hw_emu.sh"]
        assert path.stat().st_mode & stat.S_IXUSR


# ============================================================
# S3: Template rendering
# ============================================================


class TestTemplateRendering:
    """Protocol-specific content appears correctly in generated files."""

    @pytest.fixture()
    def generated(self, tmp_path):
        gen = XrtGenerator(_make_spec())
        return gen.generate(str(tmp_path))

    # -- connectivity.cfg: AXI4 interfaces produce sp= lines with ext_port --

    def test_connectivity_has_sp_line_for_axi4(self, generated):
        content = generated["connectivity.cfg"].read_text()
        assert "sp=test_kern_1.m_axi_data_in:HBM[0]" in content

    def test_connectivity_no_sp_line_for_axilite(self, generated):
        """AXI4-Lite interfaces should not appear as sp= entries."""
        content = generated["connectivity.cfg"].read_text()
        for line in content.splitlines():
            if line.startswith("sp="):
                assert "ctrl" not in line

    # -- kernel.xml: AXI4-Lite produces addr_range --

    def test_kernel_xml_axilite_has_addr_range(self, generated):
        content = generated["kernel.xml"].read_text()
        assert 'name="s_axi_ctrl"' in content
        assert 'mode="slave"' in content
        assert "0x1000" in content

    def test_kernel_xml_axi4_has_master_mode(self, generated):
        content = generated["kernel.xml"].read_text()
        assert 'name="m_axi_data_in"' in content
        assert 'mode="master"' in content

    def test_kernel_xml_axi4_data_width(self, generated):
        content = generated["kernel.xml"].read_text()
        assert 'dataWidth="256"' in content

    def test_kernel_xml_has_vlnv(self, generated):
        content = generated["kernel.xml"].read_text()
        assert 'vlnv="user.org:kernel:test_kern:1.0"' in content

    def test_kernel_xml_has_ip_c_language(self, generated):
        content = generated["kernel.xml"].read_text()
        assert 'language="ip_c"' in content

    def test_kernel_xml_has_interrupt(self, generated):
        content = generated["kernel.xml"].read_text()
        assert 'interrupt="true"' in content

    def test_kernel_xml_args_have_id_and_offset(self, generated):
        content = generated["kernel.xml"].read_text()
        assert 'id="0"' in content
        assert 'offset="0x10"' in content
        assert 'type="int*"' in content

    # -- package_ip.tcl: registers appear --

    def test_package_ip_contains_register(self, generated):
        content = generated["package_ip.tcl"].read_text()
        assert "CONTROL" in content

    def test_package_ip_has_sdx_kernel(self, generated):
        content = generated["package_ip.tcl"].read_text()
        assert "sdx_kernel true" in content
        assert "sdx_kernel_type rtl" in content

    def test_package_ip_has_clock_association(self, generated):
        content = generated["package_ip.tcl"].read_text()
        assert "ap_clk" in content

    # -- gen_xo.tcl: XO path and kernel_xml reference --

    def test_gen_xo_has_xo_path(self, generated):
        content = generated["gen_xo.tcl"].read_text()
        assert "test_kern.xo" in content

    def test_gen_xo_references_kernel_xml(self, generated):
        content = generated["gen_xo.tcl"].read_text()
        assert "kernel.xml" in content

    def test_gen_xo_uses_script_dir(self, generated):
        content = generated["gen_xo.tcl"].read_text()
        assert "set script_dir" in content
        assert "$script_dir" in content

    # -- RTL sources in package_ip.tcl --

    def test_rtl_sources_appear_in_package_ip(self, tmp_path):
        spec = _make_spec()
        cfg = {"rtl": {"sources": ["rtl/top.sv", "rtl/sub.sv"]}}
        gen = XrtGenerator(spec, project_config=cfg)
        result = gen.generate(str(tmp_path))
        content = result["package_ip.tcl"].read_text()
        assert "rtl/top.sv" in content
        assert "rtl/sub.sv" in content

    # -- Multiple registers --

    def test_multiple_registers_rendered(self, tmp_path):
        spec = _make_spec()
        ctrl = spec.interfaces["ctrl"]
        ctrl.registers.append(
            RegisterSpec(name="STATUS", offset=0x04)
        )
        gen = XrtGenerator(spec)
        result = gen.generate(str(tmp_path))
        content = result["package_ip.tcl"].read_text()
        assert "CONTROL" in content
        assert "STATUS" in content

    # -- Build script --

    def test_build_script_contains_kernel_name(self, generated):
        content = generated["build_hw_emu.sh"].read_text()
        assert "test_kern" in content

    def test_build_script_references_v_plus_plus(self, generated):
        content = generated["build_hw_emu.sh"].read_text()
        assert "v++ -l" in content
