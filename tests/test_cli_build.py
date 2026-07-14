"""Tests for vten build — Multi-Kernel 5-Stage Pipeline.

Spec references:
- 06_codegen_and_cli.md §4.3 (vten build)
- 06_codegen_and_cli.md §7   (Multi-Kernel Project Structure)
- 06_codegen_and_cli.md §8   (Staged Build Pipeline)

build_project() signature:
    build_project(project_dir, kernel_name=None, stage=None, upto=None,
                  force=False, skip_compile=False, config_overrides=None)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


MINIMAL_TOML = """\
[project]
name = "test_proj"
version = "0.1.0"

[rtl]
sources = ["rtl/*.sv"]
top_module = "passthrough"
include_dirs = []

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
"""

TOML_WITH_PARAMS = """\
[project]
name = "test_proj"
version = "0.1.0"

[parameters]
C = 32
D = 4
H = 8
W = 8

[rtl]
sources = ["rtl/*.sv"]
top_module = "passthrough"
include_dirs = []

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
"""

MULTI_KERNEL_TOML = """\
[project]
name = "multi_proj"
version = "0.1.0"

[rtl]
sources = ["rtl/**/*.sv"]
top_module = "NPU_3D_top"
include_dirs = ["rtl/include"]

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
"""


def _minimal_kernel_spec(kernel_name: str = "passthrough") -> dict:
    """Minimal AXI4-Stream passthrough kernel spec."""
    return {
        "kernel": kernel_name,
        "rtl_top": "rtl/passthrough.sv",
        "parameters": {"SIZE": "${SIZE}"},
        "interfaces": {
            "axi_stream_in": {
                "rtl_port": "s_axis_in",
                "protocol": "axi4_stream",
                "tensor": "data_in",
                "packing": {"element_width": 8, "elements_per_beat": 4},
            },
            "axi_stream_out": {
                "rtl_port": "m_axis_out",
                "protocol": "axi4_stream",
                "tensor": "data_out",
                "packing": {"element_width": 8, "elements_per_beat": 4},
            },
        },
    }


def _setup_project(
    tmp_path: Path,
    toml_content: str = MINIMAL_TOML,
    kernel_name: str = "passthrough",
    kernel_spec: dict | None = None,
) -> Path:
    """Create minimal project directory with kernels/<name>/kernel_spec.yaml.

    Directory layout (§7.1):
        proj/
        ├── vten.toml
        ├── rtl/passthrough.sv
        ├── kernels/<name>/
        │   └── kernel_spec.yaml
        └── build/              # project-level build output
    """
    project = tmp_path / "proj"
    project.mkdir()
    (project / "vten.toml").write_text(toml_content)
    (project / "rtl").mkdir()
    (project / "rtl" / "passthrough.sv").write_text("module passthrough; endmodule")
    (project / "build").mkdir()

    # Kernel directory with kernel_spec.yaml
    kernel_dir = project / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True)
    spec = kernel_spec or _minimal_kernel_spec(kernel_name)
    (kernel_dir / "kernel_spec.yaml").write_text(
        yaml.dump(spec, default_flow_style=False, sort_keys=False)
    )
    return project


def _setup_multi_kernel_project(tmp_path: Path) -> Path:
    """Create project with 3 kernels: conv3d, dma_ifm, passthrough."""
    project = tmp_path / "multi"
    project.mkdir()
    (project / "vten.toml").write_text(MULTI_KERNEL_TOML)
    (project / "rtl").mkdir()
    (project / "rtl" / "NPU_3D_top.sv").write_text("module NPU_3D_top; endmodule")
    (project / "build").mkdir()

    for name in ["conv3d", "dma_ifm", "passthrough"]:
        kd = project / "kernels" / name
        kd.mkdir(parents=True)
        (kd / "kernel_spec.yaml").write_text(
            yaml.dump(_minimal_kernel_spec(name), default_flow_style=False, sort_keys=False)
        )
    return project


# ═══════════════════════════════════════════════════════════════════
# §1  Kernel discovery (§7.3)
# ═════════════════════════════════════════════���═════════════════════


class TestDiscoverKernels:
    """discover_kernels() scans kernels/ for kernel_spec.yaml."""

    def test_discover_single_kernel(self, tmp_path: Path):
        """Single kernel directory discovered."""
        from vten.cli.build import discover_kernels

        project = _setup_project(tmp_path)
        kernels = discover_kernels(project)
        assert kernels == ["passthrough"]

    def test_discover_multiple_kernels_sorted(self, tmp_path: Path):
        """Multiple kernels discovered in sorted order."""
        from vten.cli.build import discover_kernels

        project = _setup_multi_kernel_project(tmp_path)
        kernels = discover_kernels(project)
        assert kernels == ["conv3d", "dma_ifm", "passthrough"]

    def test_discover_ignores_dirs_without_spec(self, tmp_path: Path):
        """Directories without kernel_spec.yaml are ignored."""
        from vten.cli.build import discover_kernels

        project = _setup_project(tmp_path)
        # Create a directory without kernel_spec.yaml
        (project / "kernels" / "no_spec_dir").mkdir(parents=True)
        kernels = discover_kernels(project)
        assert "no_spec_dir" not in kernels
        assert "passthrough" in kernels

    def test_discover_empty_kernels_dir(self, tmp_path: Path):
        """Empty kernels/ directory returns empty list."""
        from vten.cli.build import discover_kernels

        project = tmp_path / "empty_proj"
        project.mkdir()
        (project / "kernels").mkdir()
        kernels = discover_kernels(project)
        assert kernels == []

    def test_discover_no_kernels_dir(self, tmp_path: Path):
        """Missing kernels/ directory returns empty list."""
        from vten.cli.build import discover_kernels

        project = tmp_path / "no_kernels"
        project.mkdir()
        kernels = discover_kernels(project)
        assert kernels == []


# ═══════════════════════════════════════════════════════════════════
# §2  Build output structure — kernel-scoped (§7.1, §8.4)
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildOutput:
    """vten build: kernel-scoped output directory structure."""

    def test_build_creates_kernel_build_dir(self, tmp_path: Path):
        """kernels/<name>/build/ directory is created."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        assert (project / "kernels" / "passthrough" / "build").is_dir()

    def test_build_creates_generated_dir(self, tmp_path: Path):
        """kernels/<name>/build/generated/ directory exists after build."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        assert (project / "kernels" / "passthrough" / "build" / "generated").is_dir()

    def test_build_generates_tb_top(self, tmp_path: Path):
        """Stage 3 codegen produces tb_top.sv in kernel build/generated/."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        tb_top = project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        assert tb_top.exists()

    def test_build_tb_top_contains_dut(self, tmp_path: Path):
        """Generated tb_top.sv references the DUT top module."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        tb_top = project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        assert tb_top.exists()
        content = tb_top.read_text()
        assert "passthrough" in content

    def test_build_produces_generated_sv(self, tmp_path: Path):
        """Build pipeline produces SV files in kernel's generated/ dir."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        sv_files = list(
            (project / "kernels" / "passthrough" / "build" / "generated").glob("*.sv")
        )
        assert len(sv_files) >= 1

    def test_build_project_level_lib_dir(self, tmp_path: Path):
        """build/lib/ directory is created at project level for DPI-C."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="dpi_c")
        assert (project / "build" / "lib").is_dir()

    def test_build_no_scripts_dir(self, tmp_path: Path):
        """build/scripts/ is NOT created (replaced by Stage 4-5)."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        # Scripts dir should not exist — xvlog/xelab invoked directly by Python
        assert not (project / "build" / "scripts").exists()

    def test_build_no_build_tcl(self, tmp_path: Path):
        """build.tcl is NOT generated (§8.9: removed in v0.5.0)."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        # Search all build directories — no build.tcl anywhere
        assert list(project.rglob("build.tcl")) == []

    def test_build_no_run_tcl(self, tmp_path: Path):
        """run.tcl is NOT generated (§8.9: removed in v0.5.0)."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        assert list(project.rglob("run.tcl")) == []


# ═══════════════════════════════════════════════════════════════════
# §3  Single-kernel build (--kernel option)
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildSingleKernel:
    """--kernel option builds only the specified kernel."""

    def test_build_single_kernel(self, tmp_path: Path):
        """--kernel conv3d builds only conv3d, not others."""
        from vten.cli.build import build_project

        project = _setup_multi_kernel_project(tmp_path)
        build_project(str(project), kernel_name="conv3d", stage="codegen")

        # conv3d should have generated output
        assert (project / "kernels" / "conv3d" / "build" / "generated").is_dir()

        # Other kernels should NOT have build/generated/
        assert not (project / "kernels" / "dma_ifm" / "build" / "generated").exists()
        assert not (project / "kernels" / "passthrough" / "build" / "generated").exists()

    def test_build_all_kernels_default(self, tmp_path: Path):
        """Without --kernel, all discovered kernels are built."""
        from vten.cli.build import build_project

        project = _setup_multi_kernel_project(tmp_path)
        build_project(str(project), stage="codegen")

        for name in ["conv3d", "dma_ifm", "passthrough"]:
            assert (project / "kernels" / name / "build" / "generated").is_dir(), (
                f"kernels/{name}/build/generated/ should exist"
            )

    def test_build_missing_kernel_error(self, tmp_path: Path):
        """--kernel nonexistent raises error at codegen stage."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        with pytest.raises(Exception, match="(?i)kernel|not found|no such"):
            build_project(str(project), kernel_name="nonexistent", stage="codegen")


# ═══════════════════════════════════════════════════════════════════
# §4  Stage control (--stage, --upto, --skip-compile)
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildStageControl:
    """Stage pipeline control options (§8.7)."""

    def test_build_stage_codegen_only(self, tmp_path: Path):
        """--stage codegen runs only Stage 3, produces tb_top.sv."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")

        # Codegen output should exist
        assert (
            project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        ).exists()

        # Stage 1 output (vivado_proj) should NOT exist when only codegen requested
        assert not (project / "build" / "vivado_proj").exists()

    @pytest.mark.xsim
    def test_build_upto_codegen(self, tmp_path: Path):
        """--upto codegen runs Stages 1-3 but NOT 4-5.

        Stages 1-2 invoke Vivado, so this is marked ``xsim`` and skipped
        when no simulator is available.
        """
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), upto="codegen")

        # Codegen output should exist
        assert (
            project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        ).exists()

        # Stage 4 output (compile.prj) should NOT exist
        assert not (
            project / "kernels" / "passthrough" / "build" / "compile.prj"
        ).exists()

    @pytest.mark.xsim
    def test_build_upto_dpi_c(self, tmp_path: Path):
        """--upto dpi_c runs Stages 1-2 only (no kernel codegen).

        Stages 1-2 invoke Vivado, so this is marked ``xsim`` and skipped
        when no simulator is available.
        """
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), upto="dpi_c")

        # Project-level lib/ should exist (Stage 2)
        assert (project / "build" / "lib").is_dir()

        # Kernel codegen should NOT have run
        assert not (
            project / "kernels" / "passthrough" / "build" / "generated"
        ).exists()

    def test_build_skip_compile(self, tmp_path: Path):
        """--skip-compile runs up to codegen but skips Stage 4-5."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), skip_compile=True)

        # Codegen should exist
        assert (
            project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        ).exists()

        # xsim.dir should NOT exist (Stage 5 skipped)
        assert not (
            project / "kernels" / "passthrough" / "build" / "xsim.dir"
        ).exists()

        # compile.prj should NOT exist (Stage 4 skipped)
        assert not (
            project / "kernels" / "passthrough" / "build" / "compile.prj"
        ).exists()

    def test_build_invalid_stage_name_error(self, tmp_path: Path):
        """Invalid stage name raises error."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        with pytest.raises(Exception, match="(?i)stage|invalid|unknown"):
            build_project(str(project), stage="nonexistent_stage")


# ═══════════════════════════════════════════════════════════════════
# §5  Cache and --force rebuild (§8.1, §8.7)
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildCache:
    """Build cache and --force rebuild behavior."""

    def test_rebuild_without_force_succeeds(self, tmp_path: Path):
        """Second build without --force succeeds (idempotent)."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")
        # Second build — should not fail
        build_project(str(project), stage="codegen")
        assert (
            project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        ).exists()

    def test_force_rebuild(self, tmp_path: Path):
        """--force ignores cache and regenerates."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")

        # Verify generated file exists
        tb_top = project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        assert tb_top.exists()

        # Force rebuild
        build_project(str(project), stage="codegen", force=True)

        # File should still exist and be valid
        assert tb_top.exists()
        assert len(tb_top.read_text()) > 0

    def test_rebuild_with_different_config(self, tmp_path: Path):
        """Rebuild with different config overrides produces new output."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path, TOML_WITH_PARAMS)
        build_project(str(project), stage="codegen", config_overrides={"C": 32})
        build_project(str(project), stage="codegen", config_overrides={"C": 64})
        assert (
            project / "kernels" / "passthrough" / "build" / "generated" / "tb_top.sv"
        ).exists()

    def test_cache_file_created(self, tmp_path: Path):
        """build/.cache.json is created after build."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")

        cache_file = project / "build" / ".cache.json"
        assert cache_file.exists()


# ═══════════════════════════════════════════════════════════════════
# §6  Config overrides
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildConfigOverrides:
    """--config parameter overrides."""

    def test_config_override_single(self, tmp_path: Path):
        """--config C=32 overrides [parameters].C."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen", config_overrides={"C": 32})
        # Should succeed without error
        assert (project / "kernels" / "passthrough" / "build" / "generated").is_dir()

    def test_config_override_multiple(self, tmp_path: Path):
        """Multiple config overrides applied together."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path, TOML_WITH_PARAMS)
        build_project(str(project), stage="codegen", config_overrides={"C": 64, "D": 8, "H": 16})
        assert (project / "kernels" / "passthrough" / "build" / "generated").is_dir()


# ═══════════════════════════════════════════════════════════════════
# §7  Error cases
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildErrors:
    """Build error cases."""

    def test_build_missing_toml_error(self, tmp_path: Path):
        """No vten.toml in project dir raises error."""
        from vten.cli.build import build_project

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(Exception):
            build_project(str(empty_dir))

    def test_build_nonexistent_dir_error(self, tmp_path: Path):
        """build_project on nonexistent directory raises error."""
        from vten.cli.build import build_project

        with pytest.raises(Exception):
            build_project(str(tmp_path / "does_not_exist"))

    def test_build_invalid_toml_syntax(self, tmp_path: Path):
        """Invalid TOML syntax raises parse error."""
        from vten.cli.build import build_project

        project = tmp_path / "bad_proj"
        project.mkdir()
        (project / "vten.toml").write_text("not valid toml [[[")
        (project / "rtl").mkdir()
        (project / "build").mkdir()
        (project / "kernels" / "test").mkdir(parents=True)

        with pytest.raises(Exception):
            build_project(str(project))

    def test_build_missing_rtl_section_error(self, tmp_path: Path):
        """Missing [rtl] section raises error."""
        from vten.cli.build import build_project

        project = tmp_path / "no_rtl"
        project.mkdir()
        (project / "vten.toml").write_text("""\
[project]
name = "test"
version = "0.1.0"
""")
        (project / "build").mkdir()
        (project / "kernels" / "test").mkdir(parents=True)
        (project / "kernels" / "test" / "kernel_spec.yaml").write_text(
            yaml.dump(_minimal_kernel_spec("test"), default_flow_style=False)
        )

        with pytest.raises(Exception):
            build_project(str(project))

    def test_build_no_kernels_found_error(self, tmp_path: Path):
        """No kernels discovered raises error (or builds nothing)."""
        from vten.cli.build import build_project

        project = tmp_path / "no_kernels"
        project.mkdir()
        (project / "vten.toml").write_text(MINIMAL_TOML)
        (project / "rtl").mkdir()
        (project / "rtl" / "passthrough.sv").write_text("module passthrough; endmodule")
        (project / "build").mkdir()

        # No kernels/ directory at all — should either raise or succeed with warning
        with pytest.raises(Exception):
            build_project(str(project))


# ═══════════════════════════════════════════════════════════════════
# §8  SHM image — NOT generated at build time (§8.8)
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildNoSHM:
    """SHM image is NOT generated during build — it's created at run time (§8.8)."""

    def test_build_does_not_create_shm_dir(self, tmp_path: Path):
        """kernels/<name>/build/shm/ is NOT created during codegen."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")

        shm_dir = project / "kernels" / "passthrough" / "build" / "shm"
        # SHM dir should not exist or should be empty
        if shm_dir.exists():
            bin_files = list(shm_dir.glob("*.bin"))
            assert len(bin_files) == 0, "No .bin files should exist at build time"

    def test_build_does_not_produce_kernel_task_bin(self, tmp_path: Path):
        """kernel_task.bin is NOT present after codegen."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="codegen")

        # Search entire project tree
        bin_files = list(project.rglob("kernel_task.bin"))
        assert len(bin_files) == 0, "kernel_task.bin should only be created at run time"


# ═══════════════════════════════════════════════════════════════════
# §9  DPI-C shared library (Stage 2)
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildDPIC:
    """DPI-C shared library compilation — Stage 2 (§8.3)."""

    def test_build_lib_dir_exists(self, tmp_path: Path):
        """build/lib/ directory exists after Stage 2 (dpi_c)."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="dpi_c")
        assert (project / "build" / "lib").is_dir()

    def test_build_stage_dpi_c_creates_lib(self, tmp_path: Path):
        """--stage dpi_c creates build/lib/."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), stage="dpi_c")
        assert (project / "build" / "lib").is_dir()


# ═══════════════════════════════════════════════════════════════════
# §10  Split interface expansion
# ═══════════════════════════════════════════════════════════════════


class TestSplitInterfaceExpansion:
    """Split interfaces expand to N BFMs — one per physical port."""

    def test_no_split_returns_same_spec(self):
        """Spec without split returns unchanged."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axilite_ctrl",
                    protocol=Protocol.AXI4L,
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert result is spec  # Same object, not a copy

    def test_split_4_ports_expands_to_4_interfaces(self):
        """4-port split produces 4 individual interfaces."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": f"hbm_m{i:02d}_axi", "base_addr": 0}
                        for i in range(4)
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert len(result.interfaces) == 4
        assert "hbm" not in result.interfaces
        for i in range(4):
            name = f"hbm_m{i:02d}_axi"
            assert name in result.interfaces

    def test_split_32_ports_expands_to_32_interfaces(self):
        """32-port HBM split produces 32 individual interfaces."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="wgt", rtl_top="rtl/wgt.sv",
            interfaces={
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": f"hbm_m{i:02d}_axi", "base_addr": 0}
                        for i in range(32)
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert len(result.interfaces) == 32

    def test_expanded_ports_inherit_protocol(self):
        """Expanded ports inherit protocol from parent."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": "hbm_ch0", "base_addr": 0},
                        {"name": "hbm_ch1", "base_addr": 0},
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        for iface in result.interfaces.values():
            assert iface.protocol == Protocol.AXI4
            assert iface.data_width == 256
            assert iface.addr_width == 64

    def test_expanded_ports_use_port_name_as_rtl_port(self):
        """Each expanded port uses its port name as rtl_port."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": "hbm_ch0", "base_addr": 0},
                        {"name": "hbm_ch1", "base_addr": 0},
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert result.interfaces["hbm_ch0"].rtl_port == "hbm_ch0"
        assert result.interfaces["hbm_ch1"].rtl_port == "hbm_ch1"

    def test_non_split_interfaces_preserved(self):
        """Non-split interfaces are preserved alongside expanded ones."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axilite_ctrl",
                    protocol=Protocol.AXI4L, addr_width=16,
                ),
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": "hbm_ch0", "base_addr": 0},
                        {"name": "hbm_ch1", "base_addr": 0},
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert len(result.interfaces) == 3  # ctrl + 2 expanded
        assert "ctrl" in result.interfaces
        assert "hbm_ch0" in result.interfaces
        assert "hbm_ch1" in result.interfaces
        assert "hbm" not in result.interfaces

    def test_expanded_memory_region_inherited(self):
        """Expanded ports inherit memory_region from parent."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, MemoryRegion, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            memory_regions={
                "hbm": MemoryRegion(name="hbm", base=0, size=0x1_0000_0000),
            },
            interfaces={
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    memory_region="hbm",
                    split={"mode": "channel_interleave", "ports": [
                        {"name": "ch0", "base_addr": 0},
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert result.interfaces["ch0"].memory_region == "hbm"

    def test_expanded_packing_inherited(self):
        """Expanded ports inherit packing from parent."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, PackingScheme, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    packing=PackingScheme(element_width=8, elements_per_beat=32),
                    split={"mode": "channel_interleave", "ports": [
                        {"name": "ch0", "base_addr": 0},
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert result.interfaces["ch0"].packing is not None
        assert result.interfaces["ch0"].packing.element_width == 8
        assert result.interfaces["ch0"].packing.elements_per_beat == 32

    def test_derive_bfm_configs_with_split(self):
        """_derive_bfm_configs on expanded spec produces N BFMs."""
        from vten.cli.build import _derive_bfm_configs, _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axilite_ctrl",
                    protocol=Protocol.AXI4L, addr_width=16,
                ),
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": f"hbm_ch{i}", "base_addr": 0}
                        for i in range(4)
                    ]},
                ),
            },
        )
        expanded = _expand_split_interfaces(spec)
        configs = _derive_bfm_configs(expanded)
        assert len(configs) == 5  # 1 ctrl + 4 HBM
        names = {c.interface_name for c in configs}
        assert "ctrl" in names
        for i in range(4):
            assert f"hbm_ch{i}" in names

    def test_codegen_with_split_generates_tb(self, tmp_path: Path):
        """SVGenerator with expanded split spec generates valid tb_top.sv."""
        from vten.cli.build import _derive_bfm_configs, _expand_split_interfaces
        from vten.codegen.sv_generator import SVGenerator
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="wgt_loader", rtl_top="rtl/wgt.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axilite_ctrl",
                    protocol=Protocol.AXI4L, addr_width=16,
                ),
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": f"hbm_ch{i}", "base_addr": 0}
                        for i in range(4)
                    ]},
                ),
            },
        )
        expanded = _expand_split_interfaces(spec)
        configs = _derive_bfm_configs(expanded)
        config = {
            "project": {"name": "test"},
            "rtl": {"sources": [], "top_module": "wgt_loader"},
            "backend": {"xsim": {"vivado_path": "/tools/Xilinx"}},
        }
        gen = SVGenerator(kernel_spec=expanded, bfm_configs=configs, project_config=config)
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        assert "vten_bfm_axi4" in content
        assert "vten_bfm_axilite" in content
        # Each split port name should appear
        for i in range(4):
            assert f"hbm_ch{i}" in content

    def test_expanded_split_no_leftover_split_field(self):
        """Expanded ports have no split field."""
        from vten.cli.build import _expand_split_interfaces
        from vten.spec.models import InterfaceSpec, KernelSpec, Protocol

        spec = KernelSpec(
            kernel_name="test", rtl_top="rtl/test.sv",
            interfaces={
                "hbm": InterfaceSpec(
                    name="hbm", rtl_port="m_axi_hbm",
                    protocol=Protocol.AXI4, data_width=256,
                    split={"mode": "channel_interleave", "ports": [
                        {"name": "ch0", "base_addr": 0},
                    ]},
                ),
            },
        )
        result = _expand_split_interfaces(spec)
        assert result.interfaces["ch0"].split is None
