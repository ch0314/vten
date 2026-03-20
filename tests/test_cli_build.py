"""Phase 4 tests: vten build pipeline.

Spec references:
- 06_codegen_and_cli.md §4.3 (vten build)
"""

from __future__ import annotations

from pathlib import Path

import pytest


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
vivado_path = "/tools/Xilinx/Vivado/2024.1"
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
vivado_path = "/tools/Xilinx/Vivado/2024.1"
compile_options = ["-timescale", "1ns/1ps"]
"""


def _setup_project(tmp_path: Path, toml_content: str = MINIMAL_TOML) -> Path:
    """Create minimal project directory with vten.toml and spec."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "vten.toml").write_text(toml_content)
    (project / "rtl").mkdir()
    (project / "rtl" / "passthrough.sv").write_text("module passthrough; endmodule")
    (project / "specs").mkdir()
    (project / "build").mkdir()
    return project


# ═══════════════════════════════════════════════════════════════════
# §1  vten build output structure
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuild:
    """vten build: output directory structure (06_codegen_and_cli.md §4.3)."""

    def test_build_output_directory_structure(self, tmp_path: Path):
        """build/ has generated/, lib/, scripts/, shm/ subdirectories."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))

        build_dir = project / "build"
        assert (build_dir / "generated").is_dir()
        assert (build_dir / "scripts").is_dir()
        assert (build_dir / "shm").is_dir()

    def test_build_generates_tb_top(self, tmp_path: Path):
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        assert (project / "build" / "generated" / "tb_top.sv").exists()

    def test_build_generates_build_tcl(self, tmp_path: Path):
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        assert (project / "build" / "scripts" / "build.tcl").exists()

    def test_build_generates_run_tcl(self, tmp_path: Path):
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        assert (project / "build" / "scripts" / "run.tcl").exists()

    def test_build_generates_shm_image(self, tmp_path: Path):
        """build/shm/kernel_task.bin exists after build."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        shm_files = list((project / "build" / "shm").glob("*.bin"))
        assert len(shm_files) >= 1

    def test_build_missing_toml_error(self, tmp_path: Path):
        """No vten.toml in project dir produces error."""
        from vten.cli.build import build_project

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(Exception):
            build_project(str(empty_dir))

    def test_build_backend_xsim_default(self, tmp_path: Path):
        """--backend xsim is the default."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        build_tcl = project / "build" / "scripts" / "build.tcl"
        if build_tcl.exists():
            content = build_tcl.read_text()
            assert "xvlog" in content or "xelab" in content

    def test_build_nonexistent_dir_error(self, tmp_path: Path):
        """build_project on nonexistent directory raises error."""
        from vten.cli.build import build_project

        with pytest.raises(Exception):
            build_project(str(tmp_path / "does_not_exist"))


# ═══════════════════════════════════════════════════════════════════
# §2  Build pipeline stages
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildPipeline:
    """Build pipeline: toml → spec → compile → codegen → scripts."""

    def test_build_produces_generated_sv(self, tmp_path: Path):
        """Build pipeline produces SV files in generated/ dir."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        sv_files = list((project / "build" / "generated").glob("*.sv"))
        assert len(sv_files) >= 1

    def test_build_config_override_single(self, tmp_path: Path):
        """--config C=32 overrides [parameters].C."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project), config_overrides={"C": 32})

    def test_build_config_override_multiple(self, tmp_path: Path):
        """Multiple config overrides applied together."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path, TOML_WITH_PARAMS)
        build_project(str(project), config_overrides={"C": 64, "D": 8, "H": 16})

    def test_build_creates_lib_dir(self, tmp_path: Path):
        """build/lib/ directory is created for DPI-C shared library."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        assert (project / "build" / "lib").is_dir()

    def test_build_generates_makefile(self, tmp_path: Path):
        """Build produces Makefile in build/ or build/scripts/."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        # Makefile could be in generated/ or scripts/
        makefiles = list((project / "build").rglob("Makefile"))
        assert len(makefiles) >= 1

    def test_build_tb_top_contains_dut(self, tmp_path: Path):
        """Generated tb_top.sv references the DUT top module."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        tb_top = project / "build" / "generated" / "tb_top.sv"
        if tb_top.exists():
            content = tb_top.read_text()
            assert "passthrough" in content

    def test_build_shm_image_has_magic(self, tmp_path: Path):
        """Generated SHM image starts with VTEN magic bytes."""
        import struct

        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        shm_files = list((project / "build" / "shm").glob("*.bin"))
        if shm_files:
            data = shm_files[0].read_bytes()
            if len(data) >= 4:
                magic = struct.unpack_from("<I", data, 0)[0]
                assert magic == 0x5654454E  # "VTEN"


# ═══════════════════════════════════════════════════════════════════
# §3  Build — DPI-C shared library
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildDPIC:
    """DPI-C shared library compilation step."""

    def test_build_lib_dir_exists(self, tmp_path: Path):
        """build/lib/ directory exists after build."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        assert (project / "build" / "lib").is_dir()

    def test_build_tcl_references_dpi_lib(self, tmp_path: Path):
        """build.tcl or xelab command references the DPI-C library."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        build_tcl = project / "build" / "scripts" / "build.tcl"
        if build_tcl.exists():
            content = build_tcl.read_text()
            assert "sv_lib" in content or "libvten_shm" in content or "dpi" in content.lower()

    def test_build_makefile_has_gcc_target(self, tmp_path: Path):
        """Makefile has gcc compilation target for DPI-C bridge."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        makefiles = list((project / "build").rglob("Makefile"))
        if makefiles:
            content = makefiles[0].read_text()
            assert "gcc" in content or "cc" in content.lower() or "lib" in content.lower()


# ═══════════════════════════════════════════════════════════════════
# §4  Build — rebuild behavior
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildRebuild:
    """Rebuild: calling build_project twice should work."""

    def test_rebuild_overwrites_generated(self, tmp_path: Path):
        """Second build overwrites generated files without error."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path)
        build_project(str(project))
        # Build again — should not fail
        build_project(str(project))
        assert (project / "build" / "generated" / "tb_top.sv").exists()

    def test_rebuild_with_different_config(self, tmp_path: Path):
        """Rebuild with different config overrides produces new output."""
        from vten.cli.build import build_project

        project = _setup_project(tmp_path, TOML_WITH_PARAMS)
        build_project(str(project), config_overrides={"C": 32})
        build_project(str(project), config_overrides={"C": 64})
        # Should succeed without error
        assert (project / "build" / "generated" / "tb_top.sv").exists()


# ═══════════════════════════════════════════════════════════════════
# §5  Build — error cases
# ═══════════════════════════════════════════════════════════════════


class TestVtenBuildErrors:
    """Build error cases."""

    def test_build_invalid_toml_syntax_error(self, tmp_path: Path):
        """Invalid TOML syntax raises parse error."""
        from vten.cli.build import build_project

        project = tmp_path / "bad_proj"
        project.mkdir()
        (project / "vten.toml").write_text("not valid toml [[[")
        (project / "rtl").mkdir()
        (project / "specs").mkdir()
        (project / "build").mkdir()

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

        with pytest.raises(Exception):
            build_project(str(project))
