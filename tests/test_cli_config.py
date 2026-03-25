"""Phase 4 tests: vten.toml parsing and validation.

Spec references:
- 06_codegen_and_cli.md §6 (vten.toml Reference)
- specs/npu_3d_analysis.md §11.4 (NPU 3D config)
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _write_toml(path: Path, content: str) -> Path:
    toml_file = path / "vten.toml"
    toml_file.write_text(content)
    return toml_file


MINIMAL_TOML = """\
[project]
name = "test_proj"
version = "0.1.0"
"""

FULL_TOML = """\
[project]
name = "my_npu"
version = "1.0.0"

[parameters]
C = 64
D = 32
H = 32
W = 32

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0
submit_timeout_s = 300

[backend.scheduler]
max_bfms = 48
max_ifaces = 48
max_cmds = 512

[backend.verilator]
verilator_path = "/usr/bin/verilator"
threads = 4

[rtl]
sources = ["rtl/**/*.sv", "rtl/**/*.v"]
top_module = "npu_top"
include_dirs = ["rtl/include"]

[test]
default_seed = 42
waveform = false
waveform_on_fail = true

[report]
format = "terminal"
"""

NPU_3D_TOML = """\
[project]
name = "npu_3d"
version = "0.1.0"

[parameters]
IN_CH = 64
OUT_CH = 128
IN_DEPTH = 8
IN_HEIGHT = 16
IN_WIDTH = 16
KERNEL_SIZE = 3
IFM_STRIDE = 1
OFM_STRIDE = 1
IS_CONCAT = 0
CONCAT_CH = 0
BIAS_SHIFT = 8
IS_RELU = 1

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0
submit_timeout_s = 300

[backend.scheduler]
max_bfms = 40
max_ifaces = 42
max_cmds = 256

[rtl]
sources = ["design/**/*.sv"]
top_module = "NPU_3D_top"
include_dirs = ["design/include"]

[test]
default_seed = 42
waveform = false
waveform_on_fail = true

[report]
format = "terminal"
"""


# ═══════════════════════════════════════════════════════════════════
# §1  vten.toml parsing
# ═══════════════════════════════════════════════════════════════════


class TestVtenTomlParsing:
    """vten.toml file parsing and validation."""

    def test_parse_minimal_toml(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, MINIMAL_TOML)
        config = load_project_config(tmp_path)
        assert config["project"]["name"] == "test_proj"
        assert config["project"]["version"] == "0.1.0"

    def test_parse_full_toml(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, FULL_TOML)
        config = load_project_config(tmp_path)
        assert config["project"]["name"] == "my_npu"
        assert "parameters" in config
        assert "backend" in config
        assert "rtl" in config
        assert "test" in config
        assert "report" in config

    def test_parameters_section(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, FULL_TOML)
        config = load_project_config(tmp_path)
        params = config["parameters"]
        assert params["C"] == 64
        assert params["D"] == 32
        assert params["H"] == 32
        assert params["W"] == 32

    def test_backend_xsim_section(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, FULL_TOML)
        config = load_project_config(tmp_path)
        xsim = config["backend"]["xsim"]
        assert xsim["vivado_path"] == "/tools/Xilinx/Vivado/2023.2"
        assert xsim["compile_options"] == ["-timescale", "1ns/1ps"]
        assert xsim["timeout_ms"] == 0
        assert xsim["submit_timeout_s"] == 300

    def test_backend_scheduler_optional(self, tmp_path: Path):
        """[backend.scheduler] absent is valid — defaults used."""
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, MINIMAL_TOML)
        config = load_project_config(tmp_path)
        # scheduler section either absent or defaults
        scheduler = config.get("backend", {}).get("scheduler", {})
        assert isinstance(scheduler, dict)

    def test_backend_scheduler_override(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, FULL_TOML)
        config = load_project_config(tmp_path)
        sched = config["backend"]["scheduler"]
        assert sched["max_bfms"] == 48
        assert sched["max_ifaces"] == 48
        assert sched["max_cmds"] == 512

    def test_rtl_section(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, FULL_TOML)
        config = load_project_config(tmp_path)
        rtl = config["rtl"]
        assert rtl["sources"] == ["rtl/**/*.sv", "rtl/**/*.v"]
        assert rtl["top_module"] == "npu_top"
        assert rtl["include_dirs"] == ["rtl/include"]

    def test_test_section(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, FULL_TOML)
        config = load_project_config(tmp_path)
        test_cfg = config["test"]
        assert test_cfg["default_seed"] == 42
        assert test_cfg["waveform"] is False
        assert test_cfg["waveform_on_fail"] is True

    def test_report_section(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, FULL_TOML)
        config = load_project_config(tmp_path)
        assert config["report"]["format"] == "terminal"

    def test_missing_project_section_error(self, tmp_path: Path):
        """Error on missing [project] section."""
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, "[rtl]\nsources = []\n")
        with pytest.raises(Exception):  # VTenError or ValueError
            load_project_config(tmp_path)

    def test_toml_file_not_found_error(self, tmp_path: Path):
        """Error when vten.toml does not exist."""
        from vten.cli.config import load_project_config

        with pytest.raises((FileNotFoundError, Exception)):
            load_project_config(tmp_path)

    def test_invalid_toml_syntax_error(self, tmp_path: Path):
        """Error on invalid TOML syntax."""
        from vten.cli.config import load_project_config

        (tmp_path / "vten.toml").write_text("this is not valid toml {{{}}")
        with pytest.raises(Exception):
            load_project_config(tmp_path)

    def test_returns_dict(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, MINIMAL_TOML)
        config = load_project_config(tmp_path)
        assert isinstance(config, dict)


# ═══════════════════════════════════════════════════════════════════
# §2  NPU 3D full config
# ═══════════════════════════════════════════════════════════════════


class TestVtenTomlNPU:
    """NPU 3D vten.toml with 12 parameters and scheduler overrides."""

    def test_npu_parameters_count(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, NPU_3D_TOML)
        config = load_project_config(tmp_path)
        params = config["parameters"]
        assert len(params) == 12

    def test_npu_conv3d_parameters(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, NPU_3D_TOML)
        config = load_project_config(tmp_path)
        params = config["parameters"]
        assert params["IN_CH"] == 64
        assert params["OUT_CH"] == 128
        assert params["IN_DEPTH"] == 8
        assert params["KERNEL_SIZE"] == 3
        assert params["BIAS_SHIFT"] == 8
        assert params["IS_RELU"] == 1

    def test_npu_scheduler_config(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, NPU_3D_TOML)
        config = load_project_config(tmp_path)
        sched = config["backend"]["scheduler"]
        assert sched["max_bfms"] == 40
        assert sched["max_ifaces"] == 42

    def test_npu_rtl_top_module(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, NPU_3D_TOML)
        config = load_project_config(tmp_path)
        assert config["rtl"]["top_module"] == "NPU_3D_top"

    def test_npu_submit_timeout(self, tmp_path: Path):
        from vten.cli.config import load_project_config

        _write_toml(tmp_path, NPU_3D_TOML)
        config = load_project_config(tmp_path)
        assert config["backend"]["xsim"]["submit_timeout_s"] == 300
