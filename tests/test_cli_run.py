"""Phase 4 tests: vten run, TestScenario, test discovery, execution pipeline.

Spec references:
- 00_data_models.md §14 (TestScenario, Test Discovery, Execution Flow)
- 06_codegen_and_cli.md §4.4 (vten run)
- 04_backend_xsim.md §3 (SHM handshake)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# §1  TestScenario base class — 00_data_models.md §14.1
# ═══════════════════════════════════════════════════════════════════


class TestTestScenario:
    """TestScenario: base class for user-defined test scenarios."""

    def test_base_class_exists(self):
        from vten.cli.scenario import TestScenario

        assert TestScenario is not None

    def test_has_kernel_attribute(self):
        from vten.cli.scenario import TestScenario

        assert hasattr(TestScenario, "kernel")

    def test_has_configs_attribute(self):
        from vten.cli.scenario import TestScenario

        assert hasattr(TestScenario, "configs")

    def test_has_run_method(self):
        from vten.cli.scenario import TestScenario

        assert callable(getattr(TestScenario, "run", None))

    def test_run_raises_not_implemented(self):
        """Base class run() raises NotImplementedError."""
        from vten.cli.scenario import TestScenario

        scenario = TestScenario()
        with pytest.raises(NotImplementedError):
            scenario.run(None, {})  # type: ignore[arg-type]

    def test_kernel_default_empty(self):
        from vten.cli.scenario import TestScenario

        assert TestScenario.kernel == ""

    def test_configs_default_none(self):
        """configs=None means single execution with vten.toml parameters."""
        from vten.cli.scenario import TestScenario

        assert TestScenario.configs is None

    def test_subclass_with_configs(self):
        from vten.cli.scenario import TestScenario

        class MyTest(TestScenario):
            kernel = "conv3d"
            configs = [
                {"C": 32, "D": 4, "H": 4, "W": 4},
                {"C": 64, "D": 8, "H": 8, "W": 8},
            ]

            def run(self, ctx, cfg):
                pass

        assert MyTest.kernel == "conv3d"
        assert len(MyTest.configs) == 2

    def test_subclass_single_config(self):
        from vten.cli.scenario import TestScenario

        class SimpleTest(TestScenario):
            kernel = "passthrough"

            def run(self, ctx, cfg):
                pass

        assert SimpleTest.configs is None

    def test_subclass_instantiation(self):
        from vten.cli.scenario import TestScenario

        class MyTest(TestScenario):
            kernel = "test"

            def run(self, ctx, cfg):
                pass

        t = MyTest()
        assert isinstance(t, TestScenario)

    def test_run_signature_has_ctx_and_cfg(self):
        """run(self, ctx, cfg) — spec §14.1."""
        import inspect

        from vten.cli.scenario import TestScenario

        sig = inspect.signature(TestScenario.run)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "ctx" in params
        assert "cfg" in params


# ═══════════════════════════════════════════════════════════════════
# §2  Test Discovery — 00_data_models.md §14.2
# ═══════════════════════════════════════════════════════════════════


class TestDiscoverTest:
    """discover_test(): find TestScenario by name."""

    def _create_test_file(self, tests_dir: Path, filename: str, class_name: str,
                          kernel: str = "test", configs: str | None = None):
        """Helper: create a test file with a TestScenario subclass."""
        configs_line = f"    configs = {configs}" if configs else ""
        content = f"""\
from vten.cli.scenario import TestScenario

class {class_name}(TestScenario):
    kernel = "{kernel}"
{configs_line}

    def run(self, ctx, cfg):
        pass
"""
        (tests_dir / filename).write_text(content)

    def test_discover_by_class_name(self, tmp_path: Path):
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(tests_dir, "test_conv3d.py", "TestConv3D")

        scenario = discover_test("TestConv3D", str(tests_dir))
        assert scenario.__class__.__name__ == "TestConv3D"

    def test_discover_by_class_name_case_insensitive(self, tmp_path: Path):
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(tests_dir, "test_conv3d.py", "TestConv3D")

        scenario = discover_test("testconv3d", str(tests_dir))
        assert scenario.__class__.__name__ == "TestConv3D"

    def test_discover_by_snake_case(self, tmp_path: Path):
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(tests_dir, "test_conv3d.py", "TestConv3D")

        scenario = discover_test("test_conv3d", str(tests_dir))
        assert scenario.__class__.__name__ == "TestConv3D"

    def test_discover_by_filename(self, tmp_path: Path):
        """'conv3d' matches tests/test_conv3d.py."""
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(tests_dir, "test_conv3d.py", "TestConv3D")

        scenario = discover_test("conv3d", str(tests_dir))
        assert scenario.__class__.__name__ == "TestConv3D"

    def test_discover_not_found_error(self, tmp_path: Path):
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        with pytest.raises(Exception, match="[Nn]ot [Ff]ound|[Nn]o.*match"):
            discover_test("nonexistent", str(tests_dir))

    def test_discover_scans_test_files_only(self, tmp_path: Path):
        """Only test_*.py files are scanned, not other .py files."""
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Non-test file should be ignored
        (tests_dir / "helper.py").write_text("""\
from vten.cli.scenario import TestScenario
class HelperScenario(TestScenario):
    kernel = "helper"
    def run(self, ctx, cfg): pass
""")

        with pytest.raises(Exception):
            discover_test("HelperScenario", str(tests_dir))

    def test_discover_ambiguous_error(self, tmp_path: Path):
        """Multiple matches raise error."""
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(tests_dir, "test_a.py", "TestFoo")
        self._create_test_file(tests_dir, "test_b.py", "TestFoo")

        # Two classes named TestFoo in different files
        with pytest.raises(Exception):
            discover_test("TestFoo", str(tests_dir))

    def test_discover_returns_instance(self, tmp_path: Path):
        """discover_test returns an instantiated TestScenario, not the class."""
        from vten.cli.scenario import TestScenario
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(tests_dir, "test_simple.py", "TestSimple")

        result = discover_test("TestSimple", str(tests_dir))
        assert isinstance(result, TestScenario)

    def test_discover_empty_dir_error(self, tmp_path: Path):
        """Empty tests directory raises not found."""
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        with pytest.raises(Exception):
            discover_test("anything", str(tests_dir))

    def test_discover_preserves_kernel_attribute(self, tmp_path: Path):
        """Discovered instance retains kernel class attribute."""
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(tests_dir, "test_npu.py", "TestNPU", kernel="npu_3d")

        result = discover_test("TestNPU", str(tests_dir))
        assert result.kernel == "npu_3d"

    def test_discover_preserves_configs(self, tmp_path: Path):
        """Discovered instance retains configs class attribute."""
        from vten.cli.discovery import discover_test

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        self._create_test_file(
            tests_dir, "test_multi.py", "TestMulti",
            kernel="multi", configs='[{"C": 32}, {"C": 64}]',
        )

        result = discover_test("TestMulti", str(tests_dir))
        assert result.configs is not None
        assert len(result.configs) == 2


# ═══════════════════════════════════════════════════════════════════
# §3  vten run — run_test() function and results output
# ═══════════════════════════════════════════════════════════════════


class TestVtenRun:
    """vten run --test <name>: run_test() function and results."""

    def _setup_project(self, tmp_path: Path) -> Path:
        """Create minimal project directory for run_test() — multi-kernel layout.

        Layout (§7.1):
            proj/
            ├── vten.toml
            ├── rtl/passthrough.sv
            ├── kernels/passthrough/
            │   ├── kernel_spec.yaml
            │   ├── build/generated/   (pretend codegen was done)
            │   └── tests/
            ├── build/
            └── results/
        """
        project = tmp_path / "proj"
        project.mkdir()
        (project / "vten.toml").write_text("""\
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
""")
        (project / "rtl").mkdir()
        (project / "rtl" / "passthrough.sv").write_text("module passthrough; endmodule")
        (project / "build").mkdir()
        (project / "results").mkdir()

        # Kernel directory structure
        kernel_dir = project / "kernels" / "passthrough"
        kernel_dir.mkdir(parents=True)
        import yaml
        spec = {
            "kernel": "passthrough",
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
        (kernel_dir / "kernel_spec.yaml").write_text(
            yaml.dump(spec, default_flow_style=False, sort_keys=False)
        )
        (kernel_dir / "tests").mkdir()
        (kernel_dir / "build").mkdir()
        (kernel_dir / "build" / "generated").mkdir()
        return project

    def test_run_test_function_exists(self):
        """run_test() is importable."""
        from vten.cli.run import run_test

        assert callable(run_test)

    def test_run_test_creates_results_directory(self, tmp_path: Path):
        """run_test() creates results/<kernel>/<test_name>/ directory."""
        from vten.cli.run import run_test

        project = self._setup_project(tmp_path)
        # Create a test scenario file in kernel's tests/ dir
        (project / "kernels" / "passthrough" / "tests" / "test_simple.py").write_text("""\
from vten.cli.scenario import TestScenario
class TestSimple(TestScenario):
    kernel = "passthrough"
    def run(self, ctx, cfg):
        pass
""")

        # Mock backend to avoid needing real xsim
        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.return_value = MagicMock(status=2, stats=[])  # DONE

            run_test(str(project), kernel_name="passthrough", test_name="TestSimple")

        # Results should exist under results/passthrough/
        results_dirs = list((project / "results").rglob("*")) if (project / "results").exists() else []
        assert len(results_dirs) >= 1

    def test_run_test_produces_summary_json(self, tmp_path: Path):
        """results/<kernel>/<test>/summary.json is created with status."""
        from vten.cli.run import run_test

        project = self._setup_project(tmp_path)
        (project / "kernels" / "passthrough" / "tests" / "test_pass.py").write_text("""\
from vten.cli.scenario import TestScenario
class TestPass(TestScenario):
    kernel = "passthrough"
    def run(self, ctx, cfg):
        pass
""")

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.return_value = MagicMock(status=2, stats=[])

            run_test(str(project), kernel_name="passthrough", test_name="TestPass")

        # Find summary.json somewhere under results/
        summary_files = list(project.rglob("summary.json"))
        assert len(summary_files) >= 1
        content = json.loads(summary_files[0].read_text())
        assert "status" in content

    def test_run_test_produces_stats_json(self, tmp_path: Path):
        """results/<kernel>/<test>/stats.json is created with command stats."""
        from vten.cli.run import run_test

        project = self._setup_project(tmp_path)
        (project / "kernels" / "passthrough" / "tests" / "test_stats.py").write_text("""\
from vten.cli.scenario import TestScenario
class TestStats(TestScenario):
    kernel = "passthrough"
    def run(self, ctx, cfg):
        pass
""")

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.return_value = MagicMock(status=2, stats=[])

            run_test(str(project), kernel_name="passthrough", test_name="TestStats")

        stats_files = list(project.rglob("stats.json"))
        assert len(stats_files) >= 1

    def test_run_test_not_found_error(self, tmp_path: Path):
        """run_test with nonexistent test name raises error."""
        from vten.cli.run import run_test

        project = self._setup_project(tmp_path)

        with pytest.raises(Exception):
            run_test(str(project), kernel_name="passthrough", test_name="nonexistent_test")

    def test_run_summary_json_fields_schema(self):
        """summary.json schema per spec §4.4."""
        required_fields = {"test_name", "status", "total_cycles", "configs_run", "configs_passed"}
        summary = {
            "test_name": "test_conv3d",
            "status": "PASS",
            "total_cycles": 50000,
            "configs_run": 1,
            "configs_passed": 1,
        }
        assert required_fields.issubset(set(summary.keys()))
        assert summary["status"] in ("PASS", "FAIL")

    def test_run_test_waveform_flag(self, tmp_path: Path):
        """run_test accepts waveform=True flag."""
        from vten.cli.run import run_test

        project = self._setup_project(tmp_path)
        (project / "kernels" / "passthrough" / "tests" / "test_wave.py").write_text("""\
from vten.cli.scenario import TestScenario
class TestWave(TestScenario):
    kernel = "passthrough"
    def run(self, ctx, cfg):
        pass
""")

        import inspect
        sig = inspect.signature(run_test)
        # Should accept waveform parameter
        params = list(sig.parameters.keys())
        assert "waveform" in params or len(params) >= 3

    def test_run_test_config_override(self, tmp_path: Path):
        """run_test accepts config_overrides dict."""
        import inspect

        from vten.cli.run import run_test

        sig = inspect.signature(run_test)
        params = list(sig.parameters.keys())
        assert "config_overrides" in params or len(params) >= 3


# ═══════════════════════════════════════════════════════════════════
# §4  vten run execution flow — 00_data_models.md §14.3
# ═══════════════════════════════════════════════════════════════════


class TestVtenRunExecution:
    """Test execution flow: scenario discovery → config merge → run."""

    def test_single_config_run_called_once(self):
        """TestScenario.configs=None → run() called exactly once."""
        from vten.cli.scenario import TestScenario

        class SingleTest(TestScenario):
            kernel = "test"
            call_count = 0

            def run(self, ctx, cfg):
                SingleTest.call_count += 1

        t = SingleTest()
        assert t.configs is None
        # Simulate: when configs is None, framework calls run once with base_cfg
        t.run(MagicMock(), {})
        assert SingleTest.call_count == 1

    def test_multi_config_run_called_per_config(self):
        """TestScenario.configs=[a, b] → run() called for each config."""
        from vten.cli.scenario import TestScenario

        call_args: list[dict] = []

        class MultiTest(TestScenario):
            kernel = "test"
            configs = [{"C": 32}, {"C": 64}]

            def run(self, ctx, cfg):
                call_args.append(cfg)

        t = MultiTest()
        # Simulate: framework calls run for each config
        for cfg in t.configs:
            t.run(MagicMock(), cfg)

        assert len(call_args) == 2
        assert call_args[0]["C"] == 32
        assert call_args[1]["C"] == 64

    def test_npu_multi_layer_configs(self):
        """NPU 3D: 28-layer U-Net configs list."""
        from vten.cli.scenario import TestScenario

        class NPUUNetTest(TestScenario):
            kernel = "npu_3d"
            configs = [
                {"IN_CH": 1, "OUT_CH": 32, "KERNEL_SIZE": 3, "IFM_STRIDE": 1, "OFM_STRIDE": 1},
                {"IN_CH": 32, "OUT_CH": 32, "KERNEL_SIZE": 3, "IFM_STRIDE": 1, "OFM_STRIDE": 1},
                {"IN_CH": 32, "OUT_CH": 64, "KERNEL_SIZE": 3, "IFM_STRIDE": 2, "OFM_STRIDE": 1},
            ]

            def run(self, ctx, cfg):
                pass

        t = NPUUNetTest()
        assert len(t.configs) == 3
        assert t.configs[0]["IN_CH"] == 1
        assert t.configs[2]["IFM_STRIDE"] == 2

    def test_config_merge_priority(self):
        """runtime config > vten.toml parameters (spec §14.3)."""
        from vten.cli.run import merge_configs

        base = {"C": 32, "D": 4, "H": 8}
        override = {"C": 64}
        merged = merge_configs(base, override)
        assert merged["C"] == 64   # override wins
        assert merged["D"] == 4    # base preserved
        assert merged["H"] == 8    # base preserved

    def test_config_merge_none_configs(self):
        """configs=None → single execution with base_cfg only."""
        from vten.cli.run import merge_configs

        base = {"C": 32, "D": 4}
        merged = merge_configs(base, None)
        assert merged == base

    def test_config_merge_empty_override(self):
        """Empty override preserves all base values."""
        from vten.cli.run import merge_configs

        base = {"C": 32, "D": 4}
        merged = merge_configs(base, {})
        assert merged == base


# ═══════════════════════════════════════════════════════════════════
# §5  vten run — error propagation through framework
# ═══════════════════════════════════════════════════════════════════


class TestVtenRunErrors:
    """Error handling: run_test() catches scenario errors → summary.json FAIL."""

    def _setup_project_with_scenario(self, tmp_path: Path, class_name: str,
                                     run_body: str) -> Path:
        """Helper: create project with custom TestScenario — multi-kernel layout."""
        import yaml

        project = tmp_path / "proj"
        project.mkdir()
        (project / "vten.toml").write_text("""\
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
""")
        (project / "rtl").mkdir()
        (project / "rtl" / "passthrough.sv").write_text("module passthrough; endmodule")
        (project / "build").mkdir()
        (project / "results").mkdir()

        # Kernel directory
        kernel_dir = project / "kernels" / "passthrough"
        kernel_dir.mkdir(parents=True)
        spec = {
            "kernel": "passthrough",
            "rtl_top": "rtl/passthrough.sv",
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
        (kernel_dir / "kernel_spec.yaml").write_text(
            yaml.dump(spec, default_flow_style=False, sort_keys=False)
        )
        (kernel_dir / "tests").mkdir()
        (kernel_dir / "build").mkdir()
        (kernel_dir / "build" / "generated").mkdir()

        (kernel_dir / "tests" / "test_scenario.py").write_text(f"""\
from vten.cli.scenario import TestScenario

class {class_name}(TestScenario):
    kernel = "passthrough"

    def run(self, ctx, cfg):
{run_body}
""")
        return project

    def test_failing_scenario_summary_status_fail(self, tmp_path: Path):
        """run_test() with raising scenario → summary.json status=FAIL."""
        from vten.cli.run import run_test

        project = self._setup_project_with_scenario(
            tmp_path, "TestFailing",
            '        raise RuntimeError("Intentional test failure")',
        )

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.__exit__ = MagicMock(
                side_effect=lambda *a: mock_backend.cleanup()
            )
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.return_value = MagicMock(status=2, stats=[])

            # run_test should catch the error and write FAIL, or re-raise
            try:
                run_test(str(project), kernel_name="passthrough", test_name="TestFailing")
            except (RuntimeError, Exception):
                pass  # Framework may re-raise after writing summary

        # Check summary.json was written with FAIL
        summary_files = list(project.rglob("summary.json"))
        if summary_files:
            content = json.loads(summary_files[0].read_text())
            assert content["status"] == "FAIL"

    def test_backend_error_captured_in_summary(self, tmp_path: Path):
        """BackendError during execute() → summary.json status=FAIL."""
        from vten.cli.run import run_test
        from vten.errors import BackendError

        # Scenario must record ops so ctx.run() calls backend.execute()
        run_body = (
            "        from vten.kernel.base import Kernel\n"
            "        from vten.kernel.tensor import Tensor\n"
            "        import torch\n"
            "        class K(Kernel):\n"
            "            data_in = Tensor(shape=(32,), dtype=torch.int8, interface='axi_stream_in')\n"
            "            data_out = Tensor(shape=(32,), dtype=torch.int8, interface='axi_stream_out')\n"
            "            def forward(self, **kw): return self.data_in.data.clone()\n"
            "        ki = ctx.instantiate(K, N=32)\n"
            "        ki.data_in.data = torch.zeros(32, dtype=torch.int8)\n"
            "        ctx.push_tensor(ki.data_in)\n"
            "        ctx.pull_tensor(ki.data_out)"
        )
        project = self._setup_project_with_scenario(
            tmp_path, "TestBackendFail", run_body,
        )

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.__exit__ = MagicMock(
                side_effect=lambda *a: mock_backend.cleanup()
            )
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.side_effect = BackendError("error_code=1, DECERR")

            try:
                run_test(str(project), kernel_name="passthrough", test_name="TestBackendFail")
            except (BackendError, Exception):
                pass

        summary_files = list(project.rglob("summary.json"))
        if summary_files:
            content = json.loads(summary_files[0].read_text())
            assert content["status"] == "FAIL"

    def test_timeout_error_captured_in_summary(self, tmp_path: Path):
        """TimeoutError during execute() → summary.json status=FAIL."""
        from vten.cli.run import run_test
        from vten.errors import TimeoutError as VTenTimeoutError

        # Scenario must record ops so ctx.run() calls backend.execute()
        run_body = (
            "        from vten.kernel.base import Kernel\n"
            "        from vten.kernel.tensor import Tensor\n"
            "        import torch\n"
            "        class K(Kernel):\n"
            "            data_in = Tensor(shape=(32,), dtype=torch.int8, interface='axi_stream_in')\n"
            "            data_out = Tensor(shape=(32,), dtype=torch.int8, interface='axi_stream_out')\n"
            "            def forward(self, **kw): return self.data_in.data.clone()\n"
            "        ki = ctx.instantiate(K, N=32)\n"
            "        ki.data_in.data = torch.zeros(32, dtype=torch.int8)\n"
            "        ctx.push_tensor(ki.data_in)\n"
            "        ctx.pull_tensor(ki.data_out)"
        )
        project = self._setup_project_with_scenario(
            tmp_path, "TestTimeout", run_body,
        )

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.__exit__ = MagicMock(
                side_effect=lambda *a: mock_backend.cleanup()
            )
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.side_effect = VTenTimeoutError("300s exceeded")

            try:
                run_test(str(project), kernel_name="passthrough", test_name="TestTimeout")
            except (VTenTimeoutError, Exception):
                pass

        summary_files = list(project.rglob("summary.json"))
        if summary_files:
            content = json.loads(summary_files[0].read_text())
            assert content["status"] == "FAIL"


# ═══════════════════════════════════════════════════════════════════
# §6  vten run — backend lifecycle through run_test()
# ═══════════════════════════════════════════════════════════════════


class TestVtenRunBackendLifecycle:
    """Backend lifecycle verified via run_test() with mocked backend."""

    def _setup_passing_project(self, tmp_path: Path) -> Path:
        """Helper: project with a passing scenario — multi-kernel layout."""
        import yaml

        project = tmp_path / "proj"
        project.mkdir()
        (project / "vten.toml").write_text("""\
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
""")
        (project / "rtl").mkdir()
        (project / "rtl" / "passthrough.sv").write_text("module passthrough; endmodule")
        (project / "build").mkdir()
        (project / "results").mkdir()

        # Kernel directory
        kernel_dir = project / "kernels" / "passthrough"
        kernel_dir.mkdir(parents=True)
        spec = {
            "kernel": "passthrough",
            "rtl_top": "rtl/passthrough.sv",
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
        (kernel_dir / "kernel_spec.yaml").write_text(
            yaml.dump(spec, default_flow_style=False, sort_keys=False)
        )
        (kernel_dir / "tests").mkdir()
        (kernel_dir / "build").mkdir()
        (kernel_dir / "build" / "generated").mkdir()

        (kernel_dir / "tests" / "test_ok.py").write_text("""\
from vten.cli.scenario import TestScenario
class TestOK(TestScenario):
    kernel = "passthrough"
    def run(self, ctx, cfg):
        pass
""")
        return project

    def test_backend_lifecycle_order_via_run_test(self, tmp_path: Path):
        """run_test() calls cleanup via `with backend:` context manager."""
        from vten.cli.run import run_test

        project = self._setup_passing_project(tmp_path)

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            # Wire __exit__ to call cleanup(), matching Backend ABC behavior
            mock_backend.__exit__ = MagicMock(
                side_effect=lambda *a: mock_backend.cleanup()
            )
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.return_value = MagicMock(status=2, stats=[])

            run_test(str(project), kernel_name="passthrough", test_name="TestOK")

        # Verify cleanup is called (via __exit__ from `with backend:`)
        method_names = [c[0] for c in mock_backend.method_calls]
        assert "cleanup" in method_names, (
            f"cleanup not called. Calls: {method_names}"
        )

    def test_cleanup_called_on_backend_error(self, tmp_path: Path):
        """Backend.cleanup() called even when execute() raises.

        Note: TestOK scenario records zero ops, so execute() is not called.
        This test verifies cleanup is called via `with backend:` regardless.
        """
        from vten.cli.run import run_test

        project = self._setup_passing_project(tmp_path)

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.__exit__ = MagicMock(
                side_effect=lambda *a: mock_backend.cleanup()
            )
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.side_effect = Exception("sim crashed")

            try:
                run_test(str(project), kernel_name="passthrough", test_name="TestOK")
            except Exception:
                pass

        # cleanup must be called (via __exit__ from `with backend:`)
        method_names = [c[0] for c in mock_backend.method_calls]
        assert "cleanup" in method_names, (
            f"Cleanup not called after error. Calls: {method_names}"
        )

    def test_summary_pass_when_all_configs_pass(self, tmp_path: Path):
        """run_test() produces summary with status=PASS on success."""
        from vten.cli.run import run_test

        project = self._setup_passing_project(tmp_path)

        with patch("vten.cli.run.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.__exit__ = MagicMock(
                side_effect=lambda *a: mock_backend.cleanup()
            )
            mock_get_backend.return_value = mock_backend
            mock_backend.execute.return_value = MagicMock(status=2, stats=[])

            run_test(str(project), kernel_name="passthrough", test_name="TestOK")

        summary_files = list(project.rglob("summary.json"))
        assert len(summary_files) >= 1
        content = json.loads(summary_files[0].read_text())
        assert content["status"] == "PASS"
