"""Phase 4 tests: vten CLI entry point (main.py).

Spec references:
- 06_codegen_and_cli.md §4 (CLI Commands)

Tests cover:
- Argument parsing for all subcommands
- Subcommand dispatch
- Config override parsing
- Error handling (no command, invalid args)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# §1  Argument Parsing — basic structure
# ═══════════════════════════════════════════════════════════════════


class TestMainEntryPoint:
    """main() function exists and is callable."""

    def test_main_exists(self):
        from vten.cli.main import main

        assert callable(main)

    def test_no_command_exits_with_error(self):
        """No subcommand → prints help and exits with code 1."""
        from vten.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_help_flag_exits_zero(self):
        """--help exits with code 0."""
        from vten.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_invalid_command_exits_error(self):
        """Invalid subcommand → argparse error exit."""
        from vten.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent"])
        assert exc_info.value.code != 0


# ═══════════════════════════════════════════════════════════════════
# §2  vten init dispatch
# ═══════════════════════════════════════════════════════════════════


class TestMainInitDispatch:
    """main() dispatches 'init' subcommand to init_project()."""

    @patch("vten.cli.init_cmd.init_project")
    def test_init_calls_init_project(self, mock_init):
        from vten.cli.main import main

        main(["init", "/tmp/my_project"])
        mock_init.assert_called_once_with(
            "/tmp/my_project", kernel_name=None, backend=None, add_backend=None,
        )

    @patch("vten.cli.init_cmd.init_project")
    def test_init_with_kernel_flag(self, mock_init):
        from vten.cli.main import main

        main(["init", "/tmp/my_project", "--kernel", "conv3d"])
        mock_init.assert_called_once_with(
            "/tmp/my_project", kernel_name="conv3d", backend=None, add_backend=None,
        )


# ═══════════════════════════════════════════════════════════════════
# §3  vten build dispatch
# ═══════════════════════════════════════════════════════════════════


class TestMainBuildDispatch:
    """main() dispatches 'build' subcommand to build_project()."""

    @patch("vten.cli.build.build_project")
    def test_build_default_args(self, mock_build):
        from vten.cli.main import main

        main(["build"])
        mock_build.assert_called_once_with(
            project_dir=".",
            kernel_name=None,
            backend=None,
            stage=None,
            upto=None,
            force=False,
            clean=False,
            skip_compile=False,
            config_overrides=None,
        )

    @patch("vten.cli.build.build_project")
    def test_build_with_project_dir(self, mock_build):
        from vten.cli.main import main

        main(["build", "--project", "/home/user/my_npu"])
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["project_dir"] == "/home/user/my_npu"

    @patch("vten.cli.build.build_project")
    def test_build_with_kernel(self, mock_build):
        from vten.cli.main import main

        main(["build", "--kernel", "conv3d"])
        assert mock_build.call_args.kwargs["kernel_name"] == "conv3d"

    @patch("vten.cli.build.build_project")
    def test_build_with_stage(self, mock_build):
        from vten.cli.main import main

        main(["build", "--stage", "codegen"])
        assert mock_build.call_args.kwargs["stage"] == "codegen"

    @patch("vten.cli.build.build_project")
    def test_build_with_upto(self, mock_build):
        from vten.cli.main import main

        main(["build", "--upto", "dpi_c"])
        assert mock_build.call_args.kwargs["upto"] == "dpi_c"

    @patch("vten.cli.build.build_project")
    def test_build_with_force(self, mock_build):
        from vten.cli.main import main

        main(["build", "--force"])
        assert mock_build.call_args.kwargs["force"] is True

    @patch("vten.cli.build.build_project")
    def test_build_with_skip_compile(self, mock_build):
        from vten.cli.main import main

        main(["build", "--skip-compile"])
        assert mock_build.call_args.kwargs["skip_compile"] is True

    @patch("vten.cli.build.build_project")
    def test_build_stage_passed_through(self, mock_build):
        """--stage value is passed through to build_project (validated by pipeline)."""
        from vten.cli.main import main

        main(["build", "--stage", "codegen"])
        assert mock_build.call_args.kwargs["stage"] == "codegen"


# ═══════════════════════════════════════════════════════════════════
# §4  Config override parsing
# ═══════════════════════════════════════════════════════════════════


class TestMainConfigOverrides:
    """--config K=V parsing for build and run commands."""

    @patch("vten.cli.build.build_project")
    def test_build_config_override_string(self, mock_build):
        from vten.cli.main import main

        main(["build", "--config", "name=test"])
        overrides = mock_build.call_args.kwargs["config_overrides"]
        assert overrides == {"name": "test"}

    @patch("vten.cli.build.build_project")
    def test_build_config_override_integer(self, mock_build):
        """Numeric values are parsed as int."""
        from vten.cli.main import main

        main(["build", "--config", "N=1024"])
        overrides = mock_build.call_args.kwargs["config_overrides"]
        assert overrides == {"N": 1024}
        assert isinstance(overrides["N"], int)

    @patch("vten.cli.build.build_project")
    def test_build_config_multiple_overrides(self, mock_build):
        from vten.cli.main import main

        main(["build", "--config", "C=32", "D=4"])
        overrides = mock_build.call_args.kwargs["config_overrides"]
        assert overrides == {"C": 32, "D": 4}

    @patch("vten.cli.build.build_project")
    def test_build_no_config_passes_none(self, mock_build):
        from vten.cli.main import main

        main(["build"])
        assert mock_build.call_args.kwargs["config_overrides"] is None

    @patch("vten.cli.build.build_project")
    def test_build_config_value_with_equals(self, mock_build):
        """K=V where V contains '=' — split on first '=' only."""
        from vten.cli.main import main

        main(["build", "--config", "path=/opt/x=y"])
        overrides = mock_build.call_args.kwargs["config_overrides"]
        assert overrides == {"path": "/opt/x=y"}


# ═══════════════════════════════════════════════════════════════════
# §5  vten run dispatch
# ═══════════════════════════════════════════════════════════════════


class TestMainRunDispatch:
    """main() dispatches 'run' subcommand to run_test()."""

    @patch("vten.cli.run.run_test")
    def test_run_requires_kernel(self, mock_run):
        """--kernel is required for vten run."""
        from vten.cli.main import main

        with pytest.raises(SystemExit):
            main(["run", "--test", "TestFoo"])

    @patch("vten.cli.run.run_test")
    def test_run_without_test_runs_all(self, mock_run):
        """--test is optional; omitting it runs all tests."""
        from vten.cli.main import main

        main(["run", "--kernel", "conv3d"])
        mock_run.assert_called_once_with(
            project_dir=".",
            kernel_name="conv3d",
            test_name="",
            backend=None,
            waveform=False,
            waveform_on_fail=False,
            gui=False,
            sim_verbose=False,
            config_overrides=None,
            verify=False,
        )

    @patch("vten.cli.run.run_test")
    def test_run_dispatches_correctly(self, mock_run):
        from vten.cli.main import main

        main(["run", "--kernel", "conv3d", "--test", "TestConv3D"])
        mock_run.assert_called_once_with(
            project_dir=".",
            kernel_name="conv3d",
            test_name="TestConv3D",
            backend=None,
            waveform=False,
            waveform_on_fail=False,
            gui=False,
            sim_verbose=False,
            config_overrides=None,
            verify=False,
        )

    @patch("vten.cli.run.run_test")
    def test_run_with_waveform(self, mock_run):
        from vten.cli.main import main

        main(["run", "--kernel", "k", "--test", "t", "--waveform"])
        assert mock_run.call_args.kwargs["waveform"] is True

    @patch("vten.cli.run.run_test")
    def test_run_with_gui(self, mock_run):
        from vten.cli.main import main

        main(["run", "--kernel", "k", "--test", "t", "--gui"])
        assert mock_run.call_args.kwargs["gui"] is True

    @patch("vten.cli.run.run_test")
    def test_run_with_config_overrides(self, mock_run):
        from vten.cli.main import main

        main(["run", "--kernel", "k", "--test", "t", "--config", "N=256"])
        overrides = mock_run.call_args.kwargs["config_overrides"]
        assert overrides == {"N": 256}

    @patch("vten.cli.run.run_test")
    def test_run_with_project_dir(self, mock_run):
        from vten.cli.main import main

        main(["run", "--kernel", "k", "--test", "t", "--project", "/my/proj"])
        assert mock_run.call_args.kwargs["project_dir"] == "/my/proj"


# ═══════════════════════════════════════════════════════════════════
# §6  vten report dispatch
# ═══════════════════════════════════════════════════════════════════


class TestMainReportDispatch:
    """main() dispatches 'report' subcommand to generate_report()."""

    @patch("vten.cli.report.generate_report", return_value="Report output")
    def test_report_default(self, mock_report, capsys):
        from vten.cli.main import main

        main(["report"])
        mock_report.assert_called_once_with(".", format="terminal")
        captured = capsys.readouterr()
        assert "Report output" in captured.out

    @patch("vten.cli.report.generate_report", return_value="{}")
    def test_report_json_format(self, mock_report, capsys):
        from vten.cli.main import main

        main(["report", "--format", "json"])
        assert mock_report.call_args.kwargs["format"] == "json"

    @patch("vten.cli.report.generate_report", return_value="")
    def test_report_with_project_dir(self, mock_report, capsys):
        from vten.cli.main import main

        main(["report", "--project-dir", "/my/proj"])
        mock_report.assert_called_once_with("/my/proj", format="terminal")

    def test_report_invalid_format_rejected(self):
        """Invalid --format value rejected by argparse."""
        from vten.cli.main import main

        with pytest.raises(SystemExit):
            main(["report", "--format", "pdf"])


# ═══════════════════════════════════════════════════════════════════
# §7  Stage choices validation
# ═══════════════════════════════════════════════════════════════════


class TestMainStageChoices:
    """Build stage/upto choices match the 5-stage pipeline."""

    def test_valid_stage_choices(self):
        """All 5 pipeline stages are accepted."""
        from vten.cli.main import main

        valid_stages = ["project_setup", "dpi_c", "codegen", "compile_order", "compile"]
        for stage in valid_stages:
            # Should not raise SystemExit for valid stage names
            # (build_project will be called; we just test parsing)
            with patch("vten.cli.build.build_project"):
                main(["build", "--stage", stage])  # should not raise

    def test_valid_upto_choices(self):
        """All 5 pipeline stages are accepted for --upto."""
        from vten.cli.main import main

        with patch("vten.cli.build.build_project"):
            main(["build", "--upto", "codegen"])  # should not raise
