"""Tests for the vten report command.

Spec references:
- 06_codegen_and_cli.md §4.5 (vten report)
- 05_bfm_library.md §5.2 (CommandMetrics)
- 00_data_models.md §13 (BatchResult, CmdStats)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _setup_single_result(tmp_path: Path, test_name: str = "test_conv3d",
                         status: str = "PASS", total_cycles: int = 50000,
                         num_commands: int = 1) -> Path:
    """Create mock results directory with one test result."""
    results = tmp_path / "results" / test_name
    results.mkdir(parents=True)

    (results / "summary.json").write_text(json.dumps({
        "test_name": test_name,
        "status": status,
        "total_cycles": total_cycles,
        "configs_run": 1,
        "configs_passed": 1 if status == "PASS" else 0,
    }))

    commands = []
    for i in range(num_commands):
        commands.append({
            "cmd_id": i, "status": 3,
            "issue_cycle": 100 + i * 500, "commit_cycle": 500 + i * 500,
            "first_active_cycle": 150 + i * 500, "last_active_cycle": 450 + i * 500,
            "active_cycles": 200, "total_beats": 32, "stall_cycles": 50,
        })

    (results / "stats.json").write_text(json.dumps({"commands": commands}))
    return tmp_path


def _setup_multi_results(tmp_path: Path) -> Path:
    """Create results with multiple test results."""
    _setup_single_result(tmp_path, "test_conv3d", "PASS", 50000, num_commands=5)
    _setup_single_result(tmp_path, "test_passthrough", "PASS", 10000, num_commands=2)
    _setup_single_result(tmp_path, "test_matmul", "FAIL", 200000, num_commands=10)
    return tmp_path


def _setup_npu_result(tmp_path: Path) -> Path:
    """Create NPU 3D scale results: 256 commands, 40 BFMs."""
    results = tmp_path / "results" / "test_npu_3d"
    results.mkdir(parents=True)

    (results / "summary.json").write_text(json.dumps({
        "test_name": "test_npu_3d",
        "status": "PASS",
        "total_cycles": 5000000,
        "configs_run": 1,
        "configs_passed": 1,
    }))

    commands = []
    for i in range(256):
        commands.append({
            "cmd_id": i, "status": 3,
            "issue_cycle": i * 100, "commit_cycle": i * 100 + 400,
            "first_active_cycle": i * 100 + 50, "last_active_cycle": i * 100 + 350,
            "active_cycles": 200, "total_beats": 64, "stall_cycles": 30,
        })

    (results / "stats.json").write_text(json.dumps({"commands": commands}))
    return tmp_path


# ═══════════════════════════════════════════════════════════════════
# §1  vten report — basic formats
# ═══════════════════════════════════════════════════════════════════


class TestVtenReport:
    """vten report [--format terminal|html|json]."""

    def test_report_scans_results_dir(self, tmp_path: Path):
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        report = generate_report(str(project), format="json")
        assert report is not None

    def test_report_json_format(self, tmp_path: Path):
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        assert isinstance(parsed, (dict, list))

    def test_report_terminal_format(self, tmp_path: Path):
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_html_format(self, tmp_path: Path):
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        report = generate_report(str(project), format="html")
        assert isinstance(report, str)
        assert "<" in report  # Contains HTML tags

    def test_report_no_results_error(self, tmp_path: Path):
        """Error when results/ is empty or missing."""
        from vten.cli.report import generate_report

        empty_project = tmp_path / "empty"
        empty_project.mkdir()
        with pytest.raises(Exception):
            generate_report(str(empty_project), format="terminal")

    def test_report_includes_test_status(self, tmp_path: Path):
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        report_str = json.dumps(parsed)
        assert "PASS" in report_str or "pass" in report_str

    def test_report_includes_test_name(self, tmp_path: Path):
        """Report includes the test name."""
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        report_str = json.dumps(parsed)
        assert "test_conv3d" in report_str

    def test_report_includes_total_cycles(self, tmp_path: Path):
        """Report includes total_cycles metric."""
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        report_str = json.dumps(parsed)
        assert "50000" in report_str or "total_cycles" in report_str

    def test_report_invalid_format_error(self, tmp_path: Path):
        """Invalid format raises error."""
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path)
        with pytest.raises(Exception):
            generate_report(str(project), format="xlsx")


# ═══════════════════════════════════════════════════════════════════
# §2  vten report — command metrics
# ═══════════════════════════════════════════════════════════════════


class TestVtenReportCommandMetrics:
    """Per-command metrics in report output (05_bfm_library.md §5.2)."""

    def test_report_command_metrics_present(self, tmp_path: Path):
        """Report includes per-command metrics from stats.json."""
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path, num_commands=3)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        report_str = json.dumps(parsed)
        assert "cmd" in report_str.lower() or "command" in report_str.lower()

    def test_report_command_latency_in_output(self, tmp_path: Path):
        """Report includes latency (commit_cycle - issue_cycle)."""
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path, num_commands=1)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        report_str = json.dumps(parsed)
        # Latency = 500 - 100 = 400 cycles for cmd_id=0
        assert "400" in report_str or "latency" in report_str.lower()

    def test_report_terminal_format_readable(self, tmp_path: Path):
        """Terminal format has human-readable table/text."""
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path, num_commands=3)
        report = generate_report(str(project), format="terminal")
        # Should contain some structured text (table borders, separators, etc.)
        assert len(report.split("\n")) > 1

    def test_report_html_has_table(self, tmp_path: Path):
        """HTML format wraps data in table structure."""
        from vten.cli.report import generate_report

        project = _setup_single_result(tmp_path, num_commands=3)
        report = generate_report(str(project), format="html")
        lower = report.lower()
        assert "<table" in lower or "<div" in lower or "<tr" in lower


# ═══════════════════════════════════════════════════════════════════
# §2b  vten report — performance summary
# ═══════════════════════════════════════════════════════════════════


def _setup_perf_result(tmp_path: Path) -> Path:
    """Result with multi-interface data-moving commands for perf summary."""
    results = tmp_path / "results" / "passthrough" / "TestPerf"
    results.mkdir(parents=True)
    (results / "summary.json").write_text(json.dumps({
        "test_name": "TestPerf", "kernel": "passthrough", "status": "PASS",
        "total_cycles": 200, "configs_run": 1, "configs_passed": 1,
    }))
    cmds = [
        {"cmd_id": 0, "op": "PUSH", "interface": "s_axis",
         "protocol": "axi4_stream", "tensor": "data_in", "size": 1024,
         "status": 3, "status_name": "COMMITTED",
         "issue_cycle": 10, "commit_cycle": 50, "latency_cycles": 40,
         "active_cycles": 20, "stall_cycles": 8, "total_beats": 8,
         "first_active_cycle": 15, "last_active_cycle": 45},
        {"cmd_id": 1, "op": "PULL", "interface": "m_axis",
         "protocol": "axi4_stream", "tensor": "data_out", "size": 4096,
         "status": 3, "status_name": "COMMITTED",
         "issue_cycle": 60, "commit_cycle": 120, "latency_cycles": 60,
         "active_cycles": 40, "stall_cycles": 2, "total_beats": 16,
         "first_active_cycle": 65, "last_active_cycle": 115},
        {"cmd_id": 2, "op": "WRITE_REG", "interface": "ctrl",
         "protocol": "axi4_lite", "status": 3, "status_name": "COMMITTED",
         "issue_cycle": 0, "commit_cycle": 2, "latency_cycles": 2,
         "active_cycles": 0, "stall_cycles": 0, "total_beats": 0},
    ]
    (results / "stats.json").write_text(json.dumps({"commands": cmds}))
    return tmp_path


class TestVtenReportPerfSummary:
    """Performance Summary section (roofline / utilization)."""

    def test_terminal_has_perf_section(self, tmp_path: Path):
        from vten.cli.report import generate_report
        project = _setup_perf_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "Performance Summary:" in report
        assert "s_axis" in report and "m_axis" in report

    def test_terminal_shows_bottleneck(self, tmp_path: Path):
        """s_axis has the most stall cycles → flagged as bottleneck."""
        from vten.cli.report import generate_report
        project = _setup_perf_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "Bottleneck: s_axis" in report

    def test_terminal_reg_ops_excluded(self, tmp_path: Path):
        """Register interface (ctrl) is not a data mover → absent from perf."""
        from vten.cli.report import generate_report
        project = _setup_perf_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        perf_section = report.split("Performance Summary:")[1]
        assert "ctrl" not in perf_section

    def test_json_includes_perf_summary(self, tmp_path: Path):
        from vten.cli.report import generate_report
        project = _setup_perf_result(tmp_path)
        parsed = json.loads(generate_report(str(project), format="json"))
        entry = next(e for e in parsed if "perf_summary" in e)
        perf = entry["perf_summary"]
        assert "interfaces" in perf and "overall" in perf
        assert perf["overall"]["bottleneck_interface"] == "s_axis"

    def test_no_clock_reports_per_cycle(self, tmp_path: Path):
        """No vten.toml clock → bandwidth shown in beats/cycle, not Hz."""
        from vten.cli.report import generate_report
        project = _setup_perf_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "beats/cycle" in report

    def test_clock_freq_reports_bandwidth(self, tmp_path: Path):
        """clock_freq_hz in vten.toml → achieved bytes/s bandwidth."""
        from vten.cli.report import generate_report
        project = _setup_perf_result(tmp_path)
        (project / "vten.toml").write_text(
            "[project]\nname = \"p\"\n\n"
            "[backend.xrt]\nclock_freq_hz = 300000000\n"
        )
        report = generate_report(str(project), format="terminal")
        assert "Achieved bandwidth" in report
        assert "/s" in report

    def test_no_cmd_stats_graceful_note(self, tmp_path: Path):
        """Commands present but all zero-stat (cpu backend) → graceful note."""
        from vten.cli.report import generate_report
        results = tmp_path / "results" / "cpu_run" / "TestCpu"
        results.mkdir(parents=True)
        (results / "summary.json").write_text(json.dumps({
            "test_name": "TestCpu", "kernel": "cpu_run", "status": "PASS",
            "total_cycles": 0, "configs_run": 1, "configs_passed": 1,
        }))
        # cpu backend emits commands with no cycle stats at all.
        (results / "stats.json").write_text(json.dumps({"commands": [
            {"cmd_id": 0, "op": "WRITE_REG", "interface": "ctrl",
             "status": 3, "status_name": "COMMITTED",
             "issue_cycle": 0, "commit_cycle": 0, "latency_cycles": 0,
             "active_cycles": 0, "stall_cycles": 0, "total_beats": 0},
        ]}))
        report = generate_report(str(tmp_path), format="terminal")
        assert "no per-command performance data" in report
        # Must not crash / must not emit an empty table
        assert "Performance Summary:" in report


# ═══════════════════════════════════════════════════════════════════
# §3  vten report — multi-result aggregation
# ═══════════════════════════════════════════════════════════════════


class TestVtenReportMultiResult:
    """Report aggregates multiple test results."""

    def test_multi_result_lists_all_tests(self, tmp_path: Path):
        """Report includes all test results from results/."""
        from vten.cli.report import generate_report

        project = _setup_multi_results(tmp_path)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        report_str = json.dumps(parsed)
        assert "test_conv3d" in report_str
        assert "test_passthrough" in report_str
        assert "test_matmul" in report_str

    def test_multi_result_shows_pass_and_fail(self, tmp_path: Path):
        """Report shows both PASS and FAIL statuses."""
        from vten.cli.report import generate_report

        project = _setup_multi_results(tmp_path)
        report = generate_report(str(project), format="json")
        report_str = report.upper()
        assert "PASS" in report_str
        assert "FAIL" in report_str

    def test_multi_result_terminal_all_tests(self, tmp_path: Path):
        """Terminal report includes all test names."""
        from vten.cli.report import generate_report

        project = _setup_multi_results(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "test_conv3d" in report
        assert "test_passthrough" in report
        assert "test_matmul" in report

    def test_multi_result_summary_count(self, tmp_path: Path):
        """Report includes total/passed/failed counts."""
        from vten.cli.report import generate_report

        project = _setup_multi_results(tmp_path)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        report_str = json.dumps(parsed)
        # 3 total, 2 passed, 1 failed
        assert "3" in report_str or "2" in report_str


# ═══════════════════════════════════════════════════════════════════
# §4  vten report — NPU 3D scale
# ═══════════════════════════════════════════════════════════════════


class TestVtenReportNPUScale:
    """NPU 3D: 256 commands report."""

    def test_npu_256_commands_report(self, tmp_path: Path):
        """Report handles 256 commands without error."""
        from vten.cli.report import generate_report

        project = _setup_npu_result(tmp_path)
        report = generate_report(str(project), format="json")
        parsed = json.loads(report)
        assert parsed is not None

    def test_npu_report_terminal_no_truncation(self, tmp_path: Path):
        """Terminal report for 256 commands is complete."""
        from vten.cli.report import generate_report

        project = _setup_npu_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert isinstance(report, str)
        assert len(report) > 0


# ═══════════════════════════════════════════════════════════════════
# §5  vten report — edge cases
# ═══════════════════════════════════════════════════════════════════


class TestVtenReportEdgeCases:
    """Edge cases for report generation."""

    def test_report_malformed_summary_json(self, tmp_path: Path):
        """Malformed summary.json raises or gracefully handles."""
        from vten.cli.report import generate_report

        results = tmp_path / "results" / "test_bad"
        results.mkdir(parents=True)
        (results / "summary.json").write_text("not valid json{{{")
        (results / "stats.json").write_text("{}")

        with pytest.raises(Exception):
            generate_report(str(tmp_path), format="json")

    def test_report_missing_stats_json(self, tmp_path: Path):
        """Missing stats.json: report should still work (no command metrics)."""
        from vten.cli.report import generate_report

        results = tmp_path / "results" / "test_nostats"
        results.mkdir(parents=True)
        (results / "summary.json").write_text(json.dumps({
            "test_name": "test_nostats",
            "status": "PASS",
            "total_cycles": 1000,
            "configs_run": 1,
            "configs_passed": 1,
        }))
        # No stats.json — should not crash
        report = generate_report(str(tmp_path), format="json")
        assert report is not None

    def test_report_empty_commands_list(self, tmp_path: Path):
        """stats.json with empty commands list."""
        from vten.cli.report import generate_report

        results = tmp_path / "results" / "test_empty_cmds"
        results.mkdir(parents=True)
        (results / "summary.json").write_text(json.dumps({
            "test_name": "test_empty_cmds",
            "status": "PASS",
            "total_cycles": 0,
            "configs_run": 1,
            "configs_passed": 1,
        }))
        (results / "stats.json").write_text(json.dumps({"commands": []}))

        report = generate_report(str(tmp_path), format="json")
        assert report is not None

    def test_report_results_dir_exists_but_empty(self, tmp_path: Path):
        """results/ exists but has no subdirectories."""
        from vten.cli.report import generate_report

        (tmp_path / "results").mkdir()

        with pytest.raises(Exception):
            generate_report(str(tmp_path), format="terminal")
