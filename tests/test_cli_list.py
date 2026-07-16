"""Tests for vten list (list_cmd.py).

Spec references:
- 06_codegen_and_cli.md §4 (CLI Commands)
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_project(tmp_path: Path, test_body: str) -> Path:
    """Create a minimal project with one kernel and one test file."""
    (tmp_path / "vten.toml").write_text(
        '[project]\nname = "listproj"\n'
    )
    tests_dir = tmp_path / "kernels" / "my_accel" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_my_accel.py").write_text(test_body)
    return tmp_path


class TestListTests:
    """list_tests: list a kernel's discovered scenarios."""

    def test_scenario_without_configs_does_not_crash(self, tmp_path, capsys):
        """Regression: configs defaults to None (single run with base
        params); list_tests crashed with TypeError: object of type
        'NoneType' has no len()."""
        from vten.cli.list_cmd import list_tests

        _make_project(
            tmp_path,
            "from vten.cli.scenario import TestScenario\n"
            "\n"
            "class TestNoConfigs(TestScenario):\n"
            '    """Runs once with base parameters."""\n'
            '    kernel = "my_accel"\n',
        )
        list_tests(str(tmp_path), "my_accel")

        out = capsys.readouterr().out
        assert "TestNoConfigs" in out
        # configs=None ⇒ one run with the base parameters
        assert "(1 config)" in out

    def test_scenario_with_configs_counts_them(self, tmp_path, capsys):
        from vten.cli.list_cmd import list_tests

        _make_project(
            tmp_path,
            "from vten.cli.scenario import TestScenario\n"
            "\n"
            "class TestSweep(TestScenario):\n"
            '    """Parameter sweep."""\n'
            '    kernel = "my_accel"\n'
            '    configs = [{"N": 32}, {"N": 64}, {"N": 128}]\n',
        )
        list_tests(str(tmp_path), "my_accel")

        out = capsys.readouterr().out
        assert "TestSweep" in out
        assert "(3 config)" in out

    def test_missing_tests_dir_raises(self, tmp_path):
        from vten.cli.list_cmd import list_tests
        from vten.errors import VTenError

        (tmp_path / "vten.toml").write_text('[project]\nname = "p"\n')
        with pytest.raises(VTenError, match="tests directory not found"):
            list_tests(str(tmp_path), "nonexistent")
