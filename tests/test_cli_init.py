"""Phase 4 tests: vten init command.

Spec references:
- 06_codegen_and_cli.md §4.1 (vten init)
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════
# §1  vten init
# ═══════════════════════════════════════════════════════════════════


class TestVtenInit:
    """vten init <name>: project skeleton creation."""

    def test_creates_project_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert project_dir.exists()
        assert project_dir.is_dir()

    def test_creates_vten_toml(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        toml_file = project_dir / "vten.toml"
        assert toml_file.exists()

    def test_vten_toml_is_valid(self, tmp_path: Path):
        """Generated vten.toml is parseable."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        content = (project_dir / "vten.toml").read_text()
        # Should contain [project] section at minimum
        assert "[project]" in content

    def test_creates_rtl_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "rtl").is_dir()

    def test_creates_specs_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "specs").is_dir()

    def test_creates_kernels_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "kernels").is_dir()

    def test_creates_tests_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "tests").is_dir()

    def test_creates_build_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "build").is_dir()

    def test_creates_example_kernel(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        # kernels/ should contain an example .py file
        kernel_files = list((project_dir / "kernels").glob("*.py"))
        assert len(kernel_files) >= 1

    def test_creates_example_test(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        test_files = list((project_dir / "tests").glob("test_*.py"))
        assert len(test_files) >= 1

    def test_toml_has_project_name(self, tmp_path: Path):
        """vten.toml [project].name matches the directory name."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        content = (project_dir / "vten.toml").read_text()
        assert "my_npu" in content

    def test_existing_directory_error(self, tmp_path: Path):
        """Error if target directory already exists."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        project_dir.mkdir()
        with pytest.raises(Exception):  # FileExistsError or VTenError
            init_project(str(project_dir))

    def test_nested_path(self, tmp_path: Path):
        """Init in a nested parent directory."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "workspace" / "my_npu"
        (tmp_path / "workspace").mkdir()
        init_project(str(project_dir))
        assert project_dir.exists()
        assert (project_dir / "vten.toml").exists()

    def test_full_directory_structure(self, tmp_path: Path):
        """Verify complete skeleton per spec §4.1."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))

        expected_dirs = ["rtl", "specs", "kernels", "tests", "build"]
        for d in expected_dirs:
            assert (project_dir / d).is_dir(), f"Missing directory: {d}"
        assert (project_dir / "vten.toml").exists()
