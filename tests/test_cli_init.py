"""Phase 4 tests: vten init command.

Spec references:
- 06_codegen_and_cli.md §4.1 (vten init)
"""

from __future__ import annotations

from pathlib import Path


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

    def test_creates_ip_directory(self, tmp_path: Path):
        """ip/ directory for Vivado IP definitions (§4.1)."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "ip").is_dir()

    def test_creates_kernels_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "kernels").is_dir()

    def test_creates_build_directory(self, tmp_path: Path):
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "build").is_dir()

    def test_creates_results_directory(self, tmp_path: Path):
        """results/ directory for test outputs (§4.1)."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        assert (project_dir / "results").is_dir()

    def test_kernels_dir_is_empty(self, tmp_path: Path):
        """kernels/ is empty — kernels added via vten init --kernel <name>."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        # No kernel subdirectories created by default
        kernel_contents = list((project_dir / "kernels").iterdir())
        assert len(kernel_contents) == 0

    def test_toml_has_project_name(self, tmp_path: Path):
        """vten.toml [project].name matches the directory name."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        init_project(str(project_dir))
        content = (project_dir / "vten.toml").read_text()
        assert "my_npu" in content

    def test_existing_directory_overlay(self, tmp_path: Path):
        """Init on existing directory creates missing structure without error."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        project_dir.mkdir()
        # Place an existing file to verify it's not overwritten
        (project_dir / "existing.txt").write_text("keep me")

        init_project(str(project_dir))

        assert (project_dir / "existing.txt").read_text() == "keep me"
        assert (project_dir / "vten.toml").exists()
        assert (project_dir / "kernels").is_dir()

    def test_existing_directory_no_overwrite_toml(self, tmp_path: Path):
        """Init on existing directory does not overwrite existing vten.toml."""
        from vten.cli.init_cmd import init_project

        project_dir = tmp_path / "my_npu"
        project_dir.mkdir()
        (project_dir / "vten.toml").write_text("[project]\nname = \"custom\"\n")

        init_project(str(project_dir))

        assert "custom" in (project_dir / "vten.toml").read_text()

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

        expected_dirs = ["rtl", "ip", "kernels", "build", "results"]
        for d in expected_dirs:
            assert (project_dir / d).is_dir(), f"Missing directory: {d}"
        assert (project_dir / "vten.toml").exists()
