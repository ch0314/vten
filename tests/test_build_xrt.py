"""Phase B tests for XrtBuildPipeline.

Sections:
  §1 Initialization
  §2 Stage execution
  §3 Registry integration
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from vten.build.base import BuildPipeline
from vten.build.xrt_build import XrtBuildPipeline
from vten.errors import BuildError


# ── Fixtures ──

MINIMAL_SPEC_YAML = dedent("""\
    kernel: test_kern
    rtl_top: test_kern_top
    interfaces:
      data_in:
        protocol: axi4
        data_width: 256
        addr_width: 64
        rtl_port: m_axi_data_in
""")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Create a minimal project directory."""
    return tmp_path


@pytest.fixture()
def config() -> dict:
    """Minimal project config dict."""
    return {}


@pytest.fixture()
def pipeline(project: Path, config: dict) -> XrtBuildPipeline:
    return XrtBuildPipeline(project, config)


@pytest.fixture()
def kernel_dir(tmp_path: Path) -> Path:
    """Create a kernel directory with a minimal kernel_spec.yaml."""
    kdir = tmp_path / "kernels" / "test_kern"
    kdir.mkdir(parents=True)
    (kdir / "kernel_spec.yaml").write_text(MINIMAL_SPEC_YAML)
    return kdir


# ═══════════════════════════════════════════════════════════════
# §1 XrtBuildPipeline initialization
# ═══════════════════════════════════════════════════════════════


class TestXrtBuildPipelineInit:
    """§1 Constructor and basic properties."""

    def test_constructor(self, project: Path, config: dict) -> None:
        pipe = XrtBuildPipeline(project, config)
        assert pipe._project == project
        assert pipe._config is config

    def test_is_build_pipeline_subclass(self) -> None:
        assert issubclass(XrtBuildPipeline, BuildPipeline)

    def test_stages_returns_correct_list(self, pipeline: XrtBuildPipeline) -> None:
        expected = ["gen_packaging", "gen_xo", "gen_link_cfg", "validate"]
        assert pipeline.stages() == expected

    def test_stages_returns_new_list(self, pipeline: XrtBuildPipeline) -> None:
        """stages() should return a copy, not the internal list."""
        s1 = pipeline.stages()
        s2 = pipeline.stages()
        assert s1 == s2
        assert s1 is not s2

    def test_project_level_stages_returns_gen_link_cfg(
        self, pipeline: XrtBuildPipeline
    ) -> None:
        result = pipeline.project_level_stages()
        assert set(result) == {"gen_link_cfg"}

    def test_project_level_stages_returns_list(
        self, pipeline: XrtBuildPipeline
    ) -> None:
        result = pipeline.project_level_stages()
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════
# §2 Stage execution
# ═══════════════════════════════════════════════════════════════


class TestStageExecution:
    """§2 run_stage for each stage name."""

    def test_gen_packaging_creates_artifacts(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """gen_packaging should create package_ip.tcl, kernel.xml, xo_gen.tcl,
        and connectivity.cfg under kernel_dir/build/."""
        pipeline.run_stage("gen_packaging", "test_kern", kernel_dir, force=False)

        build_dir = kernel_dir / "build"
        packaging = build_dir / "packaging"
        assert packaging.is_dir()
        assert (packaging / "package_ip.tcl").exists()
        assert (packaging / "kernel.xml").exists()
        assert (packaging / "xo_gen.tcl").exists()

        link = build_dir / "link"
        assert (link / "connectivity.cfg").exists()

    def test_gen_packaging_no_spec_raises(
        self, pipeline: XrtBuildPipeline, tmp_path: Path
    ) -> None:
        """gen_packaging raises BuildError when kernel_spec.yaml is missing."""
        empty_dir = tmp_path / "kernels" / "missing"
        empty_dir.mkdir(parents=True)
        with pytest.raises(BuildError, match="kernel_spec.yaml not found"):
            pipeline.run_stage("gen_packaging", "missing", empty_dir, force=False)

    def test_gen_xo_succeeds_after_packaging(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """gen_xo should succeed when xo_gen.tcl already exists."""
        # First run gen_packaging to create xo_gen.tcl
        pipeline.run_stage("gen_packaging", "test_kern", kernel_dir, force=False)
        # Then gen_xo should verify it exists without error
        pipeline.run_stage("gen_xo", "test_kern", kernel_dir, force=False)

    def test_gen_xo_raises_without_packaging(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """gen_xo raises BuildError when xo_gen.tcl does not exist."""
        with pytest.raises(BuildError, match="xo_gen.tcl not found"):
            pipeline.run_stage("gen_xo", "test_kern", kernel_dir, force=False)

    def test_gen_link_cfg_project_level(
        self, pipeline: XrtBuildPipeline
    ) -> None:
        """gen_link_cfg runs at project level (no kernel_dir needed)."""
        # Should not raise even with no kernels discovered
        pipeline.run_stage("gen_link_cfg", None, None, force=False)

    def test_gen_link_cfg_with_kernel(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """gen_link_cfg discovers kernels and reports their connectivity."""
        # First generate packaging so link cfg exists
        pipeline.run_stage("gen_packaging", "test_kern", kernel_dir, force=False)
        # gen_link_cfg uses project root to discover kernels
        pipeline.run_stage("gen_link_cfg", None, None, force=False)

    def test_validate_skips_when_no_xclbin(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """validate should skip gracefully when no xclbin is configured."""
        # No xclbin_path in config => skip
        pipeline.run_stage("validate", "test_kern", kernel_dir, force=False)

    def test_validate_skips_when_xclbin_path_missing(
        self, tmp_path: Path, kernel_dir: Path
    ) -> None:
        """validate should skip when xclbin_path points to non-existent file."""
        config = {"backend": {"xrt": {"xclbin_path": "/nonexistent/path.xclbin"}}}
        pipe = XrtBuildPipeline(tmp_path, config)
        # Should not raise, just skip
        pipe.run_stage("validate", "test_kern", kernel_dir, force=False)

    def test_unknown_stage_raises_build_error(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """Unknown stage name raises BuildError."""
        with pytest.raises(BuildError, match="Unknown XRT build stage"):
            pipeline.run_stage("nonexistent", "test_kern", kernel_dir, force=False)


# ═══════════════════════════════════════════════════════════════
# §3 Registry integration
# ═══════════════════════════════════════════════════════════════


class TestRegistryIntegration:
    """§3 get_build_pipeline returns XrtBuildPipeline for 'xrt'."""

    def test_get_build_pipeline_returns_xrt(self, tmp_path: Path) -> None:
        from vten.backend.registry import get_build_pipeline

        pipe = get_build_pipeline("xrt", tmp_path, {})
        assert isinstance(pipe, XrtBuildPipeline)

    def test_get_build_pipeline_sets_project_and_config(
        self, tmp_path: Path
    ) -> None:
        from vten.backend.registry import get_build_pipeline

        config = {"key": "value"}
        pipe = get_build_pipeline("xrt", tmp_path, config)
        assert pipe._project == tmp_path
        assert pipe._config is config
