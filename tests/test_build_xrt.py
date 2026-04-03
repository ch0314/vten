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
        expected = [
            "gen_codegen", "gen_xrt_packaging",
            "package_ip", "gen_xo", "gen_link_config",
            "vpp_link", "validate",
        ]
        assert pipeline.stages() == expected

    def test_stages_returns_new_list(self, pipeline: XrtBuildPipeline) -> None:
        """stages() should return a copy, not the internal list."""
        s1 = pipeline.stages()
        s2 = pipeline.stages()
        assert s1 == s2
        assert s1 is not s2

    def test_project_level_stages_returns_empty(
        self, pipeline: XrtBuildPipeline
    ) -> None:
        result = pipeline.project_level_stages()
        assert result == []

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

    def test_gen_codegen_creates_wrapper(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """gen_codegen should create wrapper.sv under kernel_dir/build/xrt/."""
        pipeline.run_stage("gen_codegen", "test_kern", kernel_dir, force=False)

        build_dir = kernel_dir / "build" / "xrt"
        assert build_dir.is_dir()
        # Should have at least the wrapper file
        sv_files = list(build_dir.glob("*.sv"))
        assert len(sv_files) >= 1

    def test_gen_codegen_no_spec_raises(
        self, pipeline: XrtBuildPipeline, tmp_path: Path
    ) -> None:
        """gen_codegen raises BuildError when kernel_spec.yaml is missing."""
        empty_dir = tmp_path / "kernels" / "missing"
        empty_dir.mkdir(parents=True)
        with pytest.raises(BuildError, match="kernel_spec.yaml not found"):
            pipeline.run_stage("gen_codegen", "missing", empty_dir, force=False)

    def test_gen_xrt_packaging_creates_artifacts(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """gen_xrt_packaging should create TCL/XML/CFG/build script."""
        pipeline.run_stage("gen_xrt_packaging", "test_kern", kernel_dir, force=False)

        build_dir = kernel_dir / "build" / "xrt"
        assert (build_dir / "package_ip.tcl").exists()
        assert (build_dir / "kernel.xml").exists()
        assert (build_dir / "gen_xo.tcl").exists()
        assert (build_dir / "connectivity.cfg").exists()
        # Build script
        sh_files = list(build_dir.glob("build_*.sh"))
        assert len(sh_files) >= 1

    def test_gen_xrt_packaging_no_spec_raises(
        self, pipeline: XrtBuildPipeline, tmp_path: Path
    ) -> None:
        """gen_xrt_packaging raises BuildError when kernel_spec.yaml is missing."""
        empty_dir = tmp_path / "kernels" / "missing"
        empty_dir.mkdir(parents=True)
        with pytest.raises(BuildError, match="kernel_spec.yaml not found"):
            pipeline.run_stage("gen_xrt_packaging", "missing", empty_dir, force=False)

    def test_validate_skips_when_no_xclbin(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """validate should skip gracefully when no xclbin is configured."""
        pipeline.run_stage("validate", "test_kern", kernel_dir, force=False)

    def test_validate_skips_when_xclbin_path_missing(
        self, tmp_path: Path, kernel_dir: Path
    ) -> None:
        """validate should skip when xclbin_path points to non-existent file."""
        config = {"backend": {"xrt": {"xclbin_path": "/nonexistent/path.xclbin"}}}
        pipe = XrtBuildPipeline(tmp_path, config)
        pipe.run_stage("validate", "test_kern", kernel_dir, force=False)

    def test_package_ip_no_tcl_raises(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """package_ip raises BuildError when package_ip.tcl is missing."""
        with pytest.raises(BuildError, match="package_ip.tcl not found"):
            pipeline.run_stage("package_ip", "test_kern", kernel_dir, force=False)

    def test_gen_xo_no_tcl_raises(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """gen_xo raises BuildError when gen_xo.tcl is missing."""
        with pytest.raises(BuildError, match="gen_xo.tcl not found"):
            pipeline.run_stage("gen_xo", "test_kern", kernel_dir, force=False)

    def test_vpp_link_no_xo_raises(
        self, pipeline: XrtBuildPipeline, kernel_dir: Path
    ) -> None:
        """vpp_link raises BuildError when XO file is missing."""
        with pytest.raises(BuildError, match="XO not found"):
            pipeline.run_stage("vpp_link", "test_kern", kernel_dir, force=False)

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
