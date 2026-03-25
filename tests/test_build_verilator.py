"""Phase C tests: VerilatorBuildPipeline — stages, configuration, registry.

Spec reference: 08_backend_abstraction.md §8.2
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from vten.build.base import BuildPipeline
from vten.build.verilator_build import VerilatorBuildPipeline
from vten.errors import BuildError


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
    return tmp_path


@pytest.fixture()
def config() -> dict:
    return {
        "project": {"name": "test_proj"},
        "backend": {
            "verilator": {
                "verilator_path": "verilator",
                "threads": 4,
                "trace": False,
            },
        },
    }


@pytest.fixture()
def pipeline(project: Path, config: dict) -> VerilatorBuildPipeline:
    return VerilatorBuildPipeline(project, config)


@pytest.fixture()
def kernel_dir(tmp_path: Path) -> Path:
    kdir = tmp_path / "kernels" / "test_kern"
    kdir.mkdir(parents=True)
    (kdir / "kernel_spec.yaml").write_text(MINIMAL_SPEC_YAML)
    return kdir


# ═══════════════════════════════════════════════════════════════
# §1 VerilatorBuildPipeline initialization
# ═══════════════════════════════════════════════════════════════


class TestVerilatorBuildInit:
    """§1 VerilatorBuildPipeline constructor and metadata."""

    def test_is_build_pipeline_subclass(self, pipeline: VerilatorBuildPipeline) -> None:
        assert isinstance(pipeline, BuildPipeline)

    def test_stores_project_and_config(
        self, project: Path, config: dict
    ) -> None:
        p = VerilatorBuildPipeline(project, config)
        assert p._project == project
        assert p._config is config

    def test_stages_returns_four(self, pipeline: VerilatorBuildPipeline) -> None:
        assert pipeline.stages() == ["dpi_c", "codegen", "verilate", "make"]

    def test_stages_returns_new_list(self, pipeline: VerilatorBuildPipeline) -> None:
        s1 = pipeline.stages()
        s2 = pipeline.stages()
        assert s1 == s2
        assert s1 is not s2

    def test_project_level_stages(self, pipeline: VerilatorBuildPipeline) -> None:
        assert set(pipeline.project_level_stages()) == {"dpi_c"}

    def test_config_defaults(self, project: Path) -> None:
        p = VerilatorBuildPipeline(project, {"backend": {}})
        assert p._verilator_bin == "verilator"
        assert p._threads == 4
        assert p._trace is False
        assert p._opt_level == 3
        assert p._extra_args == []


# ═══════════════════════════════════════════════════════════════
# §2 Stage execution
# ═══════════════════════════════════════════════════════════════


class TestVerilatorStageExecution:
    """§2 run_stage dispatch and error handling."""

    def test_unknown_stage_raises(
        self, pipeline: VerilatorBuildPipeline, kernel_dir: Path
    ) -> None:
        with pytest.raises(BuildError, match="Unknown Verilator build stage"):
            pipeline.run_stage("invalid", "test_kern", kernel_dir, force=False)

    def test_codegen_creates_files(
        self, pipeline: VerilatorBuildPipeline, kernel_dir: Path
    ) -> None:
        """codegen should create tb_top.sv under kernel_dir/build/generated/."""
        pipeline.run_stage("codegen", "test_kern", kernel_dir, force=False)
        generated = kernel_dir / "build" / "generated"
        assert generated.is_dir()
        assert (generated / "tb_top.sv").exists()

    def test_codegen_missing_spec_raises(
        self, pipeline: VerilatorBuildPipeline, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "kernels" / "empty"
        empty_dir.mkdir(parents=True)
        with pytest.raises(BuildError, match="kernel_spec.yaml not found"):
            pipeline.run_stage("codegen", "empty", empty_dir, force=False)

    def test_verilate_missing_tb_top_raises(
        self, pipeline: VerilatorBuildPipeline, kernel_dir: Path
    ) -> None:
        with pytest.raises(BuildError, match="tb_top.sv not found"):
            pipeline.run_stage("verilate", "test_kern", kernel_dir, force=False)

    def test_make_missing_makefile_raises(
        self, pipeline: VerilatorBuildPipeline, kernel_dir: Path
    ) -> None:
        with pytest.raises(BuildError, match="Vtb_top.mk not found"):
            pipeline.run_stage("make", "test_kern", kernel_dir, force=False)


# ═══════════════════════════════════════════════════════════════
# §3 Registry integration
# ═══════════════════════════════════════════════════════════════


class TestVerilatorRegistryIntegration:
    """§3 Pipeline discoverable via registry."""

    def test_get_build_pipeline_returns_verilator(self, config: dict) -> None:
        from vten.backend.registry import get_build_pipeline
        pipeline = get_build_pipeline("verilator", Path("/tmp"), config)
        assert isinstance(pipeline, VerilatorBuildPipeline)

    def test_pipeline_project_path(self, config: dict) -> None:
        from vten.backend.registry import get_build_pipeline
        pipeline = get_build_pipeline("verilator", Path("/tmp/test"), config)
        assert pipeline._project == Path("/tmp/test")
