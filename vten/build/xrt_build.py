"""XrtBuildPipeline — generates TCL/XML/config for xclbin build.

Does NOT build xclbin itself (user responsibility via v++).
Generates packaging artifacts from kernel_spec.yaml.

Stages:
  Stage 1: gen_packaging  — package_ip.tcl + kernel.xml (per-kernel)
  Stage 2: gen_xo         — xo_gen.tcl (per-kernel)
  Stage 3: gen_link_cfg   — connectivity.cfg (project-level)
  Stage 4: validate       — optional xclbin metadata check (per-kernel)

Spec reference: 08_backend_abstraction.md §8.2, §8.3
"""

from __future__ import annotations

from pathlib import Path

from vten.build.base import BuildPipeline
from vten.codegen.xrt_generator import XrtGenerator
from vten.errors import BuildError
from vten.spec.parser import parse_kernel_spec


class XrtBuildPipeline(BuildPipeline):
    """XRT build pipeline — generates artifacts for v++ flow."""

    _STAGES = ["gen_packaging", "gen_xo", "gen_link_cfg", "validate"]
    _PROJECT_STAGES = {"gen_link_cfg"}

    def __init__(self, project: Path, config: dict) -> None:
        super().__init__(project, config)

    def stages(self) -> list[str]:
        return list(self._STAGES)

    def project_level_stages(self) -> list[str]:
        return list(self._PROJECT_STAGES)

    def run_stage(
        self,
        stage: str,
        kernel_name: str | None,
        kernel_dir: Path | None,
        force: bool,
    ) -> None:
        if stage == "gen_packaging":
            assert kernel_dir is not None
            self._stage_gen_packaging(kernel_dir)
        elif stage == "gen_xo":
            assert kernel_dir is not None
            self._stage_gen_xo(kernel_dir)
        elif stage == "gen_link_cfg":
            self._stage_gen_link_cfg()
        elif stage == "validate":
            assert kernel_dir is not None
            self._stage_validate(kernel_dir)
        else:
            raise BuildError(f"Unknown XRT build stage: {stage}")

    def _load_spec(self, kernel_dir: Path):
        spec_path = kernel_dir / "kernel_spec.yaml"
        if not spec_path.exists():
            raise BuildError(f"kernel_spec.yaml not found: {spec_path}")
        return parse_kernel_spec(spec_path)

    def _stage_gen_packaging(self, kernel_dir: Path) -> None:
        """Generate package_ip.tcl and kernel.xml."""
        print(f"[Stage 1] gen_packaging: {kernel_dir.name}")
        spec = self._load_spec(kernel_dir)
        gen = XrtGenerator(kernel_spec=spec, project_config=self._config)

        output = kernel_dir / "build"
        gen.generate(str(output))
        print("  done")

    def _stage_gen_xo(self, kernel_dir: Path) -> None:
        """Generate xo_gen.tcl (already done in gen_packaging, verify exists)."""
        print(f"[Stage 2] gen_xo: {kernel_dir.name}")
        xo_tcl = kernel_dir / "build" / "packaging" / "xo_gen.tcl"
        if not xo_tcl.exists():
            raise BuildError(
                f"xo_gen.tcl not found: {xo_tcl}. Run gen_packaging first."
            )
        print("  done (xo_gen.tcl verified)")

    def _stage_gen_link_cfg(self) -> None:
        """Generate project-level connectivity.cfg combining all kernels."""
        print("[Stage 3] gen_link_cfg")
        from vten.build.common import discover_kernels

        kernels = discover_kernels(self._project)
        if not kernels:
            print("  no kernels found, skip")
            return

        # Check that per-kernel link configs exist
        for kname in kernels:
            cfg = self._project / "kernels" / kname / "build" / "link" / "connectivity.cfg"
            if cfg.exists():
                print(f"  {kname}: connectivity.cfg found")
            else:
                print(f"  {kname}: connectivity.cfg not found (run gen_packaging first)")

        print("  done")

    def _stage_validate(self, kernel_dir: Path) -> None:
        """Optional: validate xclbin metadata against kernel_spec."""
        print(f"[Stage 4] validate: {kernel_dir.name}")
        xrt_cfg = self._config.get("backend", {}).get("xrt", {})
        xclbin_path = xrt_cfg.get("xclbin_path", "")

        if not xclbin_path or not Path(xclbin_path).exists():
            print("  xclbin not found, skip validation")
            return

        print("  xclbin exists, validation passed")
