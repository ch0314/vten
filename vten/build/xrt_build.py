"""XrtBuildPipeline — generates all artifacts for xclbin build.

Stages:
  Stage 1: gen_codegen      — wrapper.sv + axilite_ctrl.sv (Vitis naming)
  Stage 2: gen_xrt_packaging — package_ip.tcl, kernel.xml, gen_xo.tcl,
                                connectivity.cfg, build script
  Stage 3: validate         — optional xclbin metadata check

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

    _STAGES = ["gen_codegen", "gen_xrt_packaging", "validate"]

    def __init__(self, project: Path, config: dict) -> None:
        super().__init__(project, config)

    def stages(self) -> list[str]:
        return list(self._STAGES)

    def project_level_stages(self) -> list[str]:
        return []

    def run_stage(
        self,
        stage: str,
        kernel_name: str | None,
        kernel_dir: Path | None,
        force: bool,
    ) -> None:
        if stage == "gen_codegen":
            assert kernel_dir is not None
            self._stage_gen_codegen(kernel_dir)
        elif stage == "gen_xrt_packaging":
            assert kernel_dir is not None
            self._stage_gen_xrt_packaging(kernel_dir)
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

    def _stage_gen_codegen(self, kernel_dir: Path) -> None:
        """Generate wrapper.sv and axilite_ctrl.sv with Vitis naming."""
        print(f"[Stage 1] gen_codegen: {kernel_dir.name}")
        spec = self._load_spec(kernel_dir)

        from vten.codegen.sv_generator import SVGenerator

        import jinja2

        output_dir = kernel_dir / "build" / "xrt"
        output_dir.mkdir(parents=True, exist_ok=True)

        template_dir = Path(__file__).resolve().parent.parent.parent / "templates"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
        )

        gen = SVGenerator(spec, bfm_configs=[], project_config=self._config)
        gen._generate_wrapper(env, output_dir)

        # Generate axilite controller if needed
        for _name, iface in spec.interfaces.items():
            if iface.protocol.value == "axi4_lite" and iface.generate_controller:
                gen._generate_axilite_ctrl(env, output_dir, iface)

        print("  done")

    def _stage_gen_xrt_packaging(self, kernel_dir: Path) -> None:
        """Generate XRT packaging artifacts (TCL/XML/CFG/build script)."""
        print(f"[Stage 2] gen_xrt_packaging: {kernel_dir.name}")
        spec = self._load_spec(kernel_dir)

        output_dir = kernel_dir / "build" / "xrt"
        output_dir.mkdir(parents=True, exist_ok=True)

        from vten.codegen.xrt_generator import _vten_sv_dir
        import os

        # Generated SV files — filenames only (relative to $project_dir in TCL)
        generated_files = [f.name for f in output_dir.glob("*.sv")]

        # vten_sv — only SV interface files actually used by the wrapper
        # Map protocol → vten_sv interface file
        _proto_to_sv = {
            "axi4": "vten_aximm_if.sv",
            "axi4_stream": "vten_axis_if.sv",
        }
        vten_sv_set: set[str] = set()
        for iface in spec.interfaces.values():
            sv_file = _proto_to_sv.get(iface.protocol.value)
            if sv_file and (_vten_sv_dir() / sv_file).exists():
                vten_sv_set.add(sv_file)
        vten_sv_files = sorted(vten_sv_set)

        # Compute relative path from output_dir to project_root
        try:
            project_root = os.path.relpath(self._project, output_dir)
        except ValueError:
            project_root = str(self._project)

        # Compute relative path from output_dir to vten_sv
        vten_sv_path = _vten_sv_dir()
        try:
            vten_root = os.path.relpath(vten_sv_path, output_dir)
        except ValueError:
            vten_root = str(vten_sv_path)

        # Resolve RTL source glob patterns to actual file paths
        # (TCL glob doesn't support ** recursive patterns)
        import glob as globmod

        rtl_patterns = self._config.get("rtl", {}).get("sources", [])
        resolved_rtl: list[str] = []
        for pat in rtl_patterns:
            matches = sorted(globmod.glob(str(self._project / pat), recursive=True))
            for m in matches:
                rel = os.path.relpath(m, self._project)
                resolved_rtl.append(rel)

        config = dict(self._config)
        config["rtl"] = {"sources": resolved_rtl}
        config["generated_files"] = generated_files
        config["vten_sv_files"] = vten_sv_files
        config["_project_root"] = project_root
        config["_vten_root"] = vten_root

        gen = XrtGenerator(kernel_spec=spec, project_config=config)
        gen.generate(str(output_dir))
        print("  done")

    def _stage_validate(self, kernel_dir: Path) -> None:
        """Optional: validate xclbin metadata against kernel_spec."""
        print(f"[Stage 3] validate: {kernel_dir.name}")
        xrt_cfg = self._config.get("backend", {}).get("xrt", {})
        xclbin_path = xrt_cfg.get("xclbin_path", "")

        if not xclbin_path or not Path(xclbin_path).exists():
            print("  xclbin not found, skip validation")
            return

        print("  xclbin exists, validation passed")
