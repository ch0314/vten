"""VerilatorBuildPipeline — Verilator 4-stage build pipeline.

Spec reference: 08_backend_abstraction.md §8.2

Pipeline stages:
  Stage 1: dpi_c      — gcc shared library (cached, project-level)
  Stage 2: codegen    — Jinja2 → generated SV (per-kernel)
  Stage 3: verilate   — verilator --cc --exe (per-kernel)
  Stage 4: make       — make -C obj_dir (per-kernel)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from vten.build.base import BuildPipeline

logger = logging.getLogger(__name__)
from vten.build.common import (
    cache_valid,
    dir_hash,
    load_cache,
    save_cache,
    update_cache,
)
from vten.codegen.sv_generator import SVGenerator
from vten.errors import BuildError
from vten.spec.parser import parse_kernel_spec


class VerilatorBuildPipeline(BuildPipeline):
    """Verilator 4-stage build pipeline."""

    _STAGES = ["dpi_c", "codegen", "verilate", "make"]
    _PROJECT_STAGES = {"dpi_c"}

    def __init__(self, project: Path, config: dict) -> None:
        super().__init__(project, config)
        veri_cfg = config.get("backend", {}).get("verilator", {})
        self._verilator_bin = veri_cfg.get("verilator_path", "verilator")
        self._threads = veri_cfg.get("threads", 4)
        self._trace = veri_cfg.get("trace", False)
        self._opt_level = veri_cfg.get("opt_level", 3)
        self._extra_args = veri_cfg.get("extra_args", [])
        self._vten_root = Path(__file__).resolve().parent.parent.parent
        self._vten_sv_dir = self._vten_root / "vten_sv"
        self._cache = load_cache(project / "build" / ".cache.json")

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
        if stage == "dpi_c":
            self._stage_dpi_c(force)
        elif stage == "codegen":
            assert kernel_dir is not None
            self._stage_codegen(kernel_dir)
        elif stage == "verilate":
            assert kernel_dir is not None
            self._stage_verilate(kernel_dir)
        elif stage == "make":
            assert kernel_dir is not None
            self._stage_make(kernel_dir)
        else:
            raise BuildError(f"Unknown Verilator build stage: {stage}")

    def _clean_project(self) -> None:
        """Override: also reset in-memory cache."""
        super()._clean_project()
        self._cache.clear()

    def build(self, **kwargs) -> None:
        """Override to save cache after build."""
        try:
            super().build(**kwargs)
        finally:
            save_cache(self._project / "build" / ".cache.json", self._cache)

    # ── Stage implementations ──

    def _stage_dpi_c(self, force: bool) -> None:
        """Compile DPI-C shared library (same as xsim, no Vivado includes)."""
        logger.info("[Stage 1] dpi_c")
        src_c = self._vten_sv_dir / "vten_shm_bridge.c"
        src_h = self._vten_sv_dir / "vten_shm_bridge.h"
        current = dir_hash([p for p in [src_c, src_h] if p.exists()])

        if not force and cache_valid(self._cache, "dpi_c", current):
            logger.info("  cached, skip")
            return

        so_path = self._project / "build" / "lib" / "libvten_shm.so"
        so_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "gcc", "-shared", "-fPIC",
                "-I", str(self._vten_sv_dir),
                "-o", str(so_path),
                str(src_c),
                "-lrt", "-lpthread",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise BuildError(f"gcc failed:\n{result.stderr}")

        update_cache(self._cache, "dpi_c", current)
        logger.info("  done")

    def _stage_codegen(self, kernel_dir: Path) -> None:
        """Generate testbench SV from kernel_spec.yaml (same as xsim)."""
        logger.info("[Stage 2] codegen: %s", kernel_dir.name)
        from vten.build.xsim_build import _derive_bfm_configs, _expand_split_interfaces

        spec_path = kernel_dir / "kernel_spec.yaml"
        if not spec_path.exists():
            raise BuildError(f"kernel_spec.yaml not found: {spec_path}")
        spec = parse_kernel_spec(spec_path)
        spec = _expand_split_interfaces(spec)
        bfm_configs = _derive_bfm_configs(spec)

        gen = SVGenerator(
            kernel_spec=spec,
            bfm_configs=bfm_configs,
            project_config=self._config,
        )

        output = kernel_dir / "build" / "generated"
        # Clean stale generated files to avoid module conflicts
        if output.exists():
            for f in output.glob("*.sv"):
                f.unlink()
        output.mkdir(parents=True, exist_ok=True)
        gen.generate(str(output), num_commands=256)
        logger.info("  done")

    def _stage_verilate(self, kernel_dir: Path) -> None:
        """Run verilator --cc --exe to generate C++ model."""
        logger.info("[Stage 3] verilate: %s", kernel_dir.name)
        tb_top = kernel_dir / "build" / "generated" / "tb_top.sv"
        if not tb_top.exists():
            raise BuildError(
                f"tb_top.sv not found: {tb_top}. Run codegen first."
            )

        build_dir = kernel_dir / "build"
        obj_dir = build_dir / "obj_dir"
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Collect all SV source files
        sv_files = sorted(self._vten_sv_dir.glob("*.sv"))

        # RTL sources from project config
        rtl_patterns = self._config.get("rtl", {}).get("sources", [])
        rtl_files: list[Path] = []
        for pat in rtl_patterns:
            rtl_files.extend(sorted(self._project.glob(pat)))

        # DPI-C wrapper
        dpi_cpp = self._vten_sv_dir / "vten_shm_bridge_verilator.cpp"

        cmd = [
            self._verilator_bin,
            "--cc", "--exe", "--main", "--timing",
            "--top-module", "tb_top",
            f"-O{self._opt_level}",
            f"-j", str(self._threads),
            "-I" + str(self._vten_sv_dir),
            "--Mdir", str(obj_dir),
            # Suppress common SV warnings that are safe in vTen BFMs
            "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-WIDTHCONCAT",
            "-Wno-CASEINCOMPLETE", "-Wno-IGNOREDRETURN", "-Wno-MULTIDRIVEN",
            "-Wno-TIMESCALEMOD",
        ]

        if self._trace:
            cmd.append("--trace")

        cmd.extend(self._extra_args)

        # Add source files: all generated SV (tb_top, wrapper, controller)
        gen_dir = kernel_dir / "build" / "generated"
        for f in sorted(gen_dir.glob("*.sv")):
            cmd.append(str(f))
        for f in sv_files:
            cmd.append(str(f))
        for f in rtl_files:
            cmd.append(str(f))

        # DPI-C sources
        if dpi_cpp.exists():
            cmd.append(str(dpi_cpp))
        cmd.append(str(self._vten_sv_dir / "vten_shm_bridge.c"))

        # Compile/link flags: C++20 coroutines for --timing, POSIX SHM
        cmd.extend(["-CFLAGS", f"-I{self._vten_sv_dir} -fcoroutines"])
        cmd.extend(["-LDFLAGS", "-lrt -lpthread"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(build_dir),
        )
        if result.returncode != 0:
            detail = result.stderr[-1000:] if result.stderr else result.stdout[-1000:]
            raise BuildError(f"verilator failed:\n{detail}")

        logger.info("  done")

    def _stage_make(self, kernel_dir: Path) -> None:
        """Build Verilator binary from generated C++ sources."""
        logger.info("[Stage 4] make: %s", kernel_dir.name)
        obj_dir = kernel_dir / "build" / "obj_dir"
        makefile = obj_dir / "Vtb_top.mk"
        if not makefile.exists():
            raise BuildError(
                f"Vtb_top.mk not found: {makefile}. Run verilate first."
            )

        result = subprocess.run(
            [
                "make", "-C", str(obj_dir), "-f", "Vtb_top.mk",
                f"-j{self._threads}",
                "VM_USER_CFLAGS=-fcoroutines",  # C++20 coroutines for --timing
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise BuildError(f"make failed:\n{result.stderr[-500:]}")

        logger.info("  done")
