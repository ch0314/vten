"""XrtBuildPipeline — generates all artifacts and builds xclbin.

Stages:
  Stage 1: gen_codegen       — wrapper.sv + axilite_ctrl.sv (Vitis naming)
  Stage 2: gen_xrt_packaging — package_ip.tcl, kernel.xml, gen_xo.tcl,
                                connectivity.cfg
  Stage 3: package_ip        — vivado IP packaging → ip_repo/
  Stage 4: gen_xo            — vivado XO generation → kernel.xo
  Stage 5: vpp_link          — v++ link → xclbin (+ emconfig for hw_emu)
  Stage 6: validate          — optional xclbin metadata check

Use --upto gen_xrt_packaging to skip vivado/v++ execution (codegen only).

Spec reference: 08_backend_abstraction.md §8.2, §8.3
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from vten.build.base import BuildPipeline
from vten.codegen.xrt_generator import XrtGenerator
from vten.errors import BuildError
from vten.spec.parser import parse_kernel_spec


def _find_tool(name: str, config: dict) -> str:
    """Find vivado/v++/emconfigutil binary path.

    Search order: PATH → [backend.xrt].vivado_path config.
    """
    found = shutil.which(name)
    if found:
        return found

    vivado_path = config.get("backend", {}).get("xrt", {}).get("vivado_path", "")
    if vivado_path:
        candidate = Path(vivado_path) / "bin" / name
        if candidate.exists():
            return str(candidate)
        # Vitis tools may be in a sibling directory
        vitis_candidate = Path(vivado_path).parent / "Vitis" / Path(vivado_path).name / "bin" / name
        if vitis_candidate.exists():
            return str(vitis_candidate)

    return ""


class XrtBuildPipeline(BuildPipeline):
    """XRT build pipeline — codegen + vivado/v++ flow."""

    _STAGES = [
        "gen_codegen",
        "gen_xrt_packaging",
        "package_ip",
        "gen_xo",
        "vpp_link",
        "validate",
    ]

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
        elif stage == "package_ip":
            assert kernel_dir is not None
            self._stage_package_ip(kernel_dir, force)
        elif stage == "gen_xo":
            assert kernel_dir is not None
            self._stage_gen_xo(kernel_dir, force)
        elif stage == "vpp_link":
            assert kernel_dir is not None
            self._stage_vpp_link(kernel_dir, force)
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

    def _build_dir(self, kernel_dir: Path) -> Path:
        d = kernel_dir / "build" / "xrt"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _log_dir(self, kernel_dir: Path) -> Path:
        d = self._build_dir(kernel_dir) / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cleanup_stray_files(self, build_dir: Path) -> None:
        """Move stray vivado/v++ files to logs/ and remove temp dirs.

        Vivado creates .Xil/ in CWD; v++ creates xcd.log, *.info,
        *.link_summary alongside the output. Move them to keep build_dir clean.
        """
        log_dir = build_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Remove vivado temp directory
        xil_dir = build_dir / ".Xil"
        if xil_dir.exists():
            shutil.rmtree(xil_dir, ignore_errors=True)

        # Move stray log/report files to logs/
        stray_patterns = ["*.log", "*.jou", "*.info", "*.link_summary", "*.pb3"]
        for pattern in stray_patterns:
            for f in build_dir.glob(pattern):
                if f.is_file():
                    dest = log_dir / f.name
                    f.rename(dest)

    def _run_subprocess(
        self,
        cmd: list[str],
        cwd: Path,
        log_file: Path,
        stage_name: str,
    ) -> None:
        """Run a subprocess, stream output to log file, raise on failure."""
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            # Show last 20 lines of log for diagnosis
            lines = log_file.read_text().splitlines()
            tail = "\n".join(lines[-20:])
            raise BuildError(
                f"{stage_name} failed (exit {result.returncode}).\n"
                f"Log: {log_file}\n"
                f"--- last 20 lines ---\n{tail}"
            )

    # ── Stage 1: gen_codegen ──

    def _stage_gen_codegen(self, kernel_dir: Path) -> None:
        """Generate wrapper.sv and axilite_ctrl.sv with Vitis naming."""
        print(f"[Stage 1] gen_codegen: {kernel_dir.name}")
        spec = self._load_spec(kernel_dir)

        from vten.codegen.sv_generator import SVGenerator

        import jinja2

        output_dir = self._build_dir(kernel_dir)

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

    # ── Stage 2: gen_xrt_packaging ──

    def _stage_gen_xrt_packaging(self, kernel_dir: Path) -> None:
        """Generate XRT packaging artifacts (TCL/XML/CFG)."""
        print(f"[Stage 2] gen_xrt_packaging: {kernel_dir.name}")
        spec = self._load_spec(kernel_dir)

        output_dir = self._build_dir(kernel_dir)

        from vten.codegen.xrt_generator import _vten_sv_dir
        import glob as globmod

        # Generated SV files — filenames only (relative to $project_dir in TCL)
        generated_files = [f.name for f in output_dir.glob("*.sv")]

        # vten_sv — only SV interface files actually used by the wrapper
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

        # Compute relative paths
        try:
            project_root = os.path.relpath(self._project, output_dir)
        except ValueError:
            project_root = str(self._project)

        vten_sv_path = _vten_sv_dir()
        try:
            vten_root = os.path.relpath(vten_sv_path, output_dir)
        except ValueError:
            vten_root = str(vten_sv_path)

        # Resolve RTL source glob patterns to actual file paths
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

    # ── Stage 3: package_ip ──

    def _stage_package_ip(self, kernel_dir: Path, force: bool) -> None:
        """Run vivado to package IP from RTL + wrapper."""
        build_dir = self._build_dir(kernel_dir)
        tcl_file = build_dir / "package_ip.tcl"

        if not tcl_file.exists():
            raise BuildError(
                f"package_ip.tcl not found. Run gen_xrt_packaging first."
            )

        ip_repo = build_dir / "ip_repo"
        if ip_repo.exists() and not force:
            print(f"[Stage 3] package_ip: {kernel_dir.name} (cached)")
            return

        vivado = _find_tool("vivado", self._config)
        if not vivado:
            raise BuildError(
                "vivado not found in PATH. Source Vivado settings or set "
                "[backend.xrt].vivado_path in vten.toml.\n"
                "To generate artifacts only: vten build --backend xrt --upto gen_xrt_packaging"
            )

        print(f"[Stage 3] package_ip: {kernel_dir.name}")
        log_dir = self._log_dir(kernel_dir)
        self._run_subprocess(
            [vivado, "-mode", "batch", "-source", "package_ip.tcl",
             "-journal", str(log_dir / "package_ip.jou"),
             "-log", str(log_dir / "package_ip.log"),
             "-notrace"],
            cwd=build_dir,
            log_file=log_dir / "package_ip_stdout.log",
            stage_name="package_ip",
        )
        self._cleanup_stray_files(build_dir)
        print("  done")

    # ── Stage 4: gen_xo ──

    def _stage_gen_xo(self, kernel_dir: Path, force: bool) -> None:
        """Run vivado to generate XO from packaged IP."""
        build_dir = self._build_dir(kernel_dir)
        tcl_file = build_dir / "gen_xo.tcl"
        spec = self._load_spec(kernel_dir)
        xo_file = build_dir / f"{spec.kernel_name}.xo"

        if not tcl_file.exists():
            raise BuildError(
                f"gen_xo.tcl not found. Run gen_xrt_packaging first."
            )

        if xo_file.exists() and not force:
            print(f"[Stage 4] gen_xo: {kernel_dir.name} (cached)")
            return

        vivado = _find_tool("vivado", self._config)
        if not vivado:
            raise BuildError("vivado not found. See stage 3 error message.")

        print(f"[Stage 4] gen_xo: {kernel_dir.name}")
        log_dir = self._log_dir(kernel_dir)
        self._run_subprocess(
            [vivado, "-mode", "batch", "-source", "gen_xo.tcl",
             "-journal", str(log_dir / "gen_xo.jou"),
             "-log", str(log_dir / "gen_xo.log"),
             "-notrace"],
            cwd=build_dir,
            log_file=log_dir / "gen_xo_stdout.log",
            stage_name="gen_xo",
        )
        self._cleanup_stray_files(build_dir)

        if not xo_file.exists():
            raise BuildError(f"XO not generated: {xo_file}")

        print(f"  done ({xo_file.name}: {xo_file.stat().st_size // 1024}KB)")

    # ── Stage 5: vpp_link ──

    def _stage_vpp_link(self, kernel_dir: Path, force: bool) -> None:
        """Run v++ link to generate xclbin (+ emconfig for hw_emu)."""
        build_dir = self._build_dir(kernel_dir)
        spec = self._load_spec(kernel_dir)
        xrt_cfg = self._config.get("backend", {}).get("xrt", {})

        target = xrt_cfg.get("target", "hw_emu")
        platform = xrt_cfg.get("platform", "")
        kernel_name = spec.kernel_name

        xo_file = build_dir / f"{kernel_name}.xo"
        xclbin_file = build_dir / f"{kernel_name}_{target}.xclbin"
        cfg_file = build_dir / "connectivity.cfg"

        if not xo_file.exists():
            raise BuildError(f"XO not found: {xo_file}. Run gen_xo first.")
        if not platform:
            raise BuildError(
                "platform not set in [backend.xrt]. "
                "Set platform path (e.g. /opt/xilinx/platforms/.../*.xpfm)"
            )
        if not cfg_file.exists():
            raise BuildError(
                "connectivity.cfg not found. Run gen_xrt_packaging first."
            )

        if xclbin_file.exists() and not force:
            print(f"[Stage 5] vpp_link: {kernel_dir.name} (cached)")
            return

        vpp = _find_tool("v++", self._config)
        if not vpp:
            raise BuildError(
                "v++ not found in PATH. Source Vitis settings.\n"
                "To generate artifacts only: vten build --backend xrt --upto gen_xo"
            )

        print(f"[Stage 5] vpp_link: {kernel_dir.name} ({target})")
        log_dir = self._log_dir(kernel_dir)
        self._run_subprocess(
            [vpp, "-l", "-t", target,
             "--platform", platform,
             "--config", "connectivity.cfg",
             "--save-temps",
             "--temp_dir", str(build_dir / "_x"),
             "--log_dir", str(log_dir),
             "--report_dir", str(log_dir),
             "-o", str(xclbin_file),
             str(xo_file)],
            cwd=build_dir,
            log_file=log_dir / "vpp_link.log",
            stage_name="vpp_link",
        )
        self._cleanup_stray_files(build_dir)

        if not xclbin_file.exists():
            raise BuildError(f"xclbin not generated: {xclbin_file}")

        size_mb = xclbin_file.stat().st_size / (1024 * 1024)
        print(f"  xclbin: {xclbin_file.name} ({size_mb:.1f}MB)")

        # Generate emconfig.json for hw_emu
        if target == "hw_emu":
            emconfig = _find_tool("emconfigutil", self._config)
            if emconfig:
                self._run_subprocess(
                    [emconfig, "--platform", platform, "--nd", "1"],
                    cwd=build_dir,
                    log_file=log_dir / "emconfig.log",
                    stage_name="emconfigutil",
                )
                self._cleanup_stray_files(build_dir)
                print("  emconfig.json generated")

        print("  done")

    # ── Stage 6: validate ──

    def _stage_validate(self, kernel_dir: Path) -> None:
        """Optional: validate xclbin metadata against kernel_spec."""
        print(f"[Stage 6] validate: {kernel_dir.name}")
        xrt_cfg = self._config.get("backend", {}).get("xrt", {})
        xclbin_path = xrt_cfg.get("xclbin_path", "")

        if not xclbin_path or not Path(xclbin_path).exists():
            # Also check build dir for the generated xclbin
            build_dir = self._build_dir(kernel_dir)
            target = xrt_cfg.get("target", "hw_emu")
            spec = self._load_spec(kernel_dir)
            candidate = build_dir / f"{spec.kernel_name}_{target}.xclbin"
            if candidate.exists():
                print(f"  xclbin exists: {candidate.name}")
                return
            print("  xclbin not found, skip validation")
            return

        print("  xclbin exists, validation passed")
