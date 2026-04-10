"""XrtBuildPipeline — generates all artifacts and builds xclbin.

Single-kernel mode:
  vten build --backend xrt --kernel weight_loader
  Stage 1: gen_codegen       — wrapper.sv + axilite_ctrl.sv (Vitis naming)
  Stage 2: gen_xrt_packaging — package_ip.tcl, kernel.xml, gen_xo.tcl,
                                connectivity.cfg
  Stage 3: package_ip        — vivado IP packaging → ip_repo/
  Stage 4: gen_xo            — vivado XO generation → kernel.xo
  Stage 5: vpp_link          — v++ link → xclbin (+ emconfig for hw_emu)
  Stage 6: validate          — optional xclbin metadata check

Multi-kernel mode (CompositeKernel auto-detected):
  vten build --backend xrt --kernel npu_pipeline
  Stages 1-4: auto-run for each sub-kernel
  Stage 5: gen_link_config   — extract connections, unified connectivity.cfg
  Stage 6: vpp_link          — v++ link all sub-kernel .xo → xclbin
  Stage 7: validate          — xclbin metadata check

Use --upto gen_xrt_packaging to skip vivado/v++ execution (codegen only).

Spec reference: 08_backend_abstraction.md §8.2, §8.3
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from vten.build.base import BuildPipeline
from vten.codegen.xrt_generator import XrtGenerator
from vten.errors import BuildError
from vten.spec.parser import parse_kernel_spec

logger = logging.getLogger(__name__)


def _find_tool(name: str, config: dict) -> str:
    """Find vivado/v++/emconfigutil binary path.

    Search order: PATH → [tools]/[backend.xrt].vivado_path config.
    """
    found = shutil.which(name)
    if found:
        return found

    from vten.cli.config import resolve_tool_path
    vivado_path = resolve_tool_path(config, "vivado_path", "xrt")
    if vivado_path:
        candidate = Path(vivado_path) / "bin" / name
        if candidate.exists():
            return str(candidate)
        vitis_candidate = Path(vivado_path).parent / "Vitis" / Path(vivado_path).name / "bin" / name
        if vitis_candidate.exists():
            return str(vitis_candidate)

    return ""


class XrtBuildPipeline(BuildPipeline):
    """XRT build pipeline — codegen + vivado/v++ flow.

    Automatically detects CompositeKernel: when --kernel targets a composite,
    all sub-kernels are built first, then linked together via v++.
    """

    _STAGES = [
        "gen_codegen",
        "gen_xrt_packaging",
        "package_ip",
        "gen_xo",
        "gen_link_config",
        "vpp_link",
        "validate",
    ]

    def __init__(self, project: Path, config: dict) -> None:
        super().__init__(project, config)
        self._config["ip"] = self._normalize_ip_config(self._config.get("ip"))

    def stages(self) -> list[str]:
        return list(self._STAGES)

    def project_level_stages(self) -> list[str]:
        return []

    def build(
        self,
        kernel_name: str | None = None,
        stage: str | None = None,
        upto: str | None = None,
        force: bool = False,
        clean: bool = False,
        skip_compile: bool = False,
        config_overrides: dict | None = None,
    ) -> None:
        """Build entry point — auto-detects composite vs unit kernel."""
        from vten.build.common import resolve_stages

        if config_overrides:
            self._config.setdefault("parameters", {}).update(config_overrides)

        all_stages = self.stages()
        target_stages = resolve_stages(all_stages, stage, upto, skip_compile)

        if not kernel_name:
            raise BuildError(
                "XRT build requires --kernel <name>.\n"
                "  Unit kernel:      vten build --backend xrt --kernel weight_loader\n"
                "  CompositeKernel:  vten build --backend xrt --kernel npu_pipeline"
            )

        kernel_dir = self._project / "kernels" / kernel_name
        if not kernel_dir.exists():
            raise BuildError(f"Kernel directory not found: {kernel_dir}")

        # Detect: composite or unit kernel?
        from vten.build.composite import is_composite_kernel

        if clean:
            self._clean_kernel(kernel_dir)
            # For composite kernels, also clean sub-kernel build dirs
            if is_composite_kernel(kernel_dir):
                from vten.build.composite import load_composite_class
                comp_cls = load_composite_class(kernel_dir)
                for _sn, sub_cls in getattr(comp_cls, "_sub_kernel_refs", {}).items():
                    sub_dir = self._find_sub_kernel_dir(sub_cls)
                    if sub_dir:
                        self._clean_kernel(sub_dir)

        if is_composite_kernel(kernel_dir):
            self._build_composite(kernel_name, kernel_dir, target_stages, force)
        else:
            self._build_unit(kernel_name, kernel_dir, target_stages, force)

        logger.info("Build complete.")

    def _build_unit(
        self,
        kernel_name: str,
        kernel_dir: Path,
        target_stages: list[str],
        force: bool,
    ) -> None:
        """Build a single unit kernel through all targeted stages."""
        # Unit kernels skip composite-only stages
        unit_stages = ["gen_codegen", "gen_xrt_packaging", "package_ip",
                        "gen_xo", "vpp_link", "validate"]

        logger.info("=== Unit Kernel: %s ===", kernel_name)
        for s in target_stages:
            if s in unit_stages:
                self._run_unit_stage(s, kernel_dir, force)

    def _build_composite(
        self,
        composite_name: str,
        composite_dir: Path,
        target_stages: list[str],
        force: bool,
    ) -> None:
        """Build a CompositeKernel: sub-kernels first, then link."""
        from vten.build.composite import load_composite_class

        composite_cls = load_composite_class(composite_dir)
        sub_refs = getattr(composite_cls, "_sub_kernel_refs", {})

        # Resolve sub-kernel directories
        sub_kernel_dirs: dict[str, Path] = {}
        for sub_name, sub_cls in sub_refs.items():
            sub_dir = self._find_sub_kernel_dir(sub_cls)
            if sub_dir:
                sub_kernel_dirs[sub_name] = sub_dir
            else:
                raise BuildError(
                    f"Cannot find kernel directory for sub-kernel "
                    f"'{sub_name}' ({sub_cls.__name__})"
                )

        per_kernel_stages = ["gen_codegen", "gen_xrt_packaging",
                              "package_ip", "gen_xo"]
        link_stages = ["gen_link_config", "vpp_link", "validate"]

        # Phase 1: Build each sub-kernel
        active_per_kernel = [s for s in target_stages if s in per_kernel_stages]
        if active_per_kernel:
            for sub_name, sub_dir in sorted(sub_kernel_dirs.items()):
                logger.info("=== Sub-kernel: %s (%s) ===", sub_dir.name, sub_name)
                for s in active_per_kernel:
                    self._run_unit_stage(s, sub_dir, force)

        # Phase 2: Link all sub-kernels
        active_link = [s for s in target_stages if s in link_stages]
        if active_link:
            logger.info("=== Composite: %s ===", composite_name)
            for s in active_link:
                if s == "gen_link_config":
                    self._stage_gen_link_config(
                        composite_name, composite_cls, sub_kernel_dirs, force,
                    )
                elif s == "vpp_link":
                    self._stage_vpp_link_composite(
                        composite_name, sub_kernel_dirs, force,
                    )
                elif s == "validate":
                    self._stage_validate_composite(composite_name)

    def _find_sub_kernel_dir(self, sub_cls: type) -> Path | None:
        """Find kernel directory for a sub-kernel class."""
        import sys
        mod = sys.modules.get(sub_cls.__module__)
        src_file = getattr(mod, "__file__", None) if mod else None
        if src_file:
            kernel_dir = Path(src_file).resolve().parent
            if (kernel_dir / "kernel_spec.yaml").exists():
                return kernel_dir

        # Fallback: class name → snake_case directory
        import re
        name = re.sub(r"Kernel$", "", sub_cls.__name__)
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        candidate = self._project / "kernels" / name
        if candidate.exists() and (candidate / "kernel_spec.yaml").exists():
            return candidate

        return None

    def _run_unit_stage(self, stage: str, kernel_dir: Path, force: bool) -> None:
        """Run a single stage for a unit kernel."""
        if stage == "gen_codegen":
            self._stage_gen_codegen(kernel_dir)
        elif stage == "gen_xrt_packaging":
            self._stage_gen_xrt_packaging(kernel_dir)
        elif stage == "package_ip":
            self._stage_package_ip(kernel_dir, force)
        elif stage == "gen_xo":
            self._stage_gen_xo(kernel_dir, force)
        elif stage == "vpp_link":
            self._stage_vpp_link_single(kernel_dir, force)
        elif stage == "validate":
            self._stage_validate_single(kernel_dir)

    # ── Shared helpers ──

    def run_stage(self, stage, kernel_name, kernel_dir, force):
        # Not used in new flow — build() dispatches directly.
        # Kept for BuildPipeline ABC compatibility.
        if stage not in self._STAGES:
            raise BuildError(f"Unknown XRT build stage: '{stage}'")
        if kernel_dir is not None:
            self._run_unit_stage(stage, kernel_dir, force)

    def _load_spec(self, kernel_dir: Path):
        spec_path = kernel_dir / "kernel_spec.yaml"
        if not spec_path.exists():
            raise BuildError(f"kernel_spec.yaml not found: {spec_path}")
        return parse_kernel_spec(spec_path)

    def _build_dir(self, kernel_dir: Path) -> Path:
        d = kernel_dir / "build" / "xrt"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _composite_build_dir(self, composite_name: str) -> Path:
        d = self._project / "kernels" / composite_name / "build" / "xrt"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _log_dir(self, kernel_dir: Path) -> Path:
        d = self._build_dir(kernel_dir) / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cleanup_stray_files(self, build_dir: Path) -> None:
        """Move stray vivado/v++ files to logs/ and remove temp dirs."""
        log_dir = build_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        xil_dir = build_dir / ".Xil"
        if xil_dir.exists():
            shutil.rmtree(xil_dir, ignore_errors=True)

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
        """Run a subprocess, stream output to log file (and stdout in DEBUG)."""
        verbose = logger.isEnabledFor(logging.DEBUG)
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                if verbose:
                    sys.stdout.write(line)
                f.write(line)
            proc.wait()
        if proc.returncode != 0:
            lines = log_file.read_text().splitlines()
            tail = "\n".join(lines[-20:])
            raise BuildError(
                f"{stage_name} failed (exit {proc.returncode}).\n"
                f"Log: {log_file}\n"
                f"--- last 20 lines ---\n{tail}"
            )

    # ── Per-kernel stages ──

    def _stage_gen_codegen(self, kernel_dir: Path) -> None:
        """Generate wrapper.sv and axilite_ctrl.sv with Vitis naming."""
        logger.info("  [gen_codegen] %s", kernel_dir.name)
        spec = self._load_spec(kernel_dir)

        from vten.codegen.sv_generator import SVGenerator
        import jinja2

        output_dir = self._build_dir(kernel_dir)
        template_dir = Path(__file__).resolve().parent.parent / "templates" / "sim"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
        )

        gen = SVGenerator(spec, bfm_configs=[], project_config=self._config)
        gen._generate_wrapper(env, output_dir)

        for iface in spec.interfaces.values():
            if iface.protocol.value == "axi4_lite" and iface.generate_controller:
                gen._generate_axilite_ctrl(env, output_dir, iface)

    def _stage_gen_xrt_packaging(self, kernel_dir: Path) -> None:
        """Generate XRT packaging artifacts (TCL/XML/CFG)."""
        logger.info("  [gen_xrt_packaging] %s", kernel_dir.name)
        spec = self._load_spec(kernel_dir)
        output_dir = self._build_dir(kernel_dir)

        from vten.codegen.xrt_generator import _vten_sv_dir
        import glob as globmod

        generated_files = [f.name for f in output_dir.glob("*.sv")]

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

        try:
            project_root = os.path.relpath(self._project, output_dir)
        except ValueError:
            project_root = str(self._project)

        vten_sv_path = _vten_sv_dir()
        try:
            vten_root = os.path.relpath(vten_sv_path, output_dir)
        except ValueError:
            vten_root = str(vten_sv_path)

        rtl_patterns = self._config.get("rtl", {}).get("sources", [])
        resolved_rtl: list[str] = []
        for pat in rtl_patterns:
            matches = sorted(globmod.glob(str(self._project / pat), recursive=True))
            for m in matches:
                rel = os.path.relpath(m, self._project)
                resolved_rtl.append(rel)

        # Resolve include directories and collect header files (.svh, .vh)
        include_dirs_raw = self._config.get("rtl", {}).get("include_dirs", [])
        resolved_include_dirs: list[str] = []
        for inc_dir in include_dirs_raw:
            abs_dir = self._project / inc_dir
            if abs_dir.exists():
                resolved_include_dirs.append(os.path.relpath(abs_dir, self._project))
                # Add header files from include dirs to sources
                for ext in ("*.svh", "*.vh"):
                    for hdr in sorted(abs_dir.glob(ext)):
                        rel = os.path.relpath(hdr, self._project)
                        if rel not in resolved_rtl:
                            resolved_rtl.append(rel)

        ip_sources, ip_create = self._parse_ip_entries(
            self._config.get("ip", []), self._project,
        )

        config = dict(self._config)
        config["rtl"] = {"sources": resolved_rtl, "include_dirs": resolved_include_dirs}
        config["generated_files"] = generated_files
        config["vten_sv_files"] = vten_sv_files
        config["_project_root"] = project_root
        config["_vten_root"] = vten_root
        config["_ip_sources"] = ip_sources
        config["_ip_create"] = ip_create

        gen = XrtGenerator(kernel_spec=spec, project_config=config)
        gen.generate(str(output_dir))

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
            logger.info("  [package_ip] %s (cached)", kernel_dir.name)
            return

        vivado = _find_tool("vivado", self._config)
        if not vivado:
            raise BuildError(
                "vivado not found in PATH. Source Vivado settings or set "
                "[tools].vivado_path in vten.toml.\n"
                "To generate artifacts only: vten build --backend xrt --upto gen_xrt_packaging"
            )

        logger.info("  [package_ip] %s", kernel_dir.name)
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
            logger.info("  [gen_xo] %s (cached)", kernel_dir.name)
            return

        vivado = _find_tool("vivado", self._config)
        if not vivado:
            raise BuildError("vivado not found. See gen_codegen error message.")

        logger.info("  [gen_xo] %s", kernel_dir.name)
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

        logger.info("    %s (%dKB)", xo_file.name, xo_file.stat().st_size // 1024)

    # ── Composite-only stages ──

    def _stage_gen_link_config(
        self,
        composite_name: str,
        composite_cls: type,
        sub_kernel_dirs: dict[str, Path],
        force: bool,
    ) -> None:
        """Generate unified connectivity.cfg with sp= + stream_connect=."""
        logger.info("  [gen_link_config] %s", composite_name)

        build_dir = self._composite_build_dir(composite_name)
        connections = getattr(composite_cls, "_connections", [])
        sub_refs = getattr(composite_cls, "_sub_kernel_refs", {})

        # Map sub_name → (kernel_spec_name, KernelSpec)
        sub_specs: dict[str, tuple[str, object]] = {}
        for sub_name in sub_refs:
            sub_dir = sub_kernel_dirs[sub_name]
            spec = self._load_spec(sub_dir)
            sub_specs[sub_name] = (spec.kernel_name, spec)

        # Build sp= entries (memory bank mapping)
        from vten.codegen.xrt_generator import _build_interfaces_context
        sp_entries: list[str] = []
        for sub_name, (kname, spec) in sub_specs.items():
            ifaces = _build_interfaces_context(spec)
            for ictx in ifaces.values():
                if ictx["protocol"] == "axi4":
                    sp_entries.append(
                        f"sp={kname}_1.{ictx['ext_port']}:{ictx['memory_bank']}"
                    )

        # Build stream_connect= entries from connections
        stream_entries: list[str] = []
        for conn in connections:
            src_sub, dst_sub = conn.source_sub, conn.dest_sub
            src_tensor, dst_tensor = conn.source_name, conn.dest_name

            if src_sub not in sub_specs or dst_sub not in sub_specs:
                logger.warning(
                    "  skip %s.%s >> %s.%s: spec not found",
                    src_sub, src_tensor, dst_sub, dst_tensor,
                )
                continue

            src_kname, src_spec = sub_specs[src_sub]
            dst_kname, dst_spec = sub_specs[dst_sub]

            src_ports = self._resolve_stream_ports(src_spec, src_tensor)
            dst_ports = self._resolve_stream_ports(dst_spec, dst_tensor)

            if len(src_ports) != len(dst_ports):
                raise BuildError(
                    f"Array size mismatch: {src_sub}.{src_tensor}"
                    f"[{len(src_ports)}] >> {dst_sub}.{dst_tensor}"
                    f"[{len(dst_ports)}]"
                )

            # Validate data_width compatibility
            src_iface = self._find_stream_interface(src_spec, src_tensor)
            dst_iface = self._find_stream_interface(dst_spec, dst_tensor)
            if (src_iface and dst_iface
                    and src_iface.data_width != dst_iface.data_width):
                raise BuildError(
                    f"Stream data_width mismatch: "
                    f"{src_sub}.{src_tensor} ({src_iface.data_width}b) >> "
                    f"{dst_sub}.{dst_tensor} ({dst_iface.data_width}b)"
                )

            for sp, dp in zip(src_ports, dst_ports):
                stream_entries.append(
                    f"stream_connect={src_kname}_1.{sp}:{dst_kname}_1.{dp}"
                )

        # SLR placement from [backend.xrt.slr]
        xrt_config = self._config.get("backend", {}).get("xrt", {})
        slr_map = xrt_config.get("slr", {})
        slr_entries: list[str] = []
        for sub_name, (kname, _spec) in sub_specs.items():
            slr = slr_map.get(kname) or slr_map.get(sub_name)
            if slr:
                slr_entries.append(f"slr={kname}_1:{slr}")

        # Clock frequency from [backend.xrt].clock_freq_hz
        clock_freq = xrt_config.get("clock_freq_hz")
        clock_entries: list[str] = []
        if clock_freq:
            for _sub_name, (kname, _spec) in sub_specs.items():
                clock_entries.append(f"freqHz={clock_freq}:{kname}_1.ap_clk")

        # Write connectivity.cfg
        lines = ["[connectivity]"]
        # nk= (kernel instance naming)
        for _sub_name, (kname, _spec) in sorted(sub_specs.items()):
            lines.append(f"nk={kname}:1:{kname}_1")
        lines.append("")
        if sp_entries:
            lines.append("# Memory bank mapping")
            lines.extend(sp_entries)
        if stream_entries:
            lines.append("")
            lines.append("# Stream connections")
            lines.extend(stream_entries)
        if slr_entries:
            lines.append("")
            lines.append("# SLR placement")
            lines.extend(slr_entries)
        lines.append("")

        # Clock section
        if clock_entries:
            lines.append("[clock]")
            lines.extend(clock_entries)
            lines.append("")

        # Advanced section
        advanced = xrt_config.get("advanced", {})
        if advanced:
            lines.append("[advanced]")
            for key, val in advanced.items():
                lines.append(f"{key}={val}")
            lines.append("")

        cfg_path = build_dir / "connectivity.cfg"
        cfg_path.write_text("\n".join(lines))
        logger.info("    %d sp + %d stream_connect", len(sp_entries), len(stream_entries))

    def _resolve_stream_ports(self, spec, tensor_name: str) -> list[str]:
        """Resolve tensor/interface name to v++ stream port name(s)."""
        from vten.codegen.xrt_generator import _flat_ext_port

        if tensor_name in spec.interfaces:
            iface = spec.interfaces[tensor_name]
            if iface.array:
                return [_flat_ext_port(iface, fn)
                        for fn in iface.array.flat_names(tensor_name)]
            return [iface.ext_port]

        for iname, iface in spec.interfaces.items():
            if iface.tensor == tensor_name or (
                iface.tensors and tensor_name in iface.tensors
            ):
                if iface.array:
                    return [_flat_ext_port(iface, fn)
                            for fn in iface.array.flat_names(iname)]
                return [iface.ext_port]

        raise BuildError(
            f"Cannot resolve stream port for '{tensor_name}' "
            f"in kernel '{spec.kernel_name}'"
        )

    @staticmethod
    def _find_stream_interface(spec, tensor_name: str):
        """Return the InterfaceSpec for a stream tensor, or None."""
        if tensor_name in spec.interfaces:
            return spec.interfaces[tensor_name]
        for _iname, iface in spec.interfaces.items():
            if iface.tensor == tensor_name or (
                iface.tensors and tensor_name in iface.tensors
            ):
                return iface
        return None

    def _stage_vpp_link_composite(
        self,
        composite_name: str,
        sub_kernel_dirs: dict[str, Path],
        force: bool,
    ) -> None:
        """v++ link: combine all sub-kernel .xo → xclbin."""
        xrt_cfg = self._config.get("backend", {}).get("xrt", {})
        target = xrt_cfg.get("target", "hw_emu")
        platform = xrt_cfg.get("platform", "")

        build_dir = self._composite_build_dir(composite_name)
        cfg_file = build_dir / "connectivity.cfg"
        xclbin_file = build_dir / f"{composite_name}_{target}.xclbin"

        if not platform:
            raise BuildError(
                "platform not set in [backend.xrt]. "
                "Set platform path (e.g. /opt/xilinx/platforms/.../*.xpfm)"
            )
        if not cfg_file.exists():
            raise BuildError(
                "connectivity.cfg not found. Run gen_link_config first."
            )

        # Collect .xo from sub-kernel build dirs
        xo_files: list[Path] = []
        for sub_dir in sorted(sub_kernel_dirs.values()):
            xrt_dir = sub_dir / "build" / "xrt"
            for xo in sorted(xrt_dir.glob("*.xo")):
                xo_files.append(xo)

        if not xo_files:
            raise BuildError("No .xo files found. Run gen_xo first.")

        if xclbin_file.exists() and not force:
            logger.info("  [vpp_link] %s (cached)", composite_name)
            return

        vpp = _find_tool("v++", self._config)
        if not vpp:
            raise BuildError(
                "v++ not found in PATH. Source Vitis settings.\n"
                "To generate artifacts only: vten build --backend xrt --upto gen_xo"
            )

        logger.info("  [vpp_link] %s (%s) — %d kernels", composite_name, target, len(xo_files))
        for xo in xo_files:
            logger.info("    .xo: %s", xo.name)

        log_dir = build_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Build v++ link command
        link_cfg = xrt_cfg.get("link", {})
        vpp_cmd = [
            vpp, "-l", "-t", target,
            "--platform", platform,
            "--config", str(cfg_file),
            "--save-temps",
            "--temp_dir", str(build_dir / "_x"),
            "--log_dir", str(log_dir),
            "--report_dir", str(log_dir),
            "-o", str(xclbin_file),
        ]

        # --debug flag (required for waveform in hw_emu)
        if link_cfg.get("debug", False):
            vpp_cmd.append("--debug")

        # --linkhook.custom postSysLink (e.g. reset patch for user_managed)
        linkhook = link_cfg.get("linkhook_post_syslink", "")
        if linkhook:
            hook_path = self._project / linkhook
            if hook_path.exists():
                vpp_cmd.append(
                    f"--linkhook.custom=postSysLink,{hook_path}"
                )
                logger.info("    linkhook: postSysLink → %s", hook_path.name)
            else:
                logger.warning("    linkhook file not found: %s", hook_path)

        vpp_cmd += [str(xo) for xo in xo_files]

        self._run_subprocess(
            vpp_cmd,
            cwd=build_dir,
            log_file=log_dir / "vpp_link.log",
            stage_name="vpp_link",
        )
        self._cleanup_stray_files(build_dir)

        if not xclbin_file.exists():
            raise BuildError(f"xclbin not generated: {xclbin_file}")

        size_mb = xclbin_file.stat().st_size / (1024 * 1024)
        logger.info("    xclbin: %s (%.1fMB)", xclbin_file.name, size_mb)

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
                logger.info("    emconfig.json generated")

    def _stage_validate_composite(self, composite_name: str) -> None:
        """Validate composite xclbin."""
        xrt_cfg = self._config.get("backend", {}).get("xrt", {})
        target = xrt_cfg.get("target", "hw_emu")
        build_dir = self._composite_build_dir(composite_name)

        logger.info("  [validate] %s", composite_name)
        candidate = build_dir / f"{composite_name}_{target}.xclbin"
        if candidate.exists():
            logger.info("    xclbin exists: %s", candidate.name)
            self._fix_ip_layout(candidate)
        else:
            logger.info("    xclbin not found, skip")

    # ── Single-kernel stages ──

    def _stage_vpp_link_single(self, kernel_dir: Path, force: bool) -> None:
        """Single-kernel v++ link."""
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
            raise BuildError("platform not set in [backend.xrt].")
        if not cfg_file.exists():
            raise BuildError("connectivity.cfg not found. Run gen_xrt_packaging first.")

        if xclbin_file.exists() and not force:
            logger.info("  [vpp_link] %s (cached)", kernel_dir.name)
            return

        vpp = _find_tool("v++", self._config)
        if not vpp:
            raise BuildError("v++ not found in PATH. Source Vitis settings.")

        logger.info("  [vpp_link] %s (%s)", kernel_dir.name, target)
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
        logger.info("    xclbin: %s (%.1fMB)", xclbin_file.name, size_mb)

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
                logger.info("    emconfig.json generated")

    def _stage_validate_single(self, kernel_dir: Path) -> None:
        """Validate single-kernel xclbin."""
        logger.info("  [validate] %s", kernel_dir.name)
        xrt_cfg = self._config.get("backend", {}).get("xrt", {})
        build_dir = self._build_dir(kernel_dir)
        target = xrt_cfg.get("target", "hw_emu")
        spec = self._load_spec(kernel_dir)
        candidate = build_dir / f"{spec.kernel_name}_{target}.xclbin"
        if candidate.exists():
            logger.info("    xclbin exists: %s", candidate.name)
            self._fix_ip_layout(candidate)
        else:
            logger.info("    xclbin not found, skip")

    # ── IP_LAYOUT fixup ──

    def _fix_ip_layout(self, xclbin_path: Path) -> None:
        """Fix IP_LAYOUT entries with 'not_used' base address.

        v++ sometimes fails to populate base addresses in the IP_LAYOUT
        section for stream-only kernels with only AXI-Lite control (scalar
        args). The actual addresses ARE assigned in EMBEDDED_METADATA.

        This extracts the real addresses from EMBEDDED_METADATA and patches
        the IP_LAYOUT section via xclbinutil.
        """
        import json
        import tempfile
        import xml.etree.ElementTree as ET

        xclbinutil = shutil.which("xclbinutil")
        if not xclbinutil:
            return

        # 1. Extract IP_LAYOUT
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            ip_layout_path = f.name

        try:
            subprocess.run(
                [xclbinutil, "--input", str(xclbin_path),
                 "--dump-section", f"IP_LAYOUT:JSON:{ip_layout_path}",
                 "--force"],
                capture_output=True, timeout=30,
            )
            with open(ip_layout_path) as f:
                ip_layout = json.load(f)
        except Exception:
            return
        finally:
            Path(ip_layout_path).unlink(missing_ok=True)

        # 2. Find entries with not_used
        broken = []
        for entry in ip_layout.get("ip_layout", {}).get("m_ip_data", []):
            if entry.get("m_base_address") == "not_used":
                # Extract kernel name from "kernel_name:instance_name"
                cu_name = entry.get("m_name", "")
                broken.append(cu_name)

        if not broken:
            return

        logger.warning(
            "    IP_LAYOUT has 'not_used' entries: %s — attempting auto-fix",
            broken,
        )

        # 3. Extract EMBEDDED_METADATA for real addresses
        with tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False,
        ) as f:
            metadata_path = f.name

        try:
            subprocess.run(
                [xclbinutil, "--input", str(xclbin_path),
                 "--dump-section", f"EMBEDDED_METADATA:RAW:{metadata_path}",
                 "--force"],
                capture_output=True, timeout=30,
            )
            tree = ET.parse(metadata_path)
            root = tree.getroot()
        except Exception:
            logger.warning("    Failed to parse EMBEDDED_METADATA, skip fix")
            return
        finally:
            Path(metadata_path).unlink(missing_ok=True)

        # 4. Build instance_name → base_address map from metadata
        addr_map: dict[str, str] = {}
        for kernel_el in root.iter("kernel"):
            for instance_el in kernel_el.iter("instance"):
                inst_name = instance_el.get("name", "")
                for remap in instance_el.iter("addrRemap"):
                    base = remap.get("base", "")
                    if base and inst_name:
                        addr_map[inst_name] = base

        # 5. Patch IP_LAYOUT
        patched = False
        for entry in ip_layout["ip_layout"]["m_ip_data"]:
            if entry.get("m_base_address") != "not_used":
                continue
            cu_name = entry.get("m_name", "")
            # cu_name format: "kernel_name:instance_name"
            inst_name = cu_name.split(":")[-1] if ":" in cu_name else cu_name
            if inst_name in addr_map:
                entry["m_base_address"] = addr_map[inst_name]
                logger.info(
                    "    Fixed %s: not_used → %s",
                    cu_name, addr_map[inst_name],
                )
                patched = True

        if not patched:
            logger.warning("    Could not find addresses in EMBEDDED_METADATA")
            return

        # 6. Write patched IP_LAYOUT back to xclbin
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(ip_layout, f, indent=4)
            patched_path = f.name

        try:
            result = subprocess.run(
                [xclbinutil, "--input", str(xclbin_path),
                 "--replace-section", f"IP_LAYOUT:JSON:{patched_path}",
                 "--output", str(xclbin_path),
                 "--force"],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info("    IP_LAYOUT patched successfully")
            else:
                logger.warning(
                    "    xclbinutil patch failed: %s",
                    result.stderr.decode()[:200],
                )
        except Exception as e:
            logger.warning("    xclbinutil patch error: %s", e)
        finally:
            Path(patched_path).unlink(missing_ok=True)
