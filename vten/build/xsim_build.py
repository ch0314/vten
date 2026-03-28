"""XsimBuildPipeline — Vivado xsim 5-stage build pipeline.

Spec reference: 06_codegen_and_cli.md §4.3, §7, §8

Pipeline stages:
  Stage 1: project_setup  — Vivado project creation (cached)
  Stage 2: dpi_c          — gcc shared library (cached)
  Stage 3: codegen        — Jinja2 → generated SV (per-kernel, cached)
  Stage 4: compile_order  — Vivado get_compile_order (per-kernel, cached)
  Stage 5: compile        — xvlog + xelab (per-kernel, cached)
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

from vten.build.base import BuildPipeline
from vten.build.common import (
    cache_valid,
    dir_hash,
    expand_globs,
    file_hash,
    load_cache,
    render_template,
    run_vivado,
    save_cache,
    update_cache,
)
from vten.codegen.sv_generator import SVGenerator
from vten.errors import BuildError
from vten.runtime.ir import BFMConfig
from vten.spec.models import InterfaceSpec, KernelSpec, Protocol
from vten.spec.parser import parse_kernel_spec

logger = logging.getLogger(__name__)


# ── Split interface expansion ──


def _expand_split_interfaces(spec: KernelSpec) -> KernelSpec:
    """Expand split interfaces into individual port interfaces."""
    has_split = False
    for iface in spec.interfaces.values():
        if iface.split and isinstance(iface.split, dict) and "ports" in iface.split:
            has_split = True
            break
    if not has_split:
        return spec

    from dataclasses import replace

    expanded: dict[str, InterfaceSpec] = {}
    for name, iface in spec.interfaces.items():
        if iface.split and isinstance(iface.split, dict) and "ports" in iface.split:
            for port in iface.split["ports"]:
                port_name = port["name"]
                expanded[port_name] = InterfaceSpec(
                    name=port_name,
                    rtl_port=port_name,
                    protocol=iface.protocol,
                    data_width=iface.data_width,
                    addr_width=iface.addr_width,
                    memory_region=iface.memory_region,
                    tensor=iface.tensor,
                    tensors=iface.tensors,
                    packing=iface.packing,
                    role=iface.role,
                )
        else:
            expanded[name] = iface
    return replace(spec, interfaces=expanded)


# ── BFM inference ──


def _infer_bfm_role(iface: InterfaceSpec) -> str:
    """Infer BFM role from explicit role, protocol, and interface conventions."""
    if iface.role:
        # Explicit role: slave DUT port → BFM drives (master), and vice versa
        return "master" if iface.role == "slave" else "slave"
    if iface.protocol == Protocol.AXI4L:
        return "master"
    if iface.protocol == Protocol.AXI4:
        return "slave"
    if iface.rtl_port and iface.rtl_port.startswith("s_"):
        return "master"
    return "slave"


def _derive_bfm_configs(spec: KernelSpec) -> list[BFMConfig]:
    """Derive BFMConfig list from KernelSpec interfaces.

    Array interfaces are expanded into per-element BFMConfigs using
    ArraySpec.flat_names(), so each array element gets its own BFM.
    """
    configs: list[BFMConfig] = []
    for name, iface in spec.interfaces.items():
        if iface.array:
            for flat_name in iface.array.flat_names(name):
                configs.append(BFMConfig(
                    interface_name=flat_name,
                    protocol=iface.protocol,
                    data_width=iface.data_width or 256,
                    addr_width=iface.addr_width or 64,
                    role=_infer_bfm_role(iface),
                ))
        else:
            configs.append(BFMConfig(
                interface_name=name,
                protocol=iface.protocol,
                data_width=iface.data_width or 256,
                addr_width=iface.addr_width or 64,
                role=_infer_bfm_role(iface),
            ))
    return configs


# ── XsimBuildPipeline ──


class XsimBuildPipeline(BuildPipeline):
    """Vivado xsim 5-stage build pipeline."""

    _STAGES = ["project_setup", "dpi_c", "codegen", "compile_order", "compile"]
    _PROJECT_STAGES = {"project_setup", "dpi_c"}

    def __init__(self, project: Path, config: dict) -> None:
        super().__init__(project, config)
        xsim_cfg = config.get("backend", {}).get("xsim", {})
        self._vivado_path = xsim_cfg.get("vivado_path", "")
        self._vten_root = Path(__file__).resolve().parent.parent.parent
        self._vten_sv_dir = self._vten_root / "vten_sv"
        self._cache = load_cache(project / "build" / ".cache.json")
        # Normalize [ip] table format to [[ip]] list format
        self._config["ip"] = self._normalize_ip_config(self._config.get("ip"))

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
        if stage == "project_setup":
            self._stage_project_setup(force)
        elif stage == "dpi_c":
            self._stage_dpi_c(force)
        elif stage in ("codegen", "compile_order", "compile"):
            assert kernel_name is not None and kernel_dir is not None
            # Per-kernel cache check
            if force:
                self._invalidate_kernel_cache(kernel_name)
            if self._kernel_cache_valid(kernel_name, stage, kernel_dir):
                logger.info("[%s] %s: cached, skip", stage, kernel_name)
                return
            # Run stage
            if stage == "codegen":
                self._stage_codegen(kernel_dir, force=force)
            elif stage == "compile_order":
                self._stage_compile_order(kernel_dir)
            elif stage == "compile":
                self._stage_compile(kernel_dir)
            # Update cache after successful execution
            self._update_kernel_cache(kernel_name, stage, kernel_dir)
        else:
            raise BuildError(f"Unknown stage: {stage}")

    def build(self, **kwargs) -> None:
        """Override to save cache after build."""
        try:
            super().build(**kwargs)
        finally:
            save_cache(self._project / "build" / ".cache.json", self._cache)

    # ── Stage implementations ──

    def _project_setup_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self._config.get("rtl", {}), sort_keys=True).encode())
        h.update(json.dumps(self._config.get("ip", []), sort_keys=True).encode())
        h.update(
            self._config.get("backend", {}).get("xsim", {}).get("part", "").encode()
        )
        for p in sorted(self._vten_sv_dir.glob("*.sv")) + sorted(
            self._vten_sv_dir.glob("*.svh")
        ):
            h.update(p.read_bytes())
        for glob_pat in self._config.get("rtl", {}).get("sources", []):
            for p in sorted(self._project.glob(glob_pat)):
                h.update(p.read_bytes())
        # Hash existing .xci file contents
        for entry in self._config.get("ip", []):
            if "source" in entry:
                for p in sorted(self._project.glob(entry["source"])):
                    h.update(p.read_bytes())
        return h.hexdigest()

    def _stage_project_setup(self, force: bool) -> None:
        logger.info("[Stage 1] project_setup")
        current = self._project_setup_hash()
        if not force and cache_valid(self._cache, "project_setup", current):
            logger.info("  cached, skip")
            return

        rtl_patterns = self._config.get("rtl", {}).get("sources", [])
        rtl_files = expand_globs(self._project, rtl_patterns)

        ip_sources, ip_create = self._parse_ip_entries(
            self._config.get("ip", []), self._project,
        )

        tcl = render_template("project_setup.tcl.j2", {
            "rtl_files": rtl_files,
            "include_dirs": self._config.get("rtl", {}).get("include_dirs", []),
            "ip_sources": ip_sources,
            "ip_create": ip_create,
        })
        tcl_path = self._project / "build" / "project_setup.tcl"
        tcl_path.parent.mkdir(parents=True, exist_ok=True)
        tcl_path.write_text(tcl)

        part = self._config.get("backend", {}).get("xsim", {}).get("part", "")
        proj_dir = self._project / "build" / "vivado_proj"
        proj_dir.mkdir(parents=True, exist_ok=True)

        log_dir = self._project / "build" / "logs"
        run_vivado(
            self._vivado_path, tcl_path, proj_dir, part,
            self._vten_sv_dir, self._project,
            log_dir=log_dir, label="project_setup",
        )

        update_cache(self._cache, "project_setup", current)
        logger.info("  done")

    def _stage_dpi_c(self, force: bool) -> None:
        logger.info("[Stage 2] dpi_c")
        src_c = self._vten_sv_dir / "vten_shm_bridge.c"
        src_h = self._vten_sv_dir / "vten_shm_bridge.h"
        current = dir_hash([p for p in [src_c, src_h] if p.exists()])

        if not force and cache_valid(self._cache, "dpi_c", current):
            logger.info("  cached, skip")
            return

        so_path = self._project / "build" / "lib" / "libvten_shm.so"
        so_path.parent.mkdir(parents=True, exist_ok=True)

        include_args: list[str] = []
        if self._vivado_path:
            xsim_include = Path(self._vivado_path) / "data" / "xsim" / "include"
            if xsim_include.exists():
                include_args += ["-I", str(xsim_include)]
        include_args += ["-I", str(self._vten_sv_dir)]

        result = subprocess.run(
            [
                "gcc", "-shared", "-fPIC",
                *include_args,
                "-o", str(so_path),
                str(src_c),
                "-lrt", "-lpthread",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug("gcc stderr:\n%s", result.stderr)
            raise BuildError(f"gcc failed:\n{result.stderr}")

        update_cache(self._cache, "dpi_c", current)
        logger.info("  done")

    # ── Per-kernel cache system ──

    def _kernel_stage_hash(
        self, kernel_name: str, stage: str, kernel_dir: Path,
    ) -> str:
        """Compute SHA256 hash for a per-kernel stage's inputs."""
        from vten.build.composite import (
            get_sub_kernel_names,
            is_composite_kernel,
            load_composite_class,
        )

        h = hashlib.sha256()
        is_composite = is_composite_kernel(kernel_dir)

        if stage == "codegen":
            if is_composite:
                # Composite codegen depends on: *_kernel.py + sub-kernel specs + params
                for py_file in sorted(kernel_dir.glob("*_kernel.py")):
                    h.update(py_file.read_bytes())
                composite_cls = load_composite_class(kernel_dir)
                for sname in get_sub_kernel_names(composite_cls):
                    sub_spec = self._project / "kernels" / sname / "kernel_spec.yaml"
                    if sub_spec.exists():
                        h.update(sub_spec.read_bytes())
                    # Include sub-kernel codegen hash for dependency chaining
                    sub_entry = self._cache.get(f"{sname}:codegen", {})
                    h.update(sub_entry.get("hash", "").encode())
            else:
                # Unit codegen depends on: kernel_spec.yaml + params
                spec_path = kernel_dir / "kernel_spec.yaml"
                if spec_path.exists():
                    h.update(spec_path.read_bytes())
            # Common: parameters from config
            params = self._config.get("parameters", {})
            h.update(json.dumps(params, sort_keys=True).encode())

        elif stage == "compile_order":
            # Depends on: generated/*.sv + project_setup hash
            gen_dir = kernel_dir / "build" / "generated"
            if gen_dir.exists():
                for sv in sorted(gen_dir.glob("*.sv")):
                    h.update(sv.read_bytes())
            ps_entry = self._cache.get("project_setup", {})
            h.update(ps_entry.get("hash", "").encode())
            if is_composite:
                composite_cls = load_composite_class(kernel_dir)
                for sname in get_sub_kernel_names(composite_cls):
                    sub_entry = self._cache.get(f"{sname}:compile_order", {})
                    h.update(sub_entry.get("hash", "").encode())

        elif stage == "compile":
            # Depends on: compile.prj content + all referenced sources + dpi_c hash
            prj_path = kernel_dir / "build" / "compile.prj"
            if prj_path.exists():
                prj_text = prj_path.read_text()
                h.update(prj_text.encode())
                for line in prj_text.splitlines():
                    parts = line.split()
                    if len(parts) >= 3:
                        src = Path(parts[-1])
                        if src.exists():
                            h.update(src.read_bytes())
            dpi_entry = self._cache.get("dpi_c", {})
            h.update(dpi_entry.get("hash", "").encode())

        return h.hexdigest()

    def _kernel_artifacts_exist(
        self, stage: str, kernel_dir: Path,
    ) -> bool:
        """Check that stage output artifacts actually exist on disk."""
        if stage == "codegen":
            gen_dir = kernel_dir / "build" / "generated"
            return gen_dir.exists() and any(gen_dir.glob("*.sv"))
        elif stage == "compile_order":
            return (kernel_dir / "build" / "compile.prj").exists()
        elif stage == "compile":
            return (kernel_dir / "build" / "xsim.dir").exists()
        return False

    def _kernel_cache_valid(
        self, kernel_name: str, stage: str, kernel_dir: Path,
    ) -> bool:
        """Check if a per-kernel stage can be skipped (hash match + artifacts exist)."""
        cache_key = f"{kernel_name}:{stage}"
        current = self._kernel_stage_hash(kernel_name, stage, kernel_dir)
        return (
            cache_valid(self._cache, cache_key, current)
            and self._kernel_artifacts_exist(stage, kernel_dir)
        )

    def _update_kernel_cache(
        self, kernel_name: str, stage: str, kernel_dir: Path,
    ) -> None:
        """Update cache entry for a per-kernel stage."""
        cache_key = f"{kernel_name}:{stage}"
        current = self._kernel_stage_hash(kernel_name, stage, kernel_dir)
        update_cache(self._cache, cache_key, current)

    def _invalidate_kernel_cache(self, kernel_name: str) -> None:
        """Remove all cache entries for a kernel (used by --force)."""
        for stage in ("codegen", "compile_order", "compile"):
            self._cache.pop(f"{kernel_name}:{stage}", None)

    def _ensure_sub_kernels_built(
        self, kernel_dir: Path, force: bool,
    ) -> None:
        """Auto-build sub-kernels if needed (replaces check_sub_kernels_built)."""
        from vten.build.composite import (
            get_sub_kernel_names,
            load_composite_class,
        )

        composite_cls = load_composite_class(kernel_dir)
        sub_names = get_sub_kernel_names(composite_cls)

        for sname in sub_names:
            sub_dir = self._project / "kernels" / sname
            if not sub_dir.exists():
                raise BuildError(
                    f"Sub-kernel directory not found: {sub_dir}"
                )

            if force:
                self._invalidate_kernel_cache(sname)

            needs_build = False
            for stage in ("codegen", "compile_order", "compile"):
                if not self._kernel_cache_valid(sname, stage, sub_dir):
                    needs_build = True
                    break

            if needs_build:
                logger.info("  --- Auto-building sub-kernel: %s ---", sname)
                for stage in ("codegen", "compile_order", "compile"):
                    if not self._kernel_cache_valid(sname, stage, sub_dir):
                        self.run_stage(
                            stage,
                            kernel_name=sname,
                            kernel_dir=sub_dir,
                            force=False,  # individual cache already checked
                        )

    # ── Stage implementations ──

    def _stage_codegen(self, kernel_dir: Path, force: bool = False) -> None:
        logger.info("[Stage 3] codegen: %s", kernel_dir.name)

        from vten.build.composite import is_composite_kernel

        if is_composite_kernel(kernel_dir):
            self._stage_codegen_composite(kernel_dir, force=force)
            return

        spec_path = kernel_dir / "kernel_spec.yaml"
        spec = parse_kernel_spec(spec_path)
        spec = _expand_split_interfaces(spec)
        bfm_configs = _derive_bfm_configs(spec)

        gen = SVGenerator(
            kernel_spec=spec,
            bfm_configs=bfm_configs,
            project_config=self._config,
        )

        output = kernel_dir / "build" / "generated"
        output.mkdir(parents=True, exist_ok=True)
        gen.generate(str(output), num_commands=256)
        logger.info("  done")

    def _stage_codegen_composite(self, kernel_dir: Path, force: bool = False) -> None:
        """Composite kernel codegen: auto-build sub-kernels + synthesize spec + generate wrapper."""
        from vten.build.composite import (
            extract_probe_bfm_info,
            generate_composite_sv,
            load_composite_class,
            synthesize_spec,
        )

        composite_cls = load_composite_class(kernel_dir)
        kernel_name = kernel_dir.name

        # Auto-build sub-kernels if needed (replaces check_sub_kernels_built)
        self._ensure_sub_kernels_built(kernel_dir, force)

        # Synthesize KernelSpec from sub-kernel specs + connectivity
        spec = synthesize_spec(composite_cls, self._project, kernel_name)
        bfm_configs = _derive_bfm_configs(spec)

        # Extract internal probe BFM info for codegen
        probe_bfms = extract_probe_bfm_info(composite_cls, self._project)

        output = kernel_dir / "build" / "generated"
        output.mkdir(parents=True, exist_ok=True)

        # Generate composite top SV (wrapper-of-wrappers)
        generate_composite_sv(
            composite_cls, spec, self._project, output
        )

        # Generate tb_top.sv using standard SVGenerator with synthesized spec
        # Override top_module to use composite module name
        gen = SVGenerator(
            kernel_spec=spec,
            bfm_configs=bfm_configs,
            project_config=self._config,
            probe_bfms=probe_bfms,
        )
        gen.generate(str(output), num_commands=256)
        logger.info("  done")

    def _stage_compile_order(self, kernel_dir: Path) -> None:
        logger.info("[Stage 4] compile_order: %s", kernel_dir.name)
        tb_top = kernel_dir / "build" / "generated" / "tb_top.sv"
        prj_out = kernel_dir / "build" / "compile.prj"

        if not tb_top.exists():
            raise BuildError(
                f"tb_top.sv not found: {tb_top}. Run codegen first."
            )

        xpr_path = self._project / "build" / "vivado_proj" / "vten_sim.xpr"
        if not xpr_path.exists():
            raise BuildError(
                f"Vivado project not found: {xpr_path}. Run project_setup first."
            )

        tcl = self._vten_root / "templates" / "resolve_order.tcl"
        log_dir = kernel_dir / "build" / "logs"
        run_vivado(
            self._vivado_path, tcl, xpr_path, tb_top, prj_out,
            log_dir=log_dir, label="compile_order",
        )

        # For composite kernels, prepend sub-kernel generated files
        from vten.build.composite import (
            get_sub_kernel_names,
            is_composite_kernel,
            load_composite_class,
        )
        if is_composite_kernel(kernel_dir):
            composite_cls = load_composite_class(kernel_dir)
            sub_names = get_sub_kernel_names(composite_cls)
            composite_lines = prj_out.read_text().splitlines()

            # Build merged order: sub-kernel orders first (they have
            # correct dependency ordering), then composite-only files.
            seen_files: set[str] = set()
            merged_lines: list[str] = []

            for sname in sub_names:
                sub_prj = self._project / "kernels" / sname / "build" / "compile.prj"
                if sub_prj.exists():
                    for line in sub_prj.read_text().splitlines():
                        if not line.strip():
                            continue
                        fpath = line.split()[-1]
                        if Path(fpath).name == "tb_top.sv":
                            continue
                        if fpath not in seen_files:
                            merged_lines.append(line)
                            seen_files.add(fpath)
                else:
                    gen_dir = self._project / "kernels" / sname / "build" / "generated"
                    for sv_file in sorted(gen_dir.glob("*.sv")):
                        if sv_file.name == "tb_top.sv":
                            continue
                        fpath = str(sv_file)
                        if fpath not in seen_files:
                            merged_lines.append(f"sv xil_defaultlib {sv_file}")
                            seen_files.add(fpath)

            # Append composite-specific files (generated wrappers, etc.)
            for line in composite_lines:
                if not line.strip():
                    continue
                fpath = line.split()[-1]
                if fpath not in seen_files:
                    merged_lines.append(line)
                    seen_files.add(fpath)

            prj_out.write_text("\n".join(merged_lines) + "\n")

        logger.info("  done")

    def _stage_compile(self, kernel_dir: Path) -> None:
        logger.info("[Stage 5] compile: %s", kernel_dir.name)
        prj = kernel_dir / "build" / "compile.prj"
        dpi_lib = self._project / "build" / "lib" / "libvten_shm"

        if not prj.exists():
            raise BuildError(
                f"compile.prj not found: {prj}. Run compile_order first."
            )

        build_dir = kernel_dir / "build"
        log_dir = build_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # xvlog
        xvlog_cmd = [
            f"{self._vivado_path}/bin/xvlog", "--sv",
            "--include", str(self._vten_sv_dir),
        ]
        for inc_dir in self._config.get("rtl", {}).get("include_dirs", []):
            inc_path = Path(inc_dir)
            if not inc_path.is_absolute():
                inc_path = self._project / inc_path
            xvlog_cmd += ["--include", str(inc_path.resolve())]
        xvlog_cmd += [
            "--prj", str(prj),
            "--log", str(log_dir / "xvlog.log"),
        ]
        result = subprocess.run(
            xvlog_cmd,
            capture_output=True,
            text=True,
            cwd=str(build_dir),
        )
        if result.returncode != 0:
            logger.debug("xvlog stdout:\n%s", result.stdout)
            logger.debug("xvlog stderr:\n%s", result.stderr)
            logger.error("xvlog log: %s", log_dir / "xvlog.log")
            raise BuildError(f"xvlog failed:\n{result.stderr[-500:]}")

        # Compile glbl.v if backend.xsim.glbl is set (Xilinx primitive library support)
        xsim_cfg = self._config.get("backend", {}).get("xsim", {})
        use_glbl = xsim_cfg.get("glbl", False)
        if use_glbl:
            glbl_path = (
                Path(self._vivado_path) / "data" / "verilog" / "src" / "glbl.v"
            )
            result = subprocess.run(
                [
                    f"{self._vivado_path}/bin/xvlog",
                    str(glbl_path),
                    "--log", str(log_dir / "xvlog_glbl.log"),
                ],
                capture_output=True,
                text=True,
                cwd=str(build_dir),
            )
            if result.returncode != 0:
                logger.debug("xvlog glbl stderr:\n%s", result.stderr)
                raise BuildError(f"xvlog glbl.v failed:\n{result.stderr[-500:]}")

        # Determine elab top from compile.prj
        elab_top = "tb_top"
        prj_text = prj.read_text()
        for line in prj_text.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2].endswith("/tb_top.sv"):
                lib = parts[1]
                if lib != "work":
                    elab_top = f"{lib}.tb_top"
                break

        # Collect library names for -L flags
        prj_libs = set()
        for line in prj_text.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                prj_libs.add(parts[1])
        lib_args: list[str] = []
        for lib in sorted(prj_libs):
            if lib != "work":
                lib_args += ["-L", lib]

        # Append IP-specific xelab libraries (unisims_ver, xpm, etc.)
        for lib in xsim_cfg.get("xelab_libs", []):
            lib_args += ["-L", lib]

        # xelab
        elab_tops = [elab_top]
        if use_glbl:
            elab_tops.append("work.glbl")
        result = subprocess.run(
            [
                f"{self._vivado_path}/bin/xelab", *elab_tops,
                *lib_args,
                "--sv_lib", dpi_lib.name,
                "--sv_root", str(dpi_lib.parent),
                "--timescale", "1ns/1ps",
                "--debug", "typical",
                "--snapshot", "tb_top",
                "--relax",
                "--log", str(log_dir / "xelab.log"),
            ],
            capture_output=True,
            text=True,
            cwd=str(build_dir),
        )
        if result.returncode != 0:
            logger.debug("xelab stdout:\n%s", result.stdout)
            logger.debug("xelab stderr:\n%s", result.stderr)
            logger.error("xelab log: %s", log_dir / "xelab.log")
            detail = result.stdout[-500:] if result.stdout else result.stderr[-500:]
            raise BuildError(f"xelab failed:\n{detail}")

        # Clean up stray vivado/xsim files from build dir
        import shutil
        xil_dir = build_dir / ".Xil"
        if xil_dir.exists():
            shutil.rmtree(xil_dir, ignore_errors=True)
        webtalk_dir = build_dir / "webtalk"
        if webtalk_dir.exists():
            shutil.rmtree(webtalk_dir, ignore_errors=True)
        for pattern in ("*.log", "*.jou", "*.pb"):
            for f in build_dir.glob(pattern):
                if f.is_file():
                    f.rename(log_dir / f.name)

        logger.info("  done")
