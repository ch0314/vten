"""vten build: 5-stage compilation pipeline.

Spec reference: 06_codegen_and_cli.md §4.3, §7, §8

Pipeline stages:
  Stage 1: project_setup  — Vivado project creation (cached)
  Stage 2: dpi_c          — gcc shared library (cached)
  Stage 3: codegen        — Jinja2 → generated SV (per-kernel)
  Stage 4: compile_order  — Vivado get_compile_order (per-kernel)
  Stage 5: compile        — xvlog + xelab (per-kernel)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import jinja2

from vten.cli.config import load_project_config
from vten.codegen.sv_generator import SVGenerator
from vten.errors import BuildError
from vten.runtime.ir import BFMConfig
from vten.spec.models import InterfaceSpec, KernelSpec, Protocol
from vten.spec.parser import parse_kernel_spec


# ── Split interface expansion ──


def _expand_split_interfaces(spec: KernelSpec) -> KernelSpec:
    """Expand split interfaces into individual port interfaces.

    An interface with split.ports generates N interfaces (one per physical port),
    replacing the parent. Each expanded port inherits protocol, data_width,
    addr_width, memory_region, packing from the parent.
    """
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
                )
        else:
            expanded[name] = iface
    return replace(spec, interfaces=expanded)


# ── BFM inference ──


def _infer_bfm_role(iface: InterfaceSpec) -> str:
    """Infer BFM role from protocol and interface conventions.

    AXI4-Stream: rtl_port 's_*' → DUT is slave → BFM is master (pushes data)
                 rtl_port 'm_*' → DUT is master → BFM is slave (pulls data)
    AXI4:        BFM is always slave (DUT initiates reads/writes)
    AXI4-Lite:   BFM is always master (drives register access)
    """
    if iface.protocol == Protocol.AXI4L:
        return "master"
    if iface.protocol == Protocol.AXI4:
        return "slave"
    # AXI4-Stream: infer from rtl_port prefix
    if iface.rtl_port and iface.rtl_port.startswith("s_"):
        return "master"  # DUT slave input → BFM drives data
    return "slave"  # DUT master output → BFM receives data


def _derive_bfm_configs(spec: KernelSpec) -> list[BFMConfig]:
    """Derive BFMConfig list from KernelSpec interfaces."""
    configs: list[BFMConfig] = []
    for name, iface in spec.interfaces.items():
        cfg = BFMConfig(
            interface_name=name,
            protocol=iface.protocol,
            data_width=iface.data_width or 256,
            addr_width=iface.addr_width or 64,
            role=_infer_bfm_role(iface),
        )
        configs.append(cfg)
    return configs


# ── Cache system (SHA256-based) ──


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def file_hash(path: Path) -> str:
    """Single file SHA256."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dir_hash(paths: list[Path]) -> str:
    """Combined SHA256 of multiple files. Sorted by path, hashing (path + content)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def cache_valid(cache: dict, key: str, current_hash: str) -> bool:
    entry = cache.get(key)
    if not entry:
        return False
    return entry.get("hash") == current_hash


def update_cache(cache: dict, key: str, current_hash: str) -> None:
    cache[key] = {
        "hash": current_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Helpers ──


def discover_kernels(project: Path) -> list[str]:
    """Discover kernels with kernel_spec.yaml under kernels/ directory."""
    kernels_dir = project / "kernels"
    if not kernels_dir.exists():
        return []
    return sorted(
        d.name for d in kernels_dir.iterdir()
        if d.is_dir() and (d / "kernel_spec.yaml").exists()
    )


def find_kernel_spec(project: Path, kernel_name: str) -> Path:
    """Resolve kernel spec path. Only kernels/<name>/kernel_spec.yaml supported."""
    path = project / "kernels" / kernel_name / "kernel_spec.yaml"
    if not path.exists():
        raise BuildError(
            f"kernel_spec.yaml not found: {path}\n"
            f"Run: vten init --kernel {kernel_name}"
        )
    return path


def _resolve_stages(
    all_stages: list[str],
    stage: str | None,
    upto: str | None,
    skip_compile: bool,
) -> list[str]:
    if skip_compile:
        return ["codegen"]
    if stage:
        if stage not in all_stages:
            raise BuildError(
                f"Unknown stage '{stage}'. "
                f"Valid stages: {', '.join(all_stages)}"
            )
        return [stage]
    if upto:
        if upto not in all_stages:
            raise BuildError(
                f"Unknown stage '{upto}'. "
                f"Valid stages: {', '.join(all_stages)}"
            )
        idx = all_stages.index(upto)
        return all_stages[: idx + 1]
    return all_stages


def _run_vivado(vivado_path: str, tcl_script: Path, *args: str | Path) -> None:
    cmd = [
        f"{vivado_path}/bin/vivado", "-mode", "batch",
        "-source", str(tcl_script),
    ]
    if args:
        cmd += ["-tclargs"] + [str(a) for a in args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(
            f"Vivado failed (exit {result.returncode}):\n{result.stderr[-500:]}"
        )


def _render_template(name: str, context: dict) -> str:
    vten_root = Path(__file__).resolve().parent.parent.parent
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(vten_root / "templates")),
    )
    return env.get_template(name).render(context)


def _expand_globs(project: Path, patterns: list[str]) -> list[str]:
    result: list[str] = []
    for pat in patterns:
        result.extend(str(p) for p in sorted(project.glob(pat)))
    return result


# ── Stage implementations ──


def _project_setup_hash(project: Path, config: dict, vten_sv_dir: Path) -> str:
    """Stage 1 cache hash: config + vten_sv + RTL + IP sources."""
    h = hashlib.sha256()
    # config sections
    h.update(json.dumps(config.get("rtl", {}), sort_keys=True).encode())
    h.update(json.dumps(config.get("ip", {}), sort_keys=True).encode())
    h.update(config.get("backend", {}).get("xsim", {}).get("part", "").encode())
    # vten_sv sources
    for p in sorted(vten_sv_dir.glob("*.sv")) + sorted(vten_sv_dir.glob("*.svh")):
        h.update(p.read_bytes())
    # user RTL sources
    for glob_pat in config.get("rtl", {}).get("sources", []):
        for p in sorted(project.glob(glob_pat)):
            h.update(p.read_bytes())
    # IP sources
    for glob_pat in config.get("ip", {}).get("sources", []):
        for p in sorted(project.glob(glob_pat)):
            h.update(p.read_bytes())
    return h.hexdigest()


def _stage_project_setup(
    vivado_path: str,
    project: Path,
    config: dict,
    vten_sv_dir: Path,
    cache: dict,
    force: bool,
) -> None:
    print("[Stage 1] project_setup")
    current = _project_setup_hash(project, config, vten_sv_dir)

    if not force and cache_valid(cache, "project_setup", current):
        print("  cached, skip")
        return

    # Expand RTL globs in Python (Tcl glob doesn't support **)
    rtl_patterns = config.get("rtl", {}).get("sources", [])
    rtl_files = _expand_globs(project, rtl_patterns)

    # Render project_setup.tcl.j2
    tcl = _render_template("project_setup.tcl.j2", {
        "rtl_files": rtl_files,
        "include_dirs": config.get("rtl", {}).get("include_dirs", []),
        "ip_sources": _expand_globs(
            project, config.get("ip", {}).get("sources", []),
        ),
    })
    tcl_path = project / "build" / "project_setup.tcl"
    tcl_path.parent.mkdir(parents=True, exist_ok=True)
    tcl_path.write_text(tcl)

    part = config.get("backend", {}).get("xsim", {}).get("part", "")
    proj_dir = project / "build" / "vivado_proj"
    proj_dir.mkdir(parents=True, exist_ok=True)

    _run_vivado(vivado_path, tcl_path, proj_dir, part, vten_sv_dir, project)

    update_cache(cache, "project_setup", current)
    print("  done")


def _stage_dpi_c(
    project: Path,
    vten_sv_dir: Path,
    vivado_path: str,
    cache: dict,
    force: bool,
) -> None:
    print("[Stage 2] dpi_c")
    src_c = vten_sv_dir / "vten_shm_bridge.c"
    src_h = vten_sv_dir / "vten_shm_bridge.h"
    current = dir_hash([p for p in [src_c, src_h] if p.exists()])

    if not force and cache_valid(cache, "dpi_c", current):
        print("  cached, skip")
        return

    so_path = project / "build" / "lib" / "libvten_shm.so"
    so_path.parent.mkdir(parents=True, exist_ok=True)

    # Include paths: xsim's real svdpi.h, then vten_sv/ as fallback
    include_args: list[str] = []
    if vivado_path:
        xsim_include = Path(vivado_path) / "data" / "xsim" / "include"
        if xsim_include.exists():
            include_args += ["-I", str(xsim_include)]
    include_args += ["-I", str(vten_sv_dir)]

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
        raise BuildError(f"gcc failed:\n{result.stderr}")

    update_cache(cache, "dpi_c", current)
    print("  done")


def _stage_codegen(kernel_dir: Path, config: dict) -> None:
    print(f"[Stage 3] codegen: {kernel_dir.name}")
    spec_path = kernel_dir / "kernel_spec.yaml"
    spec = parse_kernel_spec(spec_path)
    spec = _expand_split_interfaces(spec)
    bfm_configs = _derive_bfm_configs(spec)

    gen = SVGenerator(
        kernel_spec=spec,
        bfm_configs=bfm_configs,
        project_config=config,
    )

    output = kernel_dir / "build" / "generated"
    output.mkdir(parents=True, exist_ok=True)
    gen.generate(str(output), num_commands=256)
    print("  done")


def _stage_compile_order(
    vivado_path: str,
    xpr_path: Path,
    kernel_dir: Path,
) -> None:
    print(f"[Stage 4] compile_order: {kernel_dir.name}")
    tb_top = kernel_dir / "build" / "generated" / "tb_top.sv"
    prj_out = kernel_dir / "build" / "compile.prj"

    if not tb_top.exists():
        raise BuildError(
            f"tb_top.sv not found: {tb_top}. Run codegen first."
        )
    if not xpr_path.exists():
        raise BuildError(
            f"Vivado project not found: {xpr_path}. Run project_setup first."
        )

    vten_root = Path(__file__).resolve().parent.parent.parent
    tcl = vten_root / "templates" / "resolve_order.tcl"

    _run_vivado(vivado_path, tcl, xpr_path, tb_top, prj_out)
    print("  done")


def _stage_compile(
    vivado_path: str,
    project: Path,
    kernel_dir: Path,
    vten_sv_dir: Path,
) -> None:
    print(f"[Stage 5] compile: {kernel_dir.name}")
    prj = kernel_dir / "build" / "compile.prj"
    dpi_lib = project / "build" / "lib" / "libvten_shm"

    if not prj.exists():
        raise BuildError(
            f"compile.prj not found: {prj}. Run compile_order first."
        )

    build_dir = kernel_dir / "build"

    # xvlog — include vten_sv dir for .svh files
    result = subprocess.run(
        [
            f"{vivado_path}/bin/xvlog", "--sv",
            "--include", str(vten_sv_dir),
            "--prj", str(prj),
        ],
        capture_output=True,
        text=True,
        cwd=str(build_dir),
    )
    if result.returncode != 0:
        raise BuildError(f"xvlog failed:\n{result.stderr[-500:]}")

    # Determine library name from compile.prj (Vivado uses xil_defaultlib)
    elab_top = "tb_top"
    prj_text = prj.read_text()
    for line in prj_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2].endswith("/tb_top.sv"):
            lib = parts[1]
            if lib != "work":
                elab_top = f"{lib}.tb_top"
            break

    # Collect library names from compile.prj for -L flags
    prj_libs = set()
    for line in prj_text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            prj_libs.add(parts[1])
    lib_args: list[str] = []
    for lib in sorted(prj_libs):
        if lib != "work":
            lib_args += ["-L", lib]

    # xelab
    result = subprocess.run(
        [
            f"{vivado_path}/bin/xelab", elab_top,
            *lib_args,
            "--sv_lib", dpi_lib.name,
            "--sv_root", str(dpi_lib.parent),
            "--timescale", "1ns/1ps",
            "--debug", "typical",
            "--snapshot", "tb_top",
            "--relax",
        ],
        capture_output=True,
        text=True,
        cwd=str(build_dir),
    )
    if result.returncode != 0:
        detail = result.stdout[-500:] if result.stdout else result.stderr[-500:]
        raise BuildError(f"xelab failed:\n{detail}")

    print("  done")


# ── Main orchestrator ──


def build_project(
    project_dir: str = ".",
    kernel_name: str | None = None,
    stage: str | None = None,
    upto: str | None = None,
    force: bool = False,
    skip_compile: bool = False,
    config_overrides: dict | None = None,
) -> None:
    """Build project: 5-stage pipeline with caching."""
    project = Path(project_dir).resolve()
    config = load_project_config(project)
    if config_overrides:
        config.setdefault("parameters", {}).update(config_overrides)

    vten_root = Path(__file__).resolve().parent.parent.parent
    vten_sv_dir = vten_root / "vten_sv"
    vivado_path = config.get("backend", {}).get("xsim", {}).get("vivado_path", "")

    # Determine target stages
    all_stages = ["project_setup", "dpi_c", "codegen", "compile_order", "compile"]
    target_stages = _resolve_stages(all_stages, stage, upto, skip_compile)

    # Determine target kernels
    target_kernels = [kernel_name] if kernel_name else discover_kernels(project)
    if not target_kernels:
        raise BuildError("No kernels found. Run: vten init --kernel <name>")

    cache = load_cache(project / "build" / ".cache.json")

    # === Project-level stages (once) ===

    if "project_setup" in target_stages:
        _stage_project_setup(vivado_path, project, config, vten_sv_dir, cache, force)

    if "dpi_c" in target_stages:
        _stage_dpi_c(project, vten_sv_dir, vivado_path, cache, force)

    # === Per-kernel stages ===

    xpr_path = project / "build" / "vivado_proj" / "vten_sim.xpr"

    for kname in target_kernels:
        kernel_dir = project / "kernels" / kname
        print(f"\n=== Kernel: {kname} ===")

        if "codegen" in target_stages:
            _stage_codegen(kernel_dir, config)

        if "compile_order" in target_stages:
            _stage_compile_order(vivado_path, xpr_path, kernel_dir)

        if "compile" in target_stages:
            _stage_compile(vivado_path, project, kernel_dir, vten_sv_dir)

    save_cache(project / "build" / ".cache.json", cache)
    print("\nBuild complete.")
