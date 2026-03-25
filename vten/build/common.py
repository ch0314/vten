"""Shared build utilities — cache, discovery, helpers.

Extracted from vten/cli/build.py for reuse across build pipelines.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import jinja2

from vten.errors import BuildError


# ── Kernel discovery ──


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


# ── Stage resolution ──


def resolve_stages(
    all_stages: list[str],
    stage: str | None,
    upto: str | None,
    skip_compile: bool,
) -> list[str]:
    """Resolve which stages to run based on CLI flags."""
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


def expand_globs(project: Path, patterns: list[str]) -> list[str]:
    """Expand glob patterns relative to project root."""
    result: list[str] = []
    for pat in patterns:
        result.extend(str(p) for p in sorted(project.glob(pat)))
    return result


def render_template(name: str, context: dict) -> str:
    """Render a Jinja2 template from the templates/ directory."""
    vten_root = Path(__file__).resolve().parent.parent.parent
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(vten_root / "templates")),
    )
    return env.get_template(name).render(context)


def run_vivado(vivado_path: str, tcl_script: Path, *args: str | Path) -> None:
    """Run Vivado in batch mode with a TCL script."""
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
