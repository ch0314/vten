"""Backend registry — factory + discovery for backends and build pipelines.

Spec reference: 08_backend_abstraction.md §9.4
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from vten.errors import VTenError

if TYPE_CHECKING:
    from vten.backend.base import Backend
    from vten.build.base import BuildPipeline


_BACKEND_MAP: dict[str, tuple[str, str]] = {
    "xsim":      ("vten.backend.xsim",      "XsimBackend"),
    "verilator": ("vten.backend.verilator",  "VerilatorBackend"),
    "xrt":       ("vten.backend.xrt",        "XrtBackend"),
}

_BUILD_MAP: dict[str, tuple[str, str]] = {
    "xsim":      ("vten.build.xsim_build",      "XsimBuildPipeline"),
    "verilator": ("vten.build.verilator_build",  "VerilatorBuildPipeline"),
    "xrt":       ("vten.build.xrt_build",        "XrtBuildPipeline"),
}


def get_backend(name: str, config: dict, **kwargs) -> Backend:
    """Create a backend instance by name (lazy import).

    Args:
        name: Backend name ("xsim", "verilator", "xrt").
        config: Project configuration dict (from vten.toml).
        **kwargs: Extra keyword arguments forwarded to the backend
            constructor (e.g. ``persistent=True`` for XrtBackend).

    Returns:
        Backend instance.

    Raises:
        VTenError: Unknown backend name.
    """
    if name not in _BACKEND_MAP:
        raise VTenError(
            f"Unknown backend: {name!r}. Available: {list(_BACKEND_MAP)}"
        )
    module_path, class_name = _BACKEND_MAP[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(config, **kwargs)


def get_build_pipeline(name: str, project: Path, config: dict) -> BuildPipeline:
    """Create a build pipeline instance by name (lazy import).

    Args:
        name: Backend name ("xsim", "verilator", "xrt").
        project: Project root directory.
        config: Project configuration dict (from vten.toml).

    Returns:
        BuildPipeline instance.

    Raises:
        ValueError: Unknown build pipeline name.
    """
    if name not in _BUILD_MAP:
        raise VTenError(
            f"Unknown build pipeline: {name!r}. Available: {list(_BUILD_MAP)}"
        )
    module_path, class_name = _BUILD_MAP[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(project, config)


def resolve_backend_name(config: dict, cli_backend: str | None = None) -> str:
    """Resolve backend name from CLI flag or project config.

    Priority: CLI --backend > vten.toml [project].default_backend > "xsim"

    Args:
        config: Project configuration dict.
        cli_backend: Backend name from CLI --backend flag.

    Returns:
        Resolved backend name string.
    """
    if cli_backend:
        return cli_backend
    return config.get("project", {}).get("default_backend", "xsim")


def available_backends() -> list[str]:
    """Return list of available backend names."""
    return list(_BACKEND_MAP.keys())
