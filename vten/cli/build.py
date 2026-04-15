"""vten build: delegating wrapper to backend-specific BuildPipeline.

Spec reference: 06_codegen_and_cli.md §4.3, 08_backend_abstraction.md §8
"""

from __future__ import annotations

from pathlib import Path

from vten.backend.registry import get_build_pipeline, resolve_backend_name
from vten.build.common import discover_kernels, find_kernel_spec  # noqa: F401
from vten.build.xsim_build import (  # noqa: F401
    _derive_bfm_configs,
    _expand_split_interfaces,
)
from vten.cli.config import load_project_config


def build_project(
    project_dir: str = ".",
    kernel_name: str | None = None,
    backend: str | None = None,
    stage: str | None = None,
    upto: str | None = None,
    force: bool = False,
    clean: bool = False,
    skip_compile: bool = False,
    config_overrides: dict | None = None,
) -> None:
    """Build project by delegating to the appropriate BuildPipeline."""
    project = Path(project_dir).resolve()
    config = load_project_config(project)

    # Apply --target override to [backend.xrt].target
    if config_overrides and "_xrt_target" in config_overrides:
        xrt_target = config_overrides.pop("_xrt_target")
        config.setdefault("backend", {}).setdefault("xrt", {})["target"] = xrt_target

    backend_name = resolve_backend_name(config, cli_backend=backend)
    pipeline = get_build_pipeline(backend_name, project, config)
    pipeline.build(
        kernel_name=kernel_name,
        stage=stage,
        upto=upto,
        force=force,
        clean=clean,
        skip_compile=skip_compile,
        config_overrides=config_overrides,
    )
