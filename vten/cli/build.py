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
    skip_compile: bool = False,
    config_overrides: dict | None = None,
    run_vivado: bool = False,
) -> None:
    """Build project by delegating to the appropriate BuildPipeline."""
    project = Path(project_dir).resolve()
    config = load_project_config(project)

    backend_name = resolve_backend_name(config, cli_backend=backend)
    pipeline = get_build_pipeline(backend_name, project, config)
    pipeline.build(
        kernel_name=kernel_name,
        stage=stage,
        upto=upto,
        force=force,
        skip_compile=skip_compile,
        config_overrides=config_overrides,
    )

    # For XRT backend: print build instructions or execute vivado/v++
    if backend_name == "xrt":
        _xrt_post_build(project, kernel_name, run_vivado)


def _xrt_post_build(
    project: Path,
    kernel_name: str | None,
    run_vivado: bool,
) -> None:
    """After XRT artifact generation, print or execute build commands."""
    from vten.build.common import discover_kernels

    kernels = [kernel_name] if kernel_name else discover_kernels(project)
    if not kernels:
        return

    for kname in kernels:
        build_dir = project / "kernels" / kname / "build" / "xrt"
        build_script = None
        for f in build_dir.glob("build_*.sh"):
            build_script = f
            break

        if build_script is None:
            print(f"  No build script found in {build_dir}")
            continue

        if run_vivado:
            import subprocess

            print(f"\n=== Executing {build_script.name} ===")
            result = subprocess.run(
                ["bash", str(build_script)],
                cwd=str(build_dir),
            )
            if result.returncode != 0:
                print(f"  Build failed with exit code {result.returncode}")
        else:
            print(f"\nXRT build artifacts generated in: {build_dir}")
            print(f"To build xclbin, run:")
            print(f"  cd {build_dir} && bash {build_script.name}")
