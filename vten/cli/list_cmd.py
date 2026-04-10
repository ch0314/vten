"""vten list — list test scenarios for a kernel."""

from __future__ import annotations

import sys
from pathlib import Path

from vten.cli.config import load_project_config
from vten.cli.discovery import discover_all_tests
from vten.errors import VTenError


def list_tests(project_dir: str, kernel_name: str) -> None:
    project = Path(project_dir).resolve()
    config = load_project_config(project)

    kernels_root = project / config.get("kernels_dir", "kernels")
    kernel_dir = kernels_root / kernel_name
    tests_dir = kernel_dir / "tests"

    if not tests_dir.exists():
        raise VTenError(f"tests directory not found: {tests_dir}")

    # Add kernels base to sys.path for shared imports (model_configs, etc.)
    kernels_base = str(kernel_dir.parent)
    if kernels_base not in sys.path:
        sys.path.insert(0, kernels_base)

    scenarios = discover_all_tests(tests_dir)
    if not scenarios:
        print(f"No test scenarios found for kernel '{kernel_name}'.")
        return

    print(f"\n  {kernel_name}: {len(scenarios)} test(s)\n")
    for name, instance in scenarios:
        n_configs = len(instance.configs) if hasattr(instance, "configs") else 0
        doc = (instance.__class__.__doc__ or "").strip().split("\n")[0]
        print(f"  {name:<30s}  ({n_configs} config)  {doc}")
    print()
