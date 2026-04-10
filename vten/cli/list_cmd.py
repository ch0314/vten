"""vten list — list test scenarios and kernel parameters."""

from __future__ import annotations

import sys
from pathlib import Path

from vten.cli.config import load_project_config
from vten.cli.discovery import discover_all_tests
from vten.errors import VTenError


def list_tests(project_dir: str, kernel_name: str) -> None:
    """List test scenarios for a kernel."""
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


def list_params(project_dir: str, kernel_name: str) -> None:
    """Show kernel parameters from kernel_spec.yaml."""
    from vten.spec.parser import parse_kernel_spec

    project = Path(project_dir).resolve()
    config = load_project_config(project)

    kernels_root = project / config.get("kernels_dir", "kernels")
    kernel_dir = kernels_root / kernel_name
    spec_path = kernel_dir / "kernel_spec.yaml"

    if not spec_path.exists():
        raise VTenError(f"kernel_spec.yaml not found: {spec_path}")

    spec = parse_kernel_spec(spec_path)

    print(f"\n  {kernel_name} parameters\n")

    # Spec-level parameters (${PARAM} placeholders)
    if spec.parameters:
        print("  spec parameters:")
        for name, default in sorted(spec.parameters.items()):
            if default is not None:
                print(f"    {name:<20s} = {default}")
            else:
                print(f"    {name:<20s}   (required)")
        print()

    # Build parameters
    if spec.build_params:
        print("  build parameters:")
        for name, value in sorted(spec.build_params.items()):
            print(f"    {name:<20s} = {value}")
        print()

    # Interfaces and their tensors
    if spec.interfaces:
        print("  interfaces:")
        for iface_name, iface in spec.interfaces.items():
            tensor = iface.tensor or "-"
            proto = iface.protocol.value if iface.protocol else "?"
            print(f"    {iface_name:<20s}  {proto:<14s}  tensor={tensor}")
        print()

    # Registers per interface
    for iface_name in (spec.interfaces or {}):
        regs = spec.get_registers(iface_name)
        if regs:
            print(f"  registers ({iface_name}):")
            for reg in regs:
                fields_str = ""
                if isinstance(reg.fields, dict):
                    fields_str = ", ".join(
                        f"{k}[{v}]" for k, v in reg.fields.items()
                    )
                print(f"    {reg.name:<20s}  offset={reg.offset:#06x}  {fields_str}")
            print()
