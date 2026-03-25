"""vten init: project skeleton creation.

Spec reference: 06_codegen_and_cli.md §4.1
"""

from __future__ import annotations

from pathlib import Path


_VTEN_TOML_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"

[parameters]

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0
submit_timeout_s = 300

[rtl]
sources = ["rtl/**/*.sv", "rtl/**/*.v"]
include_dirs = ["rtl/include"]

[ip]
sources = ["ip/**/*.xci"]

[test]
default_seed = 42
waveform = false
waveform_on_fail = true
"""


def init_project(project_dir: str, kernel_name: str | None = None) -> None:
    """Create a new vten project skeleton, or add a kernel to existing project."""
    root = Path(project_dir)

    if kernel_name:
        # Add kernel directory to existing project
        _init_kernel(root, kernel_name)
        return

    # Full project initialization
    if root.exists():
        raise FileExistsError(f"Directory already exists: {project_dir}")

    root.mkdir(parents=True)

    dirs = [
        "rtl",
        "ip",
        "kernels",
        "build/vivado_proj",
        "build/lib",
        "results",
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)

    # vten.toml
    toml_path = root / "vten.toml"
    if not toml_path.exists():
        toml_path.write_text(_VTEN_TOML_TEMPLATE.format(name=root.name))


def _init_kernel(root: Path, kernel_name: str) -> None:
    """Add a kernel subdirectory with skeleton files."""
    kdir = root / "kernels" / kernel_name
    dirs = [
        kdir,
        kdir / "tests",
        kdir / "build" / "generated",
        kdir / "build" / "shm",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # kernel_spec.yaml skeleton
    spec = kdir / "kernel_spec.yaml"
    if not spec.exists():
        spec.write_text(
            f"kernel: {kernel_name}\n"
            f"rtl_top: rtl/TODO.sv\n\n"
            f"interfaces: {{}}\n"
        )

    # <name>_kernel.py skeleton
    py = kdir / f"{kernel_name}_kernel.py"
    if not py.exists():
        py.write_text(
            f"from vten.kernel import Kernel, Tensor\n\n\n"
            f"class {kernel_name.title()}Kernel(Kernel):\n"
            f'    spec = "kernels/{kernel_name}/kernel_spec.yaml"\n'
        )

    # tests/test_<name>.py skeleton
    test = kdir / "tests" / f"test_{kernel_name}.py"
    if not test.exists():
        test.write_text(f"# TODO: implement test scenario\n")
