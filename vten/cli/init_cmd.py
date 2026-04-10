"""vten init: project skeleton creation.

Spec reference: 06_codegen_and_cli.md §4.1, 08_backend_abstraction.md §9.3
"""

from __future__ import annotations

from pathlib import Path

from vten.errors import VTenError


# ── Backend-specific TOML templates (08_backend_abstraction.md §9.3) ──

_BACKEND_TOML_TEMPLATES: dict[str, str] = {
    "xsim": """\
[backend.xsim]
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 0
submit_timeout_s = 300
""",
    "xrt": """\
[backend.xrt]
xclbin_path = "build/kernel.xclbin"
device_index = 0
kernel_name = ""
poll_timeout_ms = 30000
""",
    "verilator": """\
[backend.verilator]
verilator_path = ""
threads = 4
trace = false
opt_level = 3
""",
}

_BACKEND_DIRS: dict[str, list[str]] = {
    "xsim":      ["build/vivado_proj", "build/lib", "ip"],
    "verilator": ["build/lib"],
    "xrt":       ["build", "ip"],
}

_COMMON_DIRS = ["rtl", "kernels", "results"]


def _make_toml_content(name: str, backend: str) -> str:
    """Generate vten.toml content for a specific backend."""
    header = f"""\
[project]
name = "{name}"
version = "0.1.0"
default_backend = "{backend}"

[tools]
vivado_path = "/tools/Xilinx/Vivado/2023.2"

[parameters]

"""
    backend_section = _BACKEND_TOML_TEMPLATES.get(backend, _BACKEND_TOML_TEMPLATES["xsim"])
    footer = """
[rtl]
sources = ["rtl/**/*.sv", "rtl/**/*.v"]
include_dirs = ["rtl/include"]

[test]
default_seed = 42
waveform = false
waveform_on_fail = true
"""
    return header + backend_section + footer


def init_project(
    project_dir: str,
    kernel_name: str | None = None,
    backend: str | None = None,
    add_backend: str | None = None,
) -> None:
    """Create a new vten project skeleton, or add a kernel/backend to existing project."""
    root = Path(project_dir)

    if kernel_name:
        _init_kernel(root, kernel_name)
        return

    if add_backend:
        _add_backend(root, add_backend)
        return

    # Full project initialization — works on both new and existing directories
    target_backend = backend or "xsim"

    root.mkdir(parents=True, exist_ok=True)

    # Create common + backend-specific directories (skip existing)
    dirs = list(_COMMON_DIRS)
    dirs.extend(_BACKEND_DIRS.get(target_backend, []))
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)

    # vten.toml — only create if missing
    toml_path = root / "vten.toml"
    if not toml_path.exists():
        toml_path.write_text(_make_toml_content(root.name, target_backend))


def _add_backend(root: Path, backend_name: str) -> None:
    """Add a backend section to an existing project's vten.toml."""
    toml_path = root / "vten.toml"
    if not toml_path.exists():
        raise VTenError(f"vten.toml not found in {root}")

    content = toml_path.read_text()
    section_header = f"[backend.{backend_name}]"
    if section_header in content:
        raise VTenError(f"{section_header} section already exists in vten.toml")

    template = _BACKEND_TOML_TEMPLATES.get(backend_name)
    if template is None:
        raise VTenError(f"Unknown backend: {backend_name}")

    # Append backend section
    if not content.endswith("\n"):
        content += "\n"
    content += "\n" + template
    toml_path.write_text(content)

    # Create backend-specific directories
    for d in _BACKEND_DIRS.get(backend_name, []):
        (root / d).mkdir(parents=True, exist_ok=True)


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
