"""vten init: project skeleton creation.

Spec reference: 06_codegen_and_cli.md §4.1
"""

from __future__ import annotations

from pathlib import Path


def init_project(project_dir: str) -> None:
    """Create a new vten project skeleton."""
    path = Path(project_dir)

    if path.exists():
        raise FileExistsError(f"Directory already exists: {project_dir}")

    path.mkdir(parents=True)

    for subdir in ["rtl", "specs", "kernels", "tests", "build"]:
        (path / subdir).mkdir()

    project_name = path.name
    (path / "vten.toml").write_text(
        f'[project]\nname = "{project_name}"\nversion = "0.1.0"\n'
    )

    (path / "kernels" / "example_kernel.py").write_text(
        '"""Example vten kernel."""\n\n'
        "from vten.kernel import Kernel, Tensor\n"
    )

    (path / "tests" / "test_example.py").write_text(
        '"""Example vten test."""\n\n'
        "from vten.cli.run import TestScenario\n\n\n"
        "class TestExample(TestScenario):\n"
        '    kernel = "example"\n\n'
        "    def run(self, ctx, cfg):\n"
        "        pass\n"
    )
