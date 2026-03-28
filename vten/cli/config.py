"""vten.toml parsing and validation.

Spec reference: 06_codegen_and_cli.md §6
"""

from __future__ import annotations

from pathlib import Path

from vten.errors import VTenError

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


def load_project_config(project_dir: Path | str) -> dict:
    """Load and validate vten.toml from project directory."""
    toml_path = Path(project_dir) / "vten.toml"

    if not toml_path.exists():
        raise VTenError(f"vten.toml not found in {project_dir}")

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    if "project" not in config:
        raise VTenError("vten.toml missing required [project] section")

    return config
