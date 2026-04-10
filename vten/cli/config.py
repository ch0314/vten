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


def resolve_tool_path(config: dict, tool: str, backend: str | None = None) -> str:
    """Resolve a tool path from config with ``[tools]`` fallback.

    Lookup order:
      1. ``[backend.<backend>].<tool>``  (if *backend* given)
      2. ``[tools].<tool>``              (unified section)
      3. ``""``                          (not configured)

    This lets users write ``vivado_path`` once in ``[tools]`` while still
    allowing per-backend overrides.

    Args:
        config: Project configuration dict (from vten.toml).
        tool: Tool key name (e.g. ``"vivado_path"``).
        backend: Backend name for per-backend lookup (e.g. ``"xsim"``).

    Returns:
        Resolved path string, or ``""`` if not configured.
    """
    # 1. Per-backend override
    if backend:
        val = config.get("backend", {}).get(backend, {}).get(tool, "")
        if val:
            return val

    # 2. Unified [tools] section
    # Support both "vivado_path" (full key) and "vivado" (short key)
    tools = config.get("tools", {})
    val = tools.get(tool, "")
    if val:
        return val
    # Try short form: "vivado_path" → "vivado"
    short = tool.removesuffix("_path") if tool.endswith("_path") else ""
    if short:
        val = tools.get(short, "")
        if val:
            return val

    return ""
