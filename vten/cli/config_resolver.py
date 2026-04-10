"""Config resolver — parse --config CLI arguments into config dicts.

Supports three formats:
  1. JSON string:       '{"in_ch": 128, "out_ch": 64}'
  2. Module reference:  model_configs:LAYERS[0]     → dict
                        model_configs:LAYERS         → list[dict]
                        model_configs:LAYERS[0:3]    → list[dict] (slice)
  3. Key=Value pairs:   in_ch=128 out_ch=64

Mixed format (module ref + K=V overrides):
  model_configs:UNET_3D[0] in_depth=4 in_height=4
  → resolves module ref, then merges K=V into each config
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from vten.errors import VTenError


def resolve_config(
    args: list[str],
    *,
    kernels_base: Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Resolve --config CLI arguments into config(s).

    Args:
        args: List of config arguments from argparse.
        kernels_base: Base directory for module imports (added to sys.path).

    Returns:
        A single config dict, or a list of config dicts (for multi-config).

    Raises:
        VTenError: If the config format is invalid or module not found.
    """
    if not args:
        return {}

    # Single arg that looks like JSON → parse as JSON
    if len(args) == 1 and args[0].strip().startswith("{"):
        return _parse_json(args[0])

    # Split into module ref (first arg with ':') and K=V overrides
    module_ref = None
    kv_args = []
    for arg in args:
        if module_ref is None and ":" in arg and "=" not in arg:
            module_ref = arg
        else:
            kv_args.append(arg)

    # Module ref + optional K=V overrides
    if module_ref is not None:
        value = _resolve_module_ref(module_ref, kernels_base=kernels_base)
        overrides = _parse_kv_pairs(kv_args) if kv_args else {}

        if isinstance(value, list):
            if overrides:
                return [{**cfg, **overrides} for cfg in value]
            return value
        else:
            if overrides:
                return {**value, **overrides}
            return value

    # Pure K=V pairs
    return _parse_kv_pairs(args)


def _parse_json(s: str) -> dict[str, Any]:
    """Parse a JSON string into a config dict."""
    try:
        result = json.loads(s)
    except json.JSONDecodeError as e:
        raise VTenError(f"invalid JSON config: {e}") from e
    if not isinstance(result, dict):
        raise VTenError(f"JSON config must be a dict, got {type(result).__name__}")
    return result


def _resolve_module_ref(
    ref: str,
    *,
    kernels_base: Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Resolve a module:VAR reference to a config dict or list.

    Formats:
        ``module:VAR``        → dict or list (as-is)
        ``module:VAR[3]``     → single element (dict)
        ``module:VAR[1:4]``   → slice (list[dict])
    """
    match = re.match(r"^([^:]+):(\w+)(?:\[(\d+)?(?::(\d+))?\])?$", ref)
    if not match:
        raise VTenError(
            f"invalid config reference: {ref!r}\n"
            f"Expected: module:VAR, module:VAR[idx], or module:VAR[start:end]"
        )

    mod_name, var_name, idx_str, end_str = match.groups()

    # Add kernels base to sys.path for module discovery
    if kernels_base is not None:
        base = str(kernels_base)
        if base not in sys.path:
            sys.path.insert(0, base)

    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise VTenError(f"config module not found: {mod_name!r}") from e

    if not hasattr(module, var_name):
        raise VTenError(
            f"config module {mod_name!r} has no variable {var_name!r}"
        )

    value = getattr(module, var_name)

    # Slice: VAR[start:end]
    if end_str is not None:
        if not isinstance(value, (list, tuple)):
            raise VTenError(
                f"{mod_name}:{var_name} is not sliceable "
                f"(got {type(value).__name__})"
            )
        start = int(idx_str) if idx_str is not None else 0
        end = int(end_str)
        return list(value[start:end])

    # Index: VAR[idx]
    if idx_str is not None:
        if not isinstance(value, (list, tuple)):
            raise VTenError(
                f"{mod_name}:{var_name} is not indexable "
                f"(got {type(value).__name__})"
            )
        idx = int(idx_str)
        if idx >= len(value):
            raise VTenError(
                f"{mod_name}:{var_name}[{idx}] out of range "
                f"(length {len(value)})"
            )
        value = value[idx]

    # No index — return as-is (dict or list)
    if isinstance(value, (dict, list)):
        return value

    raise VTenError(
        f"config value must be a dict or list[dict], "
        f"got {type(value).__name__} from {ref}"
    )


def _parse_kv_pairs(args: list[str]) -> dict[str, Any]:
    """Parse K=V pairs into a config dict with type coercion."""
    result: dict[str, Any] = {}
    for item in args:
        if "=" not in item:
            raise VTenError(
                f"invalid config argument: {item!r}\n"
                f"Expected K=V format, JSON string, or module:VAR reference"
            )
        k, v = item.split("=", 1)
        result[k] = _coerce_value(v)
    return result


def _coerce_value(v: str) -> Any:
    """Coerce a string value to int, float, bool, or keep as string."""
    if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
        return int(v)
    try:
        return float(v)
    except ValueError:
        pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v
