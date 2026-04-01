"""Test generation utilities for config sweeps.

Usage::

    from vten.testing import config_table, make_tests

    _CONFIGS = config_table(
        defaults={"in_ch": 32, "out_ch": 32, "in_width": 4},
        configs=[
            {"name": "default"},
            {"name": "big", "in_ch": 128, "out_ch": 128},
        ],
    )

    def _run(ctx, cfg):
        ...

    make_tests(
        configs=_CONFIGS,
        kernel="my_kernel",
        run_fn=_run,
        module_globals=globals(),
    )
"""

from __future__ import annotations

import re
from typing import Callable

from vten.cli.run import TestScenario


def config_table(
    defaults: dict,
    configs: list[dict],
) -> list[dict]:
    """Merge defaults with each config entry.

    Each config dict MUST have a ``"name"`` key.
    Returns list of merged dicts (defaults + per-config overrides).
    """
    result = []
    for cfg in configs:
        if "name" not in cfg:
            raise ValueError("Each config must have a 'name' key")
        merged = {**defaults, **cfg}
        result.append(merged)
    return result


def _to_pascal_case(snake: str) -> str:
    """Convert snake_case / kebab-case to PascalCase."""
    return "".join(word.capitalize() for word in re.split(r"[_\-]+", snake))


def make_tests(
    configs: list[dict],
    kernel: str,
    run_fn: Callable,
    module_globals: dict,
    *,
    class_prefix: str = "Test",
) -> list[type]:
    """Generate :class:`TestScenario` subclasses and register them.

    For each config dict, creates a class named
    ``{class_prefix}{PascalCase(name)}`` whose ``run()`` delegates to
    *run_fn(ctx, cfg)*.

    Args:
        configs: List of config dicts (each with ``"name"`` key).
        kernel: Kernel name for all generated scenarios.
        run_fn: ``Callable(ctx, cfg)`` implementing the test flow.
        module_globals: Pass ``globals()`` from the calling module.
        class_prefix: Prefix for generated class names (default ``"Test"``).

    Returns:
        List of generated :class:`TestScenario` subclasses.
    """
    classes: list[type] = []
    for cfg in configs:
        name = cfg["name"]
        cls_name = f"{class_prefix}{_to_pascal_case(name)}"

        # Capture cfg in closure via factory to avoid late-binding bug
        frozen_cfg = dict(cfg)

        def _make_run(captured_cfg: dict) -> Callable:
            def run(self, ctx, runtime_cfg):
                merged = {**runtime_cfg, **captured_cfg}
                run_fn(ctx, merged)
            return run

        cls = type(
            cls_name,
            (TestScenario,),
            {
                "kernel": kernel,
                "run": _make_run(frozen_cfg),
                "__module__": module_globals.get("__name__", __name__),
                "__qualname__": cls_name,
            },
        )

        module_globals[cls_name] = cls
        classes.append(cls)

    return classes
