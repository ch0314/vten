"""Test scenario discovery — find TestScenario subclasses in test directories."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from vten.cli.scenario import TestScenario


def _is_test_scenario(obj: object) -> bool:
    """Check if obj is a TestScenario subclass by MRO name.

    Using name-based check because TestScenario can be imported from
    multiple paths (vten.cli.run, vten.cli.scenario) with different
    class identity.
    """
    if not isinstance(obj, type):
        return False
    return any(c.__name__ == "TestScenario" for c in obj.__mro__[1:])


from vten.errors import VTenError

logger = logging.getLogger(__name__)


def discover_test(name: str, tests_dir: str | Path) -> TestScenario:
    """Find and instantiate a TestScenario by name.

    Matches by: exact class name, case-insensitive class name,
    snake_case name, or filename stem.
    """
    tests_path = Path(tests_dir)
    test_files = sorted(tests_path.glob("test_*.py"))

    candidates: list[tuple[str, type]] = []

    for test_file in test_files:
        mod_name = f"_vten_discover_{test_file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, test_file)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("failed to load %s: %s", test_file.name, exc)
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                _is_test_scenario(obj)
                and obj.__name__ != "TestScenario"
            ):
                candidates.append((test_file.stem, obj))

    name_lower = name.lower()

    # Priority tiers: exact match wins over fuzzy match
    exact_matches: list[type] = []
    fuzzy_matches: list[type] = []

    for file_stem, cls in candidates:
        cls_name = cls.__name__
        # Tier 1: Exact class name match
        if cls_name == name:
            exact_matches.append(cls)
        # Tier 2: Case-insensitive class name
        elif cls_name.lower() == name_lower:
            fuzzy_matches.append(cls)
        # Tier 2: snake_case / filename stem match
        elif file_stem == name or file_stem == f"test_{name}":
            fuzzy_matches.append(cls)
        elif file_stem.removeprefix("test_") == name:
            fuzzy_matches.append(cls)

    # Use exact matches if available, otherwise fall back to fuzzy
    matches = exact_matches if exact_matches else fuzzy_matches

    if not matches:
        raise VTenError(f"Not found: no test scenario matching '{name}'")

    if len(matches) > 1:
        # Deduplicate by class identity
        unique = list({id(c): c for c in matches}.values())
        if len(unique) > 1:
            names = [c.__name__ for c in unique]
            raise VTenError(f"Ambiguous: multiple matches for '{name}': {names}")
        matches = unique

    return matches[0]()


def discover_all_tests(tests_dir: str | Path) -> list[tuple[str, TestScenario]]:
    """Discover all TestScenario subclasses in tests_dir.

    Returns a list of (class_name, instance) pairs, sorted by class name.
    """
    tests_path = Path(tests_dir)
    test_files = sorted(tests_path.glob("test_*.py"))

    seen: dict[int, tuple[str, type]] = {}

    for test_file in test_files:
        mod_name = f"_vten_discover_{test_file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, test_file)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("failed to load %s: %s", test_file.name, exc)
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                _is_test_scenario(obj)
                and obj.__name__ != "TestScenario"
                and id(obj) not in seen
            ):
                seen[id(obj)] = (obj.__name__, obj)

    return [(name, cls()) for name, cls in sorted(seen.values(), key=lambda x: x[0])]
