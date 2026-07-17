#!/usr/bin/env python3
"""Verification-LOC comparison: hand-written Cocotb testbench vs vTen DSL.

Counting rule (applied identically to both sides):
- count non-blank, non-comment source lines;
- Python module/class/function docstrings are excluded (documentation,
  not verification logic); YAML comments/blank lines likewise excluded;
- benchmark instrumentation is excluded on both sides: the Cocotb count
  covers only the functional testbench (``cocotb_tb/test_passthrough.py``),
  not the timing wrapper; the vTen count covers the user-authored
  verification assets (kernel DSL class + interface binding spec +
  CLI test scenario), none of which contain instrumentation.

Run:  python loc_count.py
"""

from __future__ import annotations

import ast
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
KDIR = REPO_ROOT / "examples" / "passthrough" / "kernels" / "passthrough"

COCOTB_FILES = [BENCH_DIR / "cocotb_tb" / "test_passthrough.py"]
VTEN_FILES = [
    KDIR / "passthrough_kernel.py",
    KDIR / "kernel_spec.yaml",
]
# The CLI additionally needs the minimal TestScenario declaration. The
# scenario file also holds an unrelated probe-mode scenario, so only the
# TestPassthrough class (plus its import) is counted.
VTEN_SCENARIO = KDIR / "tests" / "test_passthrough.py"


def docstring_lines(source: str) -> set[int]:
    """Line numbers (1-based) occupied by module/class/def docstrings."""
    lines: set[int] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return lines


def count_file(path: Path) -> int:
    source = path.read_text()
    skip = docstring_lines(source) if path.suffix == ".py" else set()
    count = 0
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or i in skip:
            continue
        count += 1
    return count


def count_scenario(path: Path, class_name: str) -> int:
    """Count LOC of one scenario class (+1 for its import), docstrings excluded."""
    source = path.read_text()
    skip = docstring_lines(source)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            span = range(node.lineno, node.end_lineno + 1)
            body = [
                line for i, line in enumerate(source.splitlines(), start=1)
                if i in span and i not in skip and line.strip()
                and not line.strip().startswith("#")
            ]
            return len(body) + 1  # + the TestScenario import
    raise SystemExit(f"{class_name} not found in {path}")


def report(label: str, files: list[Path]) -> int:
    total = 0
    print(f"{label}:")
    for f in files:
        n = count_file(f)
        total += n
        print(f"  {n:4d}  {f.relative_to(REPO_ROOT)}")
    return total


def main() -> None:
    cocotb_total = report("Cocotb (hand-written testbench)", COCOTB_FILES)
    print(f"  {cocotb_total:4d}  TOTAL")
    print()
    vten_total = report("vTen (kernel DSL + binding spec)", VTEN_FILES)
    scenario = count_scenario(VTEN_SCENARIO, "TestPassthrough")
    vten_total += scenario
    print(f"  {scenario:4d}  {VTEN_SCENARIO.relative_to(REPO_ROOT)}"
          " (TestPassthrough class + import only)")
    print(f"  {vten_total:4d}  TOTAL")
    print()
    reduction = 100.0 * (1 - vten_total / cocotb_total)
    print(f"reduction: {reduction:.1f}% ({vten_total} vs {cocotb_total} lines)")


if __name__ == "__main__":
    main()
