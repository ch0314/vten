"""vten report: test results reporting.

Spec reference: 06_codegen_and_cli.md §4.5
"""

from __future__ import annotations

import json
from pathlib import Path

from vten.errors import VTenError


def _scan_results(results_dir: Path) -> list[dict]:
    """Scan results directory, supporting both flat and nested layouts.

    Supported layouts:
      - results/<test>/summary.json          (flat, legacy)
      - results/<kernel>/<test>/summary.json  (nested, from run_test)
    """
    report_data: list[dict] = []

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue

        # Check if this directory directly contains summary.json (flat layout)
        if (child / "summary.json").exists():
            report_data.append(_load_test_result(child, child.name))
        else:
            # Nested layout: child is kernel dir, grandchildren are test dirs
            for test_dir in sorted(child.iterdir()):
                if test_dir.is_dir() and (test_dir / "summary.json").exists():
                    display_name = f"{child.name}/{test_dir.name}"
                    report_data.append(_load_test_result(test_dir, display_name))

    return report_data


def _load_test_result(test_dir: Path, display_name: str) -> dict:
    """Load summary.json and stats.json from a test result directory."""
    entry: dict = {"test_name": display_name}

    summary_path = test_dir / "summary.json"
    if summary_path.exists():
        entry["summary"] = json.loads(summary_path.read_text())

    stats_path = test_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if isinstance(stats, dict) and "commands" in stats:
            # Ensure latency_cycles is present (backward compat)
            for cmd in stats["commands"]:
                if "latency_cycles" not in cmd:
                    issue = cmd.get("issue_cycle", 0)
                    commit = cmd.get("commit_cycle", 0)
                    cmd["latency_cycles"] = commit - issue
        entry["command_stats"] = stats

    return entry


def generate_report(project_dir: str, format: str = "terminal") -> str:
    """Generate a report from test results.

    Args:
        project_dir: Project root directory containing results/.
        format: Output format — "terminal", "html", or "json".

    Returns:
        Formatted report string.
    """
    results_dir = Path(project_dir) / "results"

    if format not in ("terminal", "html", "json"):
        raise VTenError(f"Unsupported report format: {format}")

    if not results_dir.exists() or not any(results_dir.iterdir()):
        raise VTenError(f"No results found in {results_dir}")

    report_data = _scan_results(results_dir)

    if not report_data:
        raise VTenError(f"No test results in {results_dir}")

    if format == "json":
        return json.dumps(report_data, indent=2)

    if format == "html":
        return _format_html(report_data)

    return _format_terminal(report_data)


# ── Terminal format ──


def _format_terminal(report_data: list[dict]) -> str:
    """Rich terminal output with command table and sub-kernel grouping."""
    lines: list[str] = []

    for entry in report_data:
        summary = entry.get("summary", {})
        status = summary.get("status", "UNKNOWN")
        cycles = summary.get("total_cycles", 0)
        kernel = summary.get("kernel", "")

        # Header
        header = entry["test_name"]
        if kernel:
            header = f"{kernel} / {entry['test_name']}"
        lines.append(f"== {header} :: {status} ({cycles} cycles) ==")

        # Command table
        cmd_stats = entry.get("command_stats", {})
        if isinstance(cmd_stats, dict) and "commands" in cmd_stats:
            commands = cmd_stats["commands"]
            if commands:
                lines.extend(_format_command_table(commands))

        # Verification results
        verifications = summary.get("verification_results", [])
        if verifications:
            lines.append("")
            lines.append("Verification:")
            for vr in verifications:
                tensor = vr.get("tensor", "?")
                passed = vr.get("passed", False)
                if passed:
                    lines.append(f"  {tensor}: PASS")
                else:
                    max_diff = vr.get("max_diff", 0.0)
                    lines.append(f"  {tensor}: FAIL (max_diff={max_diff:.6g})")

        # Summary line
        v_count = summary.get("verification_count", 0)
        v_passed = summary.get("verification_passed", 0)
        configs_run = summary.get("configs_run", 0)
        configs_passed = summary.get("configs_passed", 0)
        parts = [f"{configs_passed}/{configs_run} configs"]
        if v_count > 0:
            parts.append(f"{v_passed}/{v_count} verifications")
        lines.append(f"  [{', '.join(parts)}]")
        lines.append("")

    return "\n".join(lines)


def _format_command_table(commands: list[dict]) -> list[str]:
    """Format commands as an aligned table, grouped by sub_kernel if present."""
    # Check if any command has sub_kernel (CompositeKernel)
    has_sub_kernel = any(
        cmd.get("sub_kernel") and cmd["sub_kernel"] != "_self"
        for cmd in commands
    )

    if has_sub_kernel:
        return _format_grouped_table(commands)
    return _format_flat_table(commands)


def _format_flat_table(commands: list[dict]) -> list[str]:
    """Flat command table for unit kernels."""
    lines: list[str] = []

    # Build rows
    rows: list[tuple[str, ...]] = []
    for cmd in commands:
        rows.append(_cmd_to_row(cmd))

    # Header
    header = ("  #", "Op", "Interface", "Tensor", "Cycles", "Beats", "Status")
    lines.append("")
    lines.append(_align_row(header, rows))

    # Separator
    widths = _calc_widths(header, rows)
    lines.append("  " + "  ".join("-" * w for w in widths))

    # Data rows
    for row in rows:
        lines.append(_align_row_data(row, widths))

    return lines


def _format_grouped_table(commands: list[dict]) -> list[str]:
    """Command table grouped by sub_kernel for CompositeKernel."""
    from collections import OrderedDict

    lines: list[str] = []

    # Group commands by sub_kernel
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for cmd in commands:
        key = cmd.get("sub_kernel", "_self") or "_self"
        groups.setdefault(key, []).append(cmd)

    header = ("  #", "Op", "Interface", "Tensor", "Cycles", "Beats", "Status")

    # Collect all rows for width calculation
    all_rows: list[tuple[str, ...]] = []
    for cmds in groups.values():
        for cmd in cmds:
            all_rows.append(_cmd_to_row(cmd))

    widths = _calc_widths(header, all_rows)

    lines.append("")
    lines.append(_align_row(header, all_rows))
    lines.append("  " + "  ".join("-" * w for w in widths))

    row_idx = 0
    for group_name, cmds in groups.items():
        lines.append(f"  [{group_name}]")
        for cmd in cmds:
            lines.append(_align_row_data(all_rows[row_idx], widths))
            row_idx += 1

    return lines


def _cmd_to_row(cmd: dict) -> tuple[str, ...]:
    """Convert a command dict to a display row tuple."""
    cmd_id = str(cmd.get("cmd_id", "?"))
    op = cmd.get("op", cmd.get("op_name", ""))
    protocol = cmd.get("protocol", "")

    iface = cmd.get("interface", cmd.get("interface_name", ""))
    if iface and protocol:
        # Shorten protocol for display
        proto_short = {
            "axi4_stream": "AXI4S",
            "axi4": "AXI4",
            "axi4_lite": "AXI4L",
        }.get(protocol, protocol)
        iface_display = f"{iface}({proto_short})"
    elif iface:
        iface_display = iface
    else:
        iface_display = "-"

    tensor = cmd.get("tensor", cmd.get("tensor_name")) or "-"
    port = cmd.get("port", "")
    if port:
        tensor = f"{tensor}:{port}"

    latency = cmd.get("latency_cycles", 0)
    beats = cmd.get("total_beats", 0)

    status_name = cmd.get("status_name", "")
    if not status_name:
        # Fallback: resolve from int
        status_code = cmd.get("status", 0)
        status_map = {0: "PENDING", 1: "ISSUED", 2: "ACTIVE", 3: "COMMITTED", 4: "ERROR"}
        status_name = status_map.get(status_code, str(status_code))

    return (f"  {cmd_id}", op, iface_display, tensor, str(latency), str(beats), status_name)


def _calc_widths(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[int]:
    """Calculate column widths from header and all rows."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    return widths


def _align_row(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Format header row with proper alignment."""
    widths = _calc_widths(header, rows)
    return "  ".join(h.ljust(w) for h, w in zip(header, widths))


def _align_row_data(row: tuple[str, ...], widths: list[int]) -> str:
    """Format a data row with proper alignment."""
    return "  ".join(cell.ljust(w) for cell, w in zip(row, widths))


# ── HTML format ──


def _format_html(report_data: list[dict]) -> str:
    """HTML report with enriched command data."""
    rows = ""
    for entry in report_data:
        summary = entry.get("summary", {})
        status = summary.get("status", "UNKNOWN")
        cycles = summary.get("total_cycles", 0)
        status_class = "pass" if status == "PASS" else "fail"
        rows += (
            f'<tr class="{status_class}">'
            f"<td>{entry['test_name']}</td>"
            f"<td>{status}</td>"
            f"<td>{cycles}</td>"
            f"</tr>\n"
        )
    return (
        "<table>\n"
        "<tr><th>Test</th><th>Status</th><th>Cycles</th></tr>\n"
        + rows
        + "</table>\n"
    )
