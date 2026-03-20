"""vten report: test results reporting.

Spec reference: 06_codegen_and_cli.md §4.5
"""

from __future__ import annotations

import json
from pathlib import Path


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
        raise ValueError(f"Unsupported report format: {format}")

    if not results_dir.exists() or not any(results_dir.iterdir()):
        raise FileNotFoundError(f"No results found in {results_dir}")

    report_data: list[dict] = []

    for test_dir in sorted(results_dir.iterdir()):
        if not test_dir.is_dir():
            continue

        entry: dict = {"test_name": test_dir.name}

        summary_path = test_dir / "summary.json"
        if summary_path.exists():
            entry["summary"] = json.loads(summary_path.read_text())

        stats_path = test_dir / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            # Compute derived metrics for each command
            if isinstance(stats, dict) and "commands" in stats:
                for cmd in stats["commands"]:
                    issue = cmd.get("issue_cycle", 0)
                    commit = cmd.get("commit_cycle", 0)
                    cmd["latency_cycles"] = commit - issue
            entry["command_stats"] = stats

        report_data.append(entry)

    if not report_data:
        raise FileNotFoundError(f"No test results in {results_dir}")

    if format == "json":
        return json.dumps(report_data, indent=2)

    if format == "html":
        rows = ""
        for entry in report_data:
            summary = entry.get("summary", {})
            status = summary.get("status", "UNKNOWN")
            cycles = summary.get("total_cycles", 0)
            rows += f"<tr><td>{entry['test_name']}</td><td>{status}</td><td>{cycles}</td></tr>\n"
        return (
            "<table>\n<tr><th>Test</th><th>Status</th><th>Cycles</th></tr>\n"
            + rows
            + "</table>\n"
        )

    # terminal format
    lines: list[str] = []
    for entry in report_data:
        summary = entry.get("summary", {})
        status = summary.get("status", "UNKNOWN")
        cycles = summary.get("total_cycles", 0)
        lines.append(f"{entry['test_name']}: {status} ({cycles} cycles)")
        cmd_stats = entry.get("command_stats", {})
        if isinstance(cmd_stats, dict) and "commands" in cmd_stats:
            for cmd in cmd_stats["commands"]:
                cmd_id = cmd.get("cmd_id", "?")
                lines.append(f"  cmd {cmd_id}: {cmd.get('status', '?')}")
    return "\n".join(lines)
