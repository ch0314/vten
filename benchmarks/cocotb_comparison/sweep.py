#!/usr/bin/env python3
"""Sweep tensor size N for the vTen-vs-Cocotb benchmark.

For each N (geometric progression, default 1K -> x4 steps) runs both
frameworks R times as isolated subprocesses, records per-run phase timings
to ``results/results.csv``, and writes a median summary
(``results/summary.csv``) plus a log-log plot (``results/plot.png``).

Bounded by construction:
- every subprocess is killed after --run-timeout seconds (default 145);
- a framework stops escalating N once its median total wall time exceeds
  --escalate-limit seconds (default 45), since the next x4 step would risk
  the timeout. The stop point is recorded in the summary.

Usage:
    python sweep.py [--repeats 3] [--max-n 16777216]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"

FIELDS = [
    "framework", "n", "repeat", "beats", "cycles",
    "t_serialize_s", "t_load_s", "t_exec_s", "t_total_s", "status",
]


def run_one(framework: str, n: int, seed: int, timeout_s: int) -> dict:
    """Run one benchmark subprocess; returns a row dict (status ok/timeout/error)."""
    script = BENCH_DIR / f"run_{framework}.py"
    with tempfile.TemporaryDirectory(prefix="bench_sweep_") as tmp:
        out_json = Path(tmp) / "out.json"
        cmd = [sys.executable, str(script), "--n", str(n),
               "--seed", str(seed), "--json", str(out_json)]
        try:
            proc = subprocess.run(
                cmd, cwd=BENCH_DIR, timeout=timeout_s,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return {"framework": framework, "n": n, "status": "timeout"}
        finally:
            reap_stray_sims()
        if proc.returncode != 0 or not out_json.exists():
            sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
            return {"framework": framework, "n": n, "status": "error"}
        r = json.loads(out_json.read_text())
    return {
        "framework": framework,
        "n": n,
        "beats": r.get("beats"),
        "cycles": r.get("cycles"),
        "t_serialize_s": r.get("t_serialize_s"),
        "t_load_s": r.get("t_load_s", 0.0),
        "t_exec_s": r.get("t_exec_s"),
        "t_total_s": r.get("t_total_s"),
        "status": "ok",
    }


def reap_stray_sims() -> None:
    """Kill leaked simulator processes and stale vten SHM segments."""
    subprocess.run(["pkill", "-9", "-x", "Vtb_top"], capture_output=True)
    subprocess.run(
        ["pkill", "-9", "-f", "[c]ocotb_sim/passthrough"], capture_output=True
    )
    for p in Path("/dev/shm").glob("*vten_*"):
        try:
            p.unlink()
        except OSError:
            pass


def median_summary(rows: list[dict]) -> list[dict]:
    """Median across repeats per (framework, n), ok rows only."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["status"] == "ok":
            groups.setdefault((r["framework"], r["n"]), []).append(r)
    out = []
    for (fw, n), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        med = {k: statistics.median(float(r[k]) for r in g)
               for k in ("t_serialize_s", "t_load_s", "t_exec_s", "t_total_s")}
        out.append({
            "framework": fw, "n": n, "runs": len(g),
            "beats": g[0]["beats"], "cycles": g[0]["cycles"], **med,
        })
    return out


def write_plot(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
    styles = {"vten": ("tab:blue", "o"), "cocotb": ("tab:orange", "s")}
    for fw, (color, marker) in styles.items():
        pts = [(s["n"], s["t_exec_s"]) for s in summary if s["framework"] == fw]
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.loglog(xs, ys, color=color, marker=marker, label=f"{fw} $T_{{exec}}$")
        tot = [(s["n"], s["t_total_s"]) for s in summary if s["framework"] == fw]
        if tot:
            xs, ys = zip(*sorted(tot))
            ax.loglog(xs, ys, color=color, linestyle="--", alpha=0.5,
                      label=f"{fw} total")
    ax.set_xlabel("tensor size N (int8 elements)")
    ax.set_ylabel("wall-clock time [s]")
    ax.set_title("Passthrough DUT on Verilator: vTen vs Cocotb")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot written: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-start", type=int, default=1024)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max-n", type=int, default=16_777_216)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-timeout", type=int, default=145)
    ap.add_argument("--escalate-limit", type=float, default=45.0,
                    help="stop a framework once median total exceeds this [s]")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "results.csv"
    rows: list[dict] = []
    active = {"vten", "cocotb"}
    stop_reason: dict[str, str] = {}

    n = args.n_start
    while n <= args.max_n and active:
        for fw in ("vten", "cocotb"):
            if fw not in active:
                continue
            totals = []
            for rep in range(args.repeats):
                row = run_one(fw, n, args.seed + rep, args.run_timeout)
                row["repeat"] = rep
                rows.append(row)
                print(f"{fw:>6} N={n:>9} rep={rep} status={row['status']}"
                      + (f" t_exec={row.get('t_exec_s', 0):.4f}s"
                         f" t_total={row.get('t_total_s', 0):.2f}s"
                         if row["status"] == "ok" else ""),
                      flush=True)
                if row["status"] != "ok":
                    active.discard(fw)
                    stop_reason[fw] = f"{row['status']} at N={n}"
                    break
                totals.append(row["t_total_s"])
            if totals and statistics.median(totals) > args.escalate_limit:
                active.discard(fw)
                stop_reason[fw] = (
                    f"median total {statistics.median(totals):.1f}s > "
                    f"{args.escalate_limit}s at N={n}"
                )
        n *= args.factor

    for fw in ("vten", "cocotb"):
        stop_reason.setdefault(fw, f"reached --max-n {args.max_n}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k) for k in FIELDS} for r in rows)
    print(f"raw results: {csv_path}")

    summary = median_summary(rows)
    sum_path = RESULTS_DIR / "summary.csv"
    if summary:
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"summary: {sum_path}")
        write_plot(summary, RESULTS_DIR / "plot.png")

    print("\nstop points:")
    for fw, reason in stop_reason.items():
        print(f"  {fw}: {reason}")

    # Markdown table for the README
    print("\n| N | vTen T_exec [s] | Cocotb T_exec [s] | Cocotb/vTen |")
    print("|---:|---:|---:|---:|")
    by_key = {(s["framework"], s["n"]): s for s in summary}
    ns = sorted({s["n"] for s in summary})
    for n in ns:
        v = by_key.get(("vten", n))
        c = by_key.get(("cocotb", n))
        vcell = f"{v['t_exec_s']:.4f}" if v else "-"
        ccell = f"{c['t_exec_s']:.4f}" if c else "-"
        ratio = (f"{c['t_exec_s'] / v['t_exec_s']:.1f}x"
                 if v and c and v["t_exec_s"] else "-")
        print(f"| {n:,} | {vcell} | {ccell} | {ratio} |")


if __name__ == "__main__":
    main()
