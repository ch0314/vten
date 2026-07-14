#!/usr/bin/env python3
"""scale_add — Level-3 InferenceModel demo (whole-network orchestration + graph capture).

This is the reference example for vTen's **InferenceModel** — a *hybrid
imperative + graph-capturing* container that lets you compose several verified
vTen kernels into one whole model. You write the dataflow as plain Python in
``forward()``; each node runs **eagerly** through a shared ``InferenceSession``
(the same zero-copy device chaining as ``InferenceModule``), and the model
records the **dataflow graph as a side effect** of that execution. There is no
deferred/lazy graph build and no second execution path — *execution IS the
capture*. The captured graph is meant for a later verification / perf / memory
agent to hook into.

────────────────────────────────────────────────────────────────────────────────
The toy model (exercises NON-LINEAR capture: fan-out)
────────────────────────────────────────────────────────────────────────────────
    h = scale(x)              # out = clamp(x * scale_factor)
    a = offset(h, +1)         # out = clamp(h + 1)
    b = offset(h, +2)         # out = clamp(h + 2)

``h`` FANS OUT to two consumers (``a`` and ``b``). Because InferenceModel keys
its provenance map on ``id(tensor)`` and the *same* physical output object is
handed to both consumers, the fan-out falls out of the id-map automatically —
the captured graph shows one ``scale`` node feeding two ``offset`` nodes.

We build on two kernels from this example project, both int8 ``data_in ->
data_out`` with identical shape ``(N,)`` so they chain cleanly:
  * ``ScaleKernel``  (kernels/scale)  — out = in * scale_factor  (saturating)
  * ``OffsetKernel`` (kernels/offset) — out = in + offset_value  (saturating)

A clean 2-input kernel (needed to demo a skip-JOIN, e.g. ``join(a, concat_mem=x)``)
is NOT present in this project — scale/offset are both single-input. So this demo
shows the fan-out case only; the skip-join case is noted for a later slice. The
capture machinery already supports it: any extra keyword input (e.g.
``concat_mem=<upstream tensor>``) is recorded as an edge.

────────────────────────────────────────────────────────────────────────────────
How graph capture works (the id-map mechanism)
────────────────────────────────────────────────────────────────────────────────
InferenceModel keeps an EXTERNAL identity map ``id(tensor) -> producing node``
— it never mutates the core ``vten.Tensor`` to carry provenance. On each node
call: for every bound input tensor it looks up ``id(tensor)`` (hit = edge from
the producing node; miss = graph input such as the ``x`` arg or an uploaded
weight); after the kernel runs it registers ``id(output) -> this node``. Fan-out
and skips both fall out of this for free.

Run it (cpu backend — no build, no Vivado, no FPGA):
    python examples/scale_add/model_demo.py

This script is self-contained: it adds the repo root to ``sys.path`` (so
``import vten`` works when launched by path) and ``chdir``s into this example
directory (so each kernel's relative ``spec:`` path resolves). Every
InferenceModel / InferenceSession call mirrors the exact signatures in
``vten/inference.py`` — no invented APIs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

# ── Make this script runnable by path from anywhere ──
# 1) repo root on sys.path so `import vten` resolves (vten is used from source).
_THIS = Path(__file__).resolve()
_EXAMPLE_DIR = _THIS.parent                       # examples/scale_add
_REPO_ROOT = _EXAMPLE_DIR.parent.parent           # repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# 2) kernel dirs on sys.path so we can import the kernel classes.
for _k in ("scale", "offset"):
    _kd = str(_EXAMPLE_DIR / "kernels" / _k)
    if _kd not in sys.path:
        sys.path.insert(0, _kd)
# 3) chdir into the example project so each kernel's relative `spec:` resolves.
os.chdir(_EXAMPLE_DIR)

from scale_kernel import ScaleKernel   # noqa: E402  (kernels/scale)
from offset_kernel import OffsetKernel  # noqa: E402  (kernels/offset)

from vten import InferenceModel, InferenceSession  # noqa: E402

# Kernel facts (from *_kernel.py + kernel_spec.yaml):
#   input tensor  : "data_in"   (int8, shape (N,))
#   output tensor : "data_out"  (int8, shape (N,))
#   params        : ScaleKernel.scale_factor, OffsetKernel.offset_value, N
N = 32


class FanOutNet(InferenceModel):
    """h = scale(x); a = offset(h,+1); b = offset(h,+2)  —  h fans out to a, b."""

    def build(self) -> None:
        # Stages declared once. scale/offset input/output slots are data_in/data_out.
        self.scale = self.stage(
            ScaleKernel, scale_factor=2, N=N,
            input_name="data_in", output_name="data_out", name="scale",
        )
        self.off1 = self.stage(
            OffsetKernel, offset_value=1, N=N,
            input_name="data_in", output_name="data_out", name="off1",
        )
        self.off2 = self.stage(
            OffsetKernel, offset_value=2, N=N,
            input_name="data_in", output_name="data_out", name="off2",
        )

    def forward(self, x):
        h = self.scale(x)   # scale node; x is a graph input
        a = self.off1(h)    # off1 consumes h  → edge scale→off1
        b = self.off2(h)    # off2 consumes h  → edge scale→off2  (fan-out!)
        # Return both branch outputs so the caller can inspect them.
        return a, b


def torch_reference(x: torch.Tensor):
    """Hand-computed reference matching the kernels' saturating int8 math."""
    h = (x.to(torch.int16) * 2).clamp(-128, 127)
    a = (h + 1).clamp(-128, 127).to(torch.int8)
    b = (h + 2).clamp(-128, 127).to(torch.int8)
    return a, b


def main() -> None:
    print("Creating InferenceSession(backend='cpu')  — no build/FPGA required")
    session = InferenceSession("cpu", project_dir=".", log_level="WARNING")

    net = FanOutNet(session)

    # Deterministic int8 input of shape (N,).
    x = torch.arange(-16, 16, dtype=torch.int8)

    # net(x): build() once, reset graph, run forward() eagerly, capture graph.
    a_t, b_t = net(x)
    a, b = a_t.cpu(), b_t.cpu()  # Tensor -> torch.Tensor

    ra, rb = torch_reference(x)
    print("\n── outputs (first 8 of each) ──")
    print(f"  x        : {x.tolist()[:8]}")
    print(f"  a=off1(h): {a.tolist()[:8]}   (expected {ra.tolist()[:8]})")
    print(f"  b=off2(h): {b.tolist()[:8]}   (expected {rb.tolist()[:8]})")
    print(f"  a matches reference: {torch.equal(a, ra)}")
    print(f"  b matches reference: {torch.equal(b, rb)}")

    # ── The captured dataflow graph (the whole point of InferenceModel) ──
    graph = net.graph()
    print("\n── captured graph: nodes ──")
    for n in graph["nodes"]:
        srcs = ", ".join(f"{i['tensor_name']}<-{i['source']}" for i in n["inputs"])
        print(f"  {n['name']:>6}  [{n['kernel']}]  inputs: {srcs}")
    print("\n── captured graph: edges (producer -> consumer) ──")
    for e in graph["edges"]:
        print(f"  {e['src']:>6}  ->  {e['dst']:<6}  ({e['tensor_name']})")

    # Show the fan-out explicitly: how many consumers read 'scale'?
    fanout = [e["dst"] for e in graph["edges"] if e["src"] == "scale"]
    print(f"\n  fan-out: 'scale' output is consumed by {len(fanout)} nodes: {fanout}")

    print("\n── full graph (JSON) ──")
    print(json.dumps(graph, indent=2))

    session.cleanup()
    print("\nDone.")


if __name__ == "__main__":
    main()
