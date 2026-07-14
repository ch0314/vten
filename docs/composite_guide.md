# CompositeKernel Guide

How to compose multiple IPs into one verification unit with **CompositeKernel**.

A `CompositeKernel` wires several single kernels together into a pipeline,
verifies the whole thing end-to-end against an auto-chained golden reference,
and lets you drive every sub-kernel's control from one `run(ctx)`. If you
haven't written a single kernel yet, read [kernel_guide.md](kernel_guide.md)
first — a composite is built entirely out of ordinary `Kernel`s.

See also: [architecture.md](architecture.md) (composite flatten is Stage 0 of
the compile pipeline), [testing_guide.md](testing_guide.md) for running
composites, and [cli_reference.md](cli_reference.md).

---

## When to use it

Reach for a `CompositeKernel` when your DUT is really *several* RTL IPs already
wired together on-chip — a Vitis-style dataflow region, or an accelerator top
that instantiates a chain of sub-modules. Instead of one monolithic kernel with
a hand-written protocol, you:

- declare each IP as its own single `Kernel` (each with its own
  `kernel_spec.yaml`), and
- describe the internal wires between them with `>>`.

vTen then flattens the graph, exposes the free (unconnected) tensors as the
composite's own I/O, and derives the golden reference by chaining each
sub-kernel's `forward()`.

If your DUT is a single RTL module, you don't need a composite — a plain
`Kernel` is simpler.

---

## The `>>` connection model — scale_add walkthrough

`scale_add` composes two single kernels — `ScaleKernel` (multiply by a factor)
and `OffsetKernel` (add a value) — into `x → scale → offset → out`. The
`scale.data_out` and `offset.data_in` streams are already wired together *inside
the RTL*.

### `kernels/scale_add/scale_add_kernel.py`

```python
import sys
from pathlib import Path

import torch

from vten.kernel.composite import CompositeKernel
from vten.kernel.register import register

# Import sub-kernels (siblings under kernels/)
_kernel_base = str(Path(__file__).resolve().parent.parent)
if _kernel_base not in sys.path:
    sys.path.insert(0, _kernel_base)

from scale.scale_kernel import ScaleKernel
from offset.offset_kernel import OffsetKernel


class ScaleAddKernel(CompositeKernel):
    """Composite: scale(x factor) → offset(+ value) pipeline."""

    # Sub-kernels: plain instances
    scale = ScaleKernel()
    offset = OffsetKernel()

    # Register proxies for each sub-kernel's ctrl
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")

    # Internal connection: scale output → offset input
    connections = [scale.data_out >> offset.data_in]

    # Auto-exposed tensors (NOT in any connection):
    #   scale.data_in   → data_in   (composite input)
    #   offset.data_out → data_out  (composite output)

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        data = inputs.get("data_in", self.data_in.data)
        x = data.to(torch.int16) * self.scale_factor
        x = x.clamp(-128, 127)
        ov = self.offset_value
        if ov >= 128:
            ov = ov - 256
        x = x + ov
        return {"data_out": x.clamp(-128, 127).to(torch.int8)}

    def run(self, ctx) -> None:
        h_push = ctx.push_tensor(self.data_in)
        h_cfg = ctx.configure(self, dep=h_push)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_cfg)

        # Start both sub-kernels
        h_start_s = ctx.write_register(self.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_o = ctx.write_register(self.offset_ctrl, {"start": 1}, dep=h_cfg)

        # Poll both done flags; pull commits only after both finish
        h_poll_s = ctx.poll_register(self.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(self.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)
```

### How the wiring works

- **Sub-kernels are plain instances.** `scale = ScaleKernel()` declared in the
  class body registers `ScaleKernel` as a sub-kernel under the attribute name
  `scale`. The attribute name (`scale`, `offset`) is the sub-kernel's *ref name*
  used everywhere else.

- **`connections` uses `>>`.** `scale.data_out >> offset.data_in` reads as
  "scale's `data_out` feeds offset's `data_in`." Under the hood, accessing
  `scale.data_out` in the class body yields a `TensorRef`, and `>>` builds a
  `Connection` ([vten/kernel/composite.py](../vten/kernel/composite.py)). A
  connection describes a **pre-existing internal RTL wire** — vTen suppresses the
  BFMs on that boundary because the two IPs are already connected in hardware. No
  host push/pull happens on an internal connection.

- **Auto-exposure.** Any sub-kernel tensor **not** named in a connection is
  automatically promoted to a top-level composite interface, keeping its name.
  Here `scale.data_in` becomes the composite's `data_in`, and
  `offset.data_out` becomes `data_out`. Those are the only tensors you push/pull
  from the host. (If two exposed tensors would collide on a name, that's when
  you'd rename — but in the common linear pipeline they don't.)

- **`{ref}_ctrl` register handles.** For each sub-kernel, vTen auto-creates a
  `{ref}_ctrl` register handle pointing at that sub-kernel's control interface —
  so `scale_ctrl` and `offset_ctrl` exist even if you don't declare them. The
  example declares them explicitly for clarity, but you can rely on the
  auto-created ones. Use them exactly like a single kernel's `register(...)`
  handle: `ctx.write_register(self.scale_ctrl, {"start": 1}, ...)`.

### A composite has no `kernel_spec.yaml` of its own

This is the biggest structural difference from a single kernel. A composite
**does not** set `spec = "..."` and has no YAML file. Each sub-kernel carries its
own `kernel_spec.yaml`; the composite is pure Python that references them. Its
interfaces are the *union* of the sub-kernels' auto-exposed interfaces, computed
at flatten time.

---

## The auto-chained `forward()`

`CompositeKernel` provides a default `forward()` that evaluates the connection
graph in topological order: it runs each sub-kernel's `forward()`, propagates
outputs along `>>` connections into the next sub-kernel's inputs, and collects
the exposed outputs
([vten/kernel/composite.py](../vten/kernel/composite.py)). **You usually don't
override it.**

For `scale_add`, the auto-chained golden is exactly
`offset.forward(scale.forward(data_in))`. The example *does* override `forward()`
above — but purely to inline the two steps as documentation; deleting the
override and relying on the auto-chain gives the same result, as long as each
sub-kernel's own `forward()` is correct.

The same auto-chaining powers **`generate_inputs()`**: a sub-kernel that has no
`generate_inputs()` of its own gets its connected inputs filled automatically by
running the upstream chain (upstream `generate_inputs()` + `forward()`,
propagated through connections). So you generally only implement
`generate_inputs()` for the composite's exposed *input* (`data_in`), and let the
chain feed everything downstream.

When to override the composite `forward()`: only when the true end-to-end golden
isn't simply the composition of the sub-kernels' `forward()`s (e.g. a shared
accumulator, a cross-stage effect, or a cycle the dataflow evaluator can't infer).

---

## Multi-stage pipeline — dma_pipeline

`dma_pipeline` is a four-stage, fully memory-mapped composite:
`ReadDMA → Scale → Offset → WriteDMA`. `ReadDMA` pulls a buffer from DDR and
emits a stream; `Scale` and `Offset` transform it; `WriteDMA` writes the result
back to DDR. This mirrors a Vitis multi-IP dataflow kernel.

### `kernels/dma_pipeline/dma_pipeline_kernel.py`

```python
from read_dma.read_dma_kernel import ReadDMAKernel
from scale.scale_kernel import ScaleKernel
from offset.offset_kernel import OffsetKernel
from write_dma.write_dma_kernel import WriteDMAKernel


class DmaPipelineKernel(CompositeKernel):
    """Composite: ReadDMA → Scale → Offset → WriteDMA."""

    # Sub-kernels
    read_dma = ReadDMAKernel()
    scale = ScaleKernel()
    offset = OffsetKernel()
    write_dma = WriteDMAKernel()

    # Register proxies (also auto-created if omitted)
    rdma_ctrl = register("rdma_ctrl")
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")
    wdma_ctrl = register("wdma_ctrl")

    # Internal connections form the chain
    connections = [
        read_dma.data_out >> scale.data_in,
        scale.data_out >> offset.data_in,
        offset.data_out >> write_dma.data_in,
    ]

    # Auto-exposed: read_dma.data_in → data_in, write_dma.data_out → data_out

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        data = inputs.get("data_in", self.data_in.data)
        x = data.to(torch.int16) * self.scale_factor
        x = x.clamp(-128, 127)
        ov = self.offset_value
        if ov >= 128:
            ov = ov - 256
        x = x + ov
        return {"data_out": x.clamp(-128, 127).to(torch.int8)}

    def run(self, ctx) -> None:
        # Push input to the AXI4 BFM (ReadDMA reads from here)
        h_push = ctx.push_tensor(self.data_in)

        # Pull output early so the AXI4 BFM has a PULL entry ready
        h_pull = ctx.pull_tensor(self.data_out, dep=h_push)

        # Configure all sub-kernels (auto_bind addresses + runtime_params)
        h_cfg = ctx.configure(self, dep=h_push)

        # Start every sub-kernel
        h_start_rdma = ctx.write_register(self.rdma_ctrl, {"start": 1}, dep=h_cfg)
        h_start_scale = ctx.write_register(self.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_offset = ctx.write_register(self.offset_ctrl, {"start": 1}, dep=h_cfg)
        h_start_wdma = ctx.write_register(self.wdma_ctrl, {"start": 1}, dep=h_cfg)

        # Poll all done flags
        h_poll_rdma = ctx.poll_register(self.rdma_ctrl, "done", dep=h_start_rdma)
        h_poll_scale = ctx.poll_register(self.scale_ctrl, "done", dep=h_start_scale)
        h_poll_offset = ctx.poll_register(self.offset_ctrl, "done", dep=h_start_offset)
        h_poll_wdma = ctx.poll_register(self.wdma_ctrl, "done", dep=h_start_wdma)

        # Output commits only after every stage is done
        h_pull.add_commit_dependency(h_poll_rdma)
        h_pull.add_commit_dependency(h_poll_scale)
        h_pull.add_commit_dependency(h_poll_offset)
        h_pull.add_commit_dependency(h_poll_wdma)
```

Notes on the four-stage pattern:

- **Only the endpoints touch the host.** `read_dma.data_in` and
  `write_dma.data_out` are the auto-exposed tensors — the only ones you
  `push_tensor` / `pull_tensor`. The three internal streams are `>>` connections
  and carry no BFM traffic.
- **One `configure(self)` programs the whole pipeline.** It expands to WRITE_REG
  for every sub-kernel's `auto_bind` registers (source/dest addresses, lengths)
  plus any `runtime_params` registers (`scale`'s `scale_factor`/`length`,
  `offset`'s `offset_value`).
- **Start all, then poll all.** Each `write_register(..., {"start": 1})` uses the
  auto-created `{ref}_ctrl` handle. The final pull gains a commit dependency on
  *every* stage's `done`, so the read-back is only valid once the whole chain has
  drained.
- **Topological order is derived from `connections`.** You list connections in
  any order; vTen computes the dependency order
  (`ReadDMA → Scale → Offset → WriteDMA`) itself for the golden chain.

---

## Running & verifying a composite

You run a composite exactly like a single kernel — through a `TestScenario` and
the CLI. The scenario names the composite by directory name:

```python
from vten import TestScenario

class TestScaleAdd(TestScenario):
    """Composite kernel: scale then offset, with a parameter sweep."""

    kernel = "scale_add"

    configs = [
        {"name": "default"},                                         # N=1024, scale=2, off=1
        {"name": "identity", "scale_factor": 1, "offset_value": 0},  # pass-through
        {"name": "big_scale", "scale_factor": 5, "offset_value": 3},
        {"name": "small_n", "N": 32},                                # 1 beat
        {"name": "negative_off", "offset_value": 251},               # -5 as uint8 (0xFB)
    ]
```

```bash
vten build --kernel scale_add
vten run --kernel scale_add --test TestScaleAdd --verify
```

Config values sweep the sub-kernels' params (`scale_factor`, `offset_value`, `N`)
— they flow into `compute_derived_params()` and the auto-chained `forward()`, so
the golden updates automatically per config. With `--verify`, each config's
`data_out` is checked against the chained golden. See
[testing_guide.md](testing_guide.md) for the full scenario API.

---

## Probing internal connections

Because internal `>>` wires carry no host traffic, you can't read them with
`pull_tensor`. Instead, vTen supports **probes** that passively snoop a wire and
compare it beat-by-beat against golden during simulation.

There are two flavors, distinguished by whether the probe name is dotted
([vten/runtime/probe_manager.py](../vten/runtime/probe_manager.py)):

- **Output probe — plain name** (e.g. `"data_out"`). Marks the matching
  `pull_tensor` for beat-level BFM comparison. You can also set it inline:

  ```python
  h_pull = ctx.pull_tensor(k.data_out, dep=h_cfg, probe=True)
  ```

- **Internal probe — dotted name** (e.g. `"scale.data_out"`). Names a
  *sub-kernel wire* — `<sub_ref>.<tensor_name>`. vTen attaches a passive probe
  BFM to that internal connection and auto-extracts the expected golden from the
  composite's `forward()` chain (the source sub-kernel's output). This is how you
  verify the *middle* of the pipeline, not just the endpoints.

Declarative probes are listed on the `TestScenario` via the `probes` attribute:

```python
class TestScaleAddProbe(TestScenario):
    kernel = "scale_add"
    probes = ["scale.data_out"]   # snoop the internal wire between scale and offset
```

At `run()` time the framework applies each probe: plain names flip `probe=True`
on the matching PULL op, and dotted names become internal-probe requests whose
golden is resolved from the composite's forward-chain pool
([vten/runtime/context.py](../vten/runtime/context.py),
`_apply_declarative_probes` / `_resolve_internal_probe_golden`). A probe whose
beats diverge from golden fails the test at the exact beat, which localizes a bug
to a specific stage instead of just the final output.

---

## Differences from single kernels

| | Single `Kernel` | `CompositeKernel` |
|---|---|---|
| `kernel_spec.yaml` | Required (one per kernel). | **None** — each sub-kernel carries its own. |
| `spec = "..."` | Set to the YAML path. | Not set. |
| Sub-kernels | — | Declared as plain instances (`scale = ScaleKernel()`). |
| Interfaces | Declared via `Tensor(interface=...)`. | Auto-exposed from unconnected sub-kernel tensors. |
| Internal wires | — | `connections = [a.out >> b.in]` (suppresses BFMs). |
| Control handles | `register("ctrl")`. | Auto-created `{ref}_ctrl` per sub-kernel (declare optionally). |
| `forward()` | You write it. | Auto-chained across sub-kernels; override only for cross-stage golden. |
| `generate_inputs()` | You write it. | Only for exposed inputs; downstream filled by the chain. |
| Probing internals | N/A | Dotted probe names (`"scale.data_out"`) snoop `>>` wires. |

The mental model: a `CompositeKernel` is a thin orchestration layer. All the
protocol detail (packing, registers, `auto_bind`, `memory_regions`) lives in the
sub-kernels' own specs, described in [kernel_guide.md](kernel_guide.md). The
composite only adds the wiring (`>>`), the exposure rules, and one `run(ctx)`
that starts and gates the whole pipeline.
