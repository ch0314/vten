# scale_add — CompositeKernel pipeline example

This example teaches **kernel composition**: two independent IPs are wired into a
single pipeline using vTen's `CompositeKernel` and the `>>` connection operator.

```
data_in → Scale(× scale_factor) → Offset(+ offset_value) → data_out
```

Each stage is a real, separately-verifiable AXI4-Stream kernel with its own
AXI4-Lite control register. The composite exposes a single `data_in` / `data_out`
pair to the user, hiding the internal `scale → offset` wiring.

## What this teaches

- **`CompositeKernel`:** declaring sub-kernels as instances and connecting them.
- **The `>>` operator:** `scale.data_out >> offset.data_in` connects the scale
  stage's output stream directly to the offset stage's input stream. That
  internal tensor pair is consumed by the connection and is **not** exposed.
- **Auto-expose:** tensors *not* consumed by a connection (`scale.data_in`,
  `offset.data_out`) are automatically promoted to the composite's `data_in` /
  `data_out`. Each sub-kernel's `ctrl` register is exposed as
  `scale_ctrl` / `offset_ctrl`.
- **Per-stage control:** `run()` writes `start` to *both* controllers and polls
  *both* `done` flags before committing the PULL — coordinating two IPs in one run.
- **Composed golden:** `forward()` chains the two behavioral models
  (int8-saturating multiply, then add), and `--verify` checks the DUT pipeline
  against it end-to-end.

## Files

| File | Role |
|------|------|
| `kernels/scale_add/scale_add_kernel.py` | `ScaleAddKernel(CompositeKernel)` — holds `scale = ScaleKernel()` and `offset = OffsetKernel()`, `connections = [scale.data_out >> offset.data_in]`, register proxies `scale_ctrl` / `offset_ctrl`, and a composed `forward()` / `run()`. |
| `kernels/scale/scale_kernel.py` | `ScaleKernel` — multiplies each `int8` element by `scale_factor` (saturating). AXI4-Stream in/out + AXI4-Lite `ctrl`. |
| `kernels/offset/offset_kernel.py` | `OffsetKernel` — adds `offset_value` to each `int8` element (saturating, with uint8→signed wrap). AXI4-Stream in/out + AXI4-Lite `ctrl`. |
| `kernels/scale/kernel_spec.yaml`, `kernels/offset/kernel_spec.yaml` | Per-stage interface/packing/register maps. (The composite has no spec of its own — it flattens its sub-kernels.) |
| `kernels/scale_add/tests/test_scale_add.py` | `TestScaleAdd` (parameter sweep), `TestScaleAddProbe` (output probe), and `TestScaleAddInternalProbe` (internal/dotted probe). |
| `kernels/dma_pipeline/dma_pipeline_kernel.py` | `DmaPipelineKernel(CompositeKernel)` — 4-stage `ReadDMA → Scale → Offset → WriteDMA` memory-mapped pipeline (documented below). |
| `vten.toml` | Project config. `[parameters] N = 1024`. Backend: `xsim`. |

> The standalone `scale` / `offset`, and `read_dma` / `write_dma` directories
> under `kernels/` are the individual-stage fixtures. The canonical composite
> tutorial kernel is `scale_add`; the larger **4-stage `dma_pipeline`** composite
> is documented in its own section below.

### Test scenarios

- **`TestScaleAdd`** — parameter sweep. Runs six configs: `default`
  (`N=1024`, scale=2, offset=1), `identity` (scale=1, offset=0, i.e. pass-through),
  `big_scale` (scale=5, offset=3), `small_n` (`N=32`, one beat), `large_n`
  (`N=4096`, 128 beats), and `negative_off` (offset=251 → `-5` as signed int8).
- **`TestScaleAddProbe`** — like `default` but with `probe=True` on the PULL, so
  the BFM checks each **output** beat against the composed golden during simulation.
- **`TestScaleAddInternalProbe`** — an **internal (dotted) probe**. See the
  [Output probe vs. internal probe](#output-probe-vs-internal-probe) section.

## Output probe vs. internal probe

vTen can compare DUT data against golden at two different taps:

| | `TestScaleAddProbe` (output probe) | `TestScaleAddInternalProbe` (internal probe) |
|--|--|--|
| Declared as | `probe=True` on `ctx.pull_tensor(data_out, ...)` | `probes = ["scale.data_out"]` (dotted name) |
| Where it taps | the **exposed output** `data_out` (offset stage result) | the **hidden internal wire** `scale.data_out` consumed by the `scale.data_out >> offset.data_in` connection |
| Golden it checks against | full `scale → offset` composed golden | mid-pipeline: the scale stage's output **before** offset is applied |
| RTL | reuses the same scale + offset DUTs — **no new RTL** | reuses the same scale + offset DUTs — **no new RTL** |

The dotted `probes=["scale.data_out"]` form asks a **passive probe BFM** to tap
the internal `scale → offset` connection and compare each beat against golden
scale-stage data. Because the `scale_add` composite defines a *custom* `forward()`
(it does not auto-chain), it exposes no `_golden_pool` for automatic internal
golden extraction — so `TestScaleAddInternalProbe.run()` supplies the wire golden
explicitly with `ctx.set_internal_probe_golden("scale", "data_out", scaled)`
(the saturating int8 multiply). This is the reliable pattern for internal probes
on custom-`forward()` composites; verified by reading
`vten/runtime/context.py`, `vten/runtime/probe_manager.py`, and the tensor names
in `kernels/scale_add/scale_add_kernel.py` (`scale.data_out`).

```bash
vten run --kernel scale_add --test TestScaleAddInternalProbe --backend verilator --verify
```

## `dma_pipeline` — a 4-stage memory-mapped composite

`kernels/dma_pipeline/` composes **four** IPs into a full DDR→compute→DDR
dataflow, the way a Vitis multi-IP kernel would. It reads a buffer from DDR over
AXI4, scales and offsets the stream, and writes the result back to DDR:

```
             ┌──────────┐   ┌───────┐   ┌────────┐   ┌───────────┐
  DDR ─AXI4─▶│ ReadDMA  │──▶│ Scale │──▶│ Offset │──▶│ WriteDMA  │─AXI4─▶ DDR
 (data_in)   │ read_dma │ s │ scale │ s │ offset │ s │ write_dma │      (data_out)
             └──────────┘   └───────┘   └────────┘   └───────────┘
                  ▲ AXI4-Lite ctrl on every stage (start / done) ▲
   ── s ── = internal AXI4-Stream wire created by a `>>` connection ──
```

Defined in `kernels/dma_pipeline/dma_pipeline_kernel.py` as
`DmaPipelineKernel(CompositeKernel)`:

```python
class DmaPipelineKernel(CompositeKernel):
    read_dma  = ReadDMAKernel()
    scale     = ScaleKernel()
    offset    = OffsetKernel()
    write_dma = WriteDMAKernel()

    connections = [
        read_dma.data_out >> scale.data_in,    # ReadDMA  → Scale
        scale.data_out    >> offset.data_in,   # Scale    → Offset
        offset.data_out   >> write_dma.data_in,# Offset   → WriteDMA
    ]
    # Auto-exposed: read_dma.data_in → data_in, write_dma.data_out → data_out
```

Walkthrough of `run()`:

1. **push** `data_in` to the AXI4 BFM that `ReadDMA` reads from.
2. **pull** `data_out` (dispatched early so the AXI4 BFM has a PULL entry ready).
3. **`configure(self)`** — one call writes every stage's `auto_bind` (DMA
   src/dst addresses + length) and `runtime_params` (scale_factor, offset_value)
   registers.
4. **start** all four stages (`write_register(..., {"start": 1})`).
5. **poll** all four `done` flags, and make the PULL commit only after every
   stage has finished (`h_pull.add_commit_dependency(...)`).

`forward()` is the composed golden: read (identity) → int8-saturating multiply →
saturating add → write (identity). `--verify` checks the DUT's DDR output against
it end-to-end.

Test scenarios (`kernels/dma_pipeline/tests/test_dma_pipeline.py`):

- **`TestDmaPipeline`** — parameter sweep over four configs: `default`
  (`N=1024`, scale=2, offset=1), `identity` (scale=1, offset=0, a pure DMA
  round-trip), `small_n` (`N=32`, minimal DMA), and `overflow`
  (scale=10, offset=50, exercising int8 saturation).
- **`TestDmaPipelineStore`** — same pipeline but drives `store_tensor()` readback
  after the PULL.

```bash
vten build --kernel dma_pipeline --backend verilator
vten run   --kernel dma_pipeline --test TestDmaPipeline --backend verilator --verify
```

## Build & run (open-source path)

`scale_add`'s `vten.toml` sets `default_backend = "xsim"`, but you can force the
open-source Verilator backend with `--backend verilator` (no Vivado needed).
Run from inside this directory, or pass `--project examples/scale_add` from the
repo root.

```bash
# 1. Build the Verilator simulation for the composite kernel
vten build --kernel scale_add --backend verilator

# 2. Run the full parameter sweep with bit-exact verification
vten run --kernel scale_add --test TestScaleAdd --backend verilator --verify

# 3. (Optional) run the beat-level probe scenario
vten run --kernel scale_add --test TestScaleAddProbe --backend verilator --verify

# 4. Summarize results
vten report
```

Quick no-RTL smoke test (runs the composed `forward()` only, no build needed):

```bash
vten run --kernel scale_add --test TestScaleAdd --backend cpu --verify
```

## Expected result

A bit-exact **PASS** for every config: the DUT pipeline output matches the
composed `scale → offset` golden, so each config reports `verification: 1/1
passed` and the summary reports `status: PASS` (`6/6 configs passed` for
`TestScaleAdd`).

## Backend availability

| Backend | Needs | Notes |
|---------|-------|-------|
| `cpu` | nothing | Runs the composed `forward()` only; no RTL. |
| `verilator` | Verilator (open source) | No Vivado required. |
| `xsim` | Vivado | Default backend for this example; set `vivado_path` in `vten.toml`. |
| `xrt` | Alveo FPGA / Vitis emulation + `.xclbin` | Not configured for this streaming composite. |

## See also

- [`../README.md`](../README.md) — the examples index & feature→example map.
- [`../../docs/composite_guide.md`](../../docs/composite_guide.md) — `CompositeKernel`, the `>>` operator, auto-expose, and multi-IP `run()` patterns.
- [`../../docs/kernel_guide.md`](../../docs/kernel_guide.md) — writing the individual `Scale` / `Offset` kernels and their DUTs.
- [`../../docs/testing_guide.md`](../../docs/testing_guide.md) — `TestScenario`, config sweeps, and the verification workflow.
- [`../../docs/cli_reference.md`](../../docs/cli_reference.md) — full CLI reference.
