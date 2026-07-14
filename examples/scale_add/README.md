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
| `kernels/scale_add/tests/test_scale_add.py` | `TestScaleAdd` (parameter sweep) and `TestScaleAddProbe` (beat-level probe). |
| `vten.toml` | Project config. `[parameters] N = 1024`. Backend: `xsim`. |

> The `dma_pipeline`, `read_dma`, `write_dma`, and standalone `scale` / `offset`
> directories under `kernels/` are additional fixtures for the individual stages
> and DMA variants — the canonical composite tutorial kernel is `scale_add`.

### Test scenarios

- **`TestScaleAdd`** — parameter sweep. Runs six configs: `default`
  (`N=1024`, scale=2, offset=1), `identity` (scale=1, offset=0, i.e. pass-through),
  `big_scale` (scale=5, offset=3), `small_n` (`N=32`, one beat), `large_n`
  (`N=4096`, 128 beats), and `negative_off` (offset=251 → `-5` as signed int8).
- **`TestScaleAddProbe`** — like `default` but with `probe=True` on the PULL, so
  the BFM checks each output beat against the composed golden during simulation.

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

- [`../../docs/composite_guide.md`](../../docs/composite_guide.md) — `CompositeKernel`, the `>>` operator, auto-expose, and multi-IP `run()` patterns.
- [`../../docs/kernel_guide.md`](../../docs/kernel_guide.md) — writing the individual `Scale` / `Offset` kernels and their DUTs.
- [`../../docs/testing_guide.md`](../../docs/testing_guide.md) — `TestScenario`, config sweeps, and the verification workflow.
- [`../../docs/cli_reference.md`](../../docs/cli_reference.md) — full CLI reference.
