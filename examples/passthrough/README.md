# passthrough — AXI4-Stream single-kernel example

This is the canonical "hello world" of vTen: a **single kernel** driven over
**AXI4-Stream**. The DUT copies its input stream straight to its output stream
(`m_axis_tdata = s_axis_tdata`), so a bit-exact verification succeeds when the
output equals the input.

Use this example to learn the end-to-end flow before moving on to the
memory-mapped ([`../mm_loopback`](../mm_loopback/README.md)) and composite
([`../scale_add`](../scale_add/README.md)) examples.

## What this teaches

- **Protocol class:** AXI4-Stream (`axi4_stream`) — the simplest of the three
  BFM protocols. PUSH streams `data_in` into the DUT, PULL drains `data_out`.
- **Single kernel:** one `Kernel` subclass, no control registers, no composition.
- **Host-side verification:** the behavioral `forward()` returns a clone of the
  input, and `--verify` compares the DUT's streamed output against it byte-for-byte.

## Files

| File | Role |
|------|------|
| `kernels/passthrough/passthrough_kernel.py` | `PassthroughKernel` — declares `data_in` (`input_stream`) / `data_out` (`output_stream`) tensors, `int8`, shape `(${N},)`; `forward()` returns `data_in.data.clone()`; `run()` does `push_tensor` → `pull_tensor`. |
| `kernels/passthrough/kernel_spec.yaml` | Maps the two tensors to the RTL ports `s_axis` / `m_axis`, both `axi4_stream`, packing 32 × 8-bit elements per beat (256-bit datapath). |
| `rtl/passthrough.sv` | The DUT: combinational stream passthrough. |
| `kernels/passthrough/tests/test_passthrough.py` | Two test scenarios (see below). |
| `vten.toml` | Project config. `[parameters] N = 1024`. Backends: `xsim`, `verilator`. |

### Test scenarios

Test names are the `TestScenario` **class names** (pass them to `--test`):

- **`TestPassthrough`** — standard run; PUSH input, PULL output, verify against golden.
- **`TestPassthroughProbe`** — same data path but with `probe=True` on the PULL, so
  the BFM compares each output *beat* against the golden buffer during simulation
  (beat-level checking, in addition to the final tensor comparison).

## Build & run (open-source path)

The **verilator** and **cpu** backends need no Vivado. Run from inside this
example directory (commands use `--project .` implicitly), or pass
`--project examples/passthrough` from the repo root.

```bash
# 1. Build the Verilator simulation for the `passthrough` kernel
vten build --kernel passthrough --backend verilator

# 2. Run the standard scenario with bit-exact verification
vten run --kernel passthrough --test TestPassthrough --backend verilator --verify

# 3. (Optional) run the beat-level probe scenario
vten run --kernel passthrough --test TestPassthroughProbe --backend verilator --verify

# 4. Summarize results (reads results/<kernel>/<test>/summary.json)
vten report
```

To run every scenario in the kernel's `tests/` directory, omit `--test`:

```bash
vten run --kernel passthrough --backend verilator --verify
```

The `cpu` backend skips RTL entirely and just executes `forward()` — handy as a
zero-dependency smoke test (`--verify` trivially passes since DUT output ==
golden). It needs no `vten build` step:

```bash
vten run --kernel passthrough --test TestPassthrough --backend cpu --verify
```

## Expected result

A bit-exact **PASS**: the DUT's `data_out` equals `data_in`, so
`verification: 1/1 passed` and the run summary reports `status: PASS`.

## Backend availability

| Backend | Needs | Notes |
|---------|-------|-------|
| `cpu` | nothing | Runs `forward()` only; no RTL. Fastest smoke test. |
| `verilator` | Verilator (open source) | No Vivado required. |
| `xsim` | Vivado (`xvlog`/`xelab`/`xsim`) | Set `vivado_path` in `vten.toml`. |
| `xrt` | Alveo FPGA or Vitis emulation + an `.xclbin` | Not configured for this example (streaming loopback). See [`../mm_loopback`](../mm_loopback/README.md) for the XRT path. |

## Edge-case & regression fixtures

`passthrough` is the **canonical tutorial kernel**, and it is the only kernel in
this example. The packing-width/dtype variants, DMA, scatter/multi-port, compute,
and intentionally-faulty regression fixtures that used to live here have been
split out into the sibling project
[`../passthrough_regression/`](../passthrough_regression/README.md). Look there if
you want to see how vTen handles odd datapath widths, DMA, multi-port scatter,
or mismatch detection.

## See also

- [`../../docs/kernel_guide.md`](../../docs/kernel_guide.md) — writing `kernel_spec.yaml`, the `Kernel` class, and the DUT.
- [`../../docs/testing_guide.md`](../../docs/testing_guide.md) — `TestScenario`, configs, and the verification workflow.
- [`../../docs/cli_reference.md`](../../docs/cli_reference.md) — full `vten build` / `vten run` / `vten report` reference.
- [`../passthrough_regression/`](../passthrough_regression/README.md) — edge-case / regression fixtures split out of this tutorial.
