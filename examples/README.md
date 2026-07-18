# vTen examples — index & learning path

This directory holds runnable vTen projects that build up from a one-line
"hello world" to multi-IP memory-mapped pipelines. Every project has its own
`vten.toml`, `kernels/`, `rtl/`, and a README. The **verilator** and **cpu**
backends need no Vivado, so you can run everything here open-source.

## Learning path (read the READMEs in this order)

1. **[`passthrough/`](passthrough/README.md)** — the canonical single-kernel,
   AXI4-Stream "hello world". The DUT copies its input stream to its output. Also
   the home of the **`chunks=`** (host-side read splitting) and **layout hook**
   (`layout_{name}()`) demonstrations, both built on the same passthrough DUT.
2. **[`mm_loopback/`](mm_loopback/README.md)** — adds **AXI4 memory-mapped**
   masters + **AXI4-Lite** control + **`auto_bind`** address/length splits, plus
   **array interfaces** (one tensor fanned across many ports) and the
   **Inference API** demo (`infer.py`).
3. **[`scale_add/`](scale_add/README.md)** — **`CompositeKernel`**: wire several
   IPs into one pipeline with `>>`. Covers output probes, **internal (dotted)
   probes**, multi-config sweeps, and the 4-stage `dma_pipeline` composite.
4. **[`passthrough_regression/`](passthrough_regression/README.md)** — edge-case
   and regression fixtures: odd/prime datapath widths, many dtypes, DMA,
   **split interfaces**, multi-port scatter, compute, and an intentionally-broken
   DUT used to confirm vTen actually *detects* mismatches.

> Also runnable, but living under `benchmarks/` rather than `examples/`:
> [`../benchmarks/cocotb_comparison/`](../benchmarks/cocotb_comparison/README.md)
> — a cocotb-vs-vTen benchmark on the `passthrough` DUT.

## Feature → example map

Each feature points to the example (and the specific kernel / test) that teaches
it. `Test*` names are `TestScenario` class names — pass them to `--test`.

| Feature | Where it's taught | Kernel / test |
|---------|-------------------|---------------|
| **AXI4-Stream** | [`passthrough/`](passthrough/README.md) | `passthrough` — `TestPassthrough` |
| **AXI4 memory-mapped** | [`mm_loopback/`](mm_loopback/README.md) | `mm_loopback` — `TestMmLoopback` |
| **AXI4-Lite** (control regs, start/done) | [`mm_loopback/`](mm_loopback/README.md) | `mm_loopback` `ctrl` register block |
| **Packing widths / dtypes** | [`passthrough_regression/`](passthrough_regression/README.md) | `narrow8`, `wide512`, `odd24`, `beat7`, `beat13`, `int16_x1`, `int32_x2`, `float32_x4`, … |
| **`auto_bind`** (address `_lo`/`_hi`, `size_beats`) | [`mm_loopback/`](mm_loopback/README.md) | `mm_loopback` `kernel_spec.yaml` `auto_bind:` entries |
| **`memory_regions`** | [`mm_loopback/`](mm_loopback/README.md) | `mm_loopback` `ddr` region |
| **Array interfaces** (`array: dimensions`) | [`mm_loopback/`](mm_loopback/README.md#array-interfaces--one-tensor-fanned-across-many-ports) | `stream_array_pt` (`[4]`), `stream_array_2d` (`[2,2]`), `mm_array_lb` (`[4]` AXI4) |
| **Split interfaces** (`split: ports`) | [`passthrough_regression/`](passthrough_regression/README.md) | `multi_port_scatter` — `TestMultiPortScatter` |
| **`CompositeKernel`** (`>>` wiring, auto-expose) | [`scale_add/`](scale_add/README.md) | `scale_add` — `TestScaleAdd`; 4-stage `dma_pipeline` |
| **Output probes** (`probe=True` on PULL) | [`passthrough/`](passthrough/README.md), [`scale_add/`](scale_add/README.md) | `TestPassthroughProbe`, `TestScaleAddProbe` |
| **Internal (dotted) probes** (`probes=["scale.data_out"]`) | [`scale_add/`](scale_add/README.md#output-probe-vs-internal-probe) | `scale_add` — `TestScaleAddInternalProbe` |
| **Multi-config sweeps** (`configs = [...]`) | [`scale_add/`](scale_add/README.md) | `scale_add` — `TestScaleAdd` (6 configs); `dma_pipeline` — `TestDmaPipeline` |
| **`chunks=`** on `pull_tensor` (host-side read split) | [`passthrough/`](passthrough/README.md#chunked-pull--chunk_passthrough-kernel-chunks-on-pull_tensor) | `chunk_passthrough` — `TestChunkPassthrough` |
| **Layout hooks** (`layout_{name}()` / `unlayout_{name}()`) | [`passthrough/`](passthrough/README.md#layout-hook--layout_passthrough-kernel) | `layout_passthrough` — `TestLayoutPassthrough` |
| **Inference API** (`InferenceSession` / `InferenceModule`) | [`mm_loopback/`](mm_loopback/README.md#inference-api-demo--inferpy) | `mm_loopback/infer.py` |

> Kernels/tests marked with `chunks=`, internal probes, and layout hooks were
> added to demonstrate those features on **existing, already-passing DUTs** (no
> new RTL). They ship with a comment noting they should be confirmed with
> `vten run … --verify` on a real backend.

## Running any example

```bash
cd <example>/                       # e.g. cd passthrough/
vten build --kernel <kernel> --backend verilator
vten run   --kernel <kernel> --test <TestName> --backend verilator --verify
vten report
```

The `cpu` backend runs `forward()` only (no build, no RTL) as a zero-dependency
smoke test; `xsim` needs Vivado; `xrt` needs an Alveo FPGA or Vitis emulation.

## Docs

- [`../docs/kernel_guide.md`](../docs/kernel_guide.md) — `kernel_spec.yaml`, tensors, packing, `memory_regions`, `auto_bind`, and the **layout hook**.
- [`../docs/composite_guide.md`](../docs/composite_guide.md) — `CompositeKernel`, the `>>` operator, auto-expose, multi-IP `run()`.
- [`../docs/testing_guide.md`](../docs/testing_guide.md) — `TestScenario`, config sweeps, probes, and the verification workflow.
- [`../docs/cli_reference.md`](../docs/cli_reference.md) — full `vten build` / `vten run` / `vten report` reference and `vten.toml`.
- [`../docs/architecture.md`](../docs/architecture.md) — how vTen works internally (compile pipeline, array/split flattening, probes, layout, Inference API).
- [`../docs/paper_vs_code.md`](../docs/paper_vs_code.md) — maps the DAC paper's terminology onto the code.
