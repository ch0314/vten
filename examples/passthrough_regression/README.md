# passthrough_regression — edge-case & regression fixtures

This project holds the **edge-case and regression fixtures** that were split out
of the [`passthrough`](../passthrough/README.md) tutorial to keep that example
minimal. Each kernel here is a small AXI4-Stream / AXI4 / AXI4-Lite DUT that
exercises one corner of vTen's packing/serialization, DMA, multi-port, compute,
or mismatch-detection machinery.

If you are learning vTen, start with the [`passthrough`](../passthrough/README.md)
tutorial first; come here when you want to see how the framework behaves at odd
datapath widths, with DMA, with multiple ports, or when a DUT is wrong on
purpose.

This project is self-contained: it has its own `vten.toml`, its own `rtl/`
(including a private copy of `passthrough.sv` used by the `unaligned` kernel),
and its own `kernels/`. Build and run it exactly like any other vTen project.

## Kernels

Each kernel's `TestScenario` class names (the values you pass to `--test`) are
listed alongside it.

### Packing-width & dtype variants

AXI4-Stream passthrough DUTs with varying `element_width` / `elements_per_beat`,
used to validate the tensor serialization / beat-packing code across bus widths
and element dtypes.

| Kernel | Bus / packing | Test class names |
|--------|---------------|------------------|
| `narrow8` | 8-bit bus, 1 × int8/beat | `TestNarrow8`, `TestNarrow8Probe` |
| `narrow32` | 32-bit bus, 4 × int8/beat | `TestNarrow32`, `TestNarrow32Probe` |
| `wide512` | 512-bit bus, 64 × int8/beat | `TestWide512`, `TestWide512Probe` |
| `odd24` | 24-bit bus, 3 × int8/beat (non-power-of-two) | `TestOdd24`, `TestOdd24Probe` |
| `beat7` | 56-bit bus, 7 × int8/beat (prime) | `TestBeat7`, `TestBeat7Probe` |
| `beat13` | 104-bit bus, 13 × int8/beat (prime) | `TestBeat13`, `TestBeat13Probe` |
| `int16_x1` | 16-bit bus, 1 × int16/beat | `TestInt16X1` |
| `int16_x7` | 112-bit bus, 7 × int16/beat | `TestInt16X7` |
| `int32_x2` | 64-bit bus, 2 × int32/beat | `TestInt32X2` |
| `float32_x4` | 128-bit bus, 4 × float32/beat | `TestFloat32X4` |
| `unaligned` | 256-bit bus, 32 × int8/beat; reuses `rtl/passthrough.sv` to exercise unaligned-length handling | `TestUnaligned` |

### DMA

DUTs with an AXI4 DMA master plus AXI4-Lite control.

| Kernel | Exercises | Test class names |
|--------|-----------|------------------|
| `stream_dma` | AXI4-Stream + AXI4 DMA + AXI4-Lite control | `TestStreamDma` |
| `stream_dma_v2` | Revised DMA control/handshake variant | `TestStreamDmaV2` |

### Scatter / multi-port

| Kernel | Exercises | Test class names |
|--------|-----------|------------------|
| `stream_scatter` | Stream fan-out into dual AXI4 memory regions under AXI4-Lite control | `TestStreamScatter` |
| `multi_port_scatter` | Multiple AXI4-Stream ports fanned out from one input | `TestMultiPortScatter` |

### Compute

| Kernel | Exercises | Test class names |
|--------|-----------|------------------|
| `vector_alu` | AXI4 memory-mapped element-wise compute (ADD / SUB / MUL) with AXI4-Lite register control | `TestVectorAluAdd`, `TestVectorAluSub`, `TestVectorAluMul`, `TestVectorAluProbe` |

### Negative test

| Kernel | Exercises | Test class names |
|--------|-----------|------------------|
| `broken_passthrough` | An **intentionally-faulty DUT** — it XORs every data byte with `0x01`, so the DUT output no longer equals the input. Verification is **expected to FAIL**, which is how we confirm vTen actually detects a mismatch (both host-side compare and, via the probe scenario, beat-level probe mismatch reporting). Do **not** "fix" the RTL — a PASS here would be the real bug. | `TestBrokenPassthrough`, `TestBrokenPassthroughProbe` |

## Build & run one kernel

The `verilator` and `cpu` backends need no Vivado. Run from inside this project
directory, or pass `--project examples/passthrough_regression` from the repo
root.

```bash
# 1. Build the Verilator simulation for a kernel
vten build --kernel beat13 --backend verilator

# 2. Run one of its scenarios with bit-exact verification
vten run --kernel beat13 --test TestBeat13 --backend verilator --verify

# 3. Summarize results
vten report
```

Substitute any kernel name and one of its `Test*` class names from the tables
above. To run every scenario in a kernel's `tests/` directory, omit `--test`:

```bash
vten run --kernel beat13 --backend verilator --verify
```

The `broken_passthrough` fixture is the exception — its verification is
**supposed to fail**:

```bash
vten build --kernel broken_passthrough --backend verilator
vten run --kernel broken_passthrough --test TestBrokenPassthrough --backend verilator --verify
# => verification: 0/1 passed  (expected: the DUT corrupts the data)
```

## See also

- [`../passthrough/`](../passthrough/README.md) — the canonical single-kernel tutorial these fixtures were split out of.
- [`E2E_TEST_RESULTS.md`](E2E_TEST_RESULTS.md) — recorded pass/fail status for several of these kernels.
- [`../../docs/kernel_guide.md`](../../docs/kernel_guide.md) — writing `kernel_spec.yaml`, the `Kernel` class, and the DUT.
- [`../../docs/testing_guide.md`](../../docs/testing_guide.md) — `TestScenario`, configs, and the verification workflow.
- [`../../docs/cli_reference.md`](../../docs/cli_reference.md) — full `vten build` / `vten run` / `vten report` reference.
