# mm_loopback — AXI4 memory-mapped + AXI4-Lite control example

This example adds two protocol classes on top of the streaming
[`../passthrough`](../passthrough/README.md) example:

- **AXI4 (memory-mapped master):** the DUT reads a buffer from DDR over its
  `m_axi_in` port and writes it back to another DDR buffer over `m_axi_out`.
- **AXI4-Lite (control):** an `s_axilite` register block starts the kernel and
  reports `done`, and — crucially — supplies the DDR **addresses** and transfer
  **length** to the DUT.

The transfer is a pure loopback, so `data_out == data_in` bit-for-bit.

This is also the home of vTen's **Level-3 Inference API demo**
([`infer.py`](infer.py)), the only runnable example of
`InferenceSession` / `InferenceModule`.

## What this teaches

- **AXI4 memory-mapped transfers** driven by the runtime's LOAD/PUSH/PULL/STORE
  commands into a declared `memory_region` (`ddr`).
- **AXI4-Lite control** with `WRITE_REG` (start) / `POLL_REG` (done).
- **`auto_bind` address split:** the 64-bit source/destination addresses that the
  runtime allocates for `data_in` / `data_out` are automatically split into
  `*_addr_lo` (bits `31:0`) and `*_addr_hi` (bits `63:32`) register writes, and
  the beat count is bound to `length` via `value: size_beats`. No manual address
  bookkeeping — see the `auto_bind:` entries in `kernel_spec.yaml`.

## Files

| File | Role |
|------|------|
| `kernels/mm_loopback/mm_loopback_kernel.py` | `MmLoopbackKernel` — declares `ctrl = register("ctrl")`, `data_in` (`mem_in`, `uint8`, `HOST_TO_DEV`) and `data_out` (`mem_out`, `uint8`, `DEV_TO_HOST`), both shape `(${N},)`. `run()` does push → configure → write `start` → poll `done` → pull. `forward()` returns `data_in.data.clone()` (identity). |
| `kernels/mm_loopback/kernel_spec.yaml` | Declares the `ddr` memory region, the `ctrl` AXI4-Lite register map (with `auto_bind` address/length splits), and the `mem_in` / `mem_out` AXI4 masters (256-bit, `DDR[0]`, `arg_index` 0/1 for XRT). |
| `kernels/mm_loopback/rtl/mm_loopback_core.sv` | The DUT: AXI4 read → AXI4 write loopback with an AXI4-Lite controller. |
| `kernels/mm_loopback/tests/test_mm_loopback.py` | `TestMmLoopback` scenario. |
| `infer.py` | Level-3 Inference API demo (`InferenceSession` / `InferenceModule`). See below. |
| `vten.toml` | Project config: `[parameters] N = 1024`; backends `xsim`, `verilator`, and a fully-populated `[backend.xrt]` (U280, `target = "hw_emu"`). |

### Test scenario

- **`TestMmLoopback`** — pushes a random `data_in`, configures the DUT via
  AXI4-Lite, runs the loopback, pulls `data_out`, and (with `--verify`) checks it
  is bit-identical to the input.

## Array interfaces — one tensor fanned across many ports

Beyond the single-port `mm_loopback` kernel, this project ships three
**already-passing** kernels that demonstrate vTen's **`array:` interface**: one
logical tensor is block-split across an `array` of identical RTL ports. In the
`kernel_spec.yaml`, an interface gains an `array:` block:

```yaml
interfaces:
  din:
    rtl_port: s_axis_din
    protocol: axi4_stream
    tensor: data_in
    array:
      dimensions: [4]      # 4 parallel ports
    packing: { element_width: 8, elements_per_beat: 32 }
```

At flatten time the logical interface name expands into one physical port per
array element, following the flattened naming convention
`<iface>_<idx...>` — e.g. `dimensions: [4]` → `din_0, din_1, din_2, din_3`, and
`dimensions: [2, 2]` → `din_0_0, din_0_1, din_1_0, din_1_1`. The tensor is
**block-split** (`vten/runtime/serializer.py::block_split_data`) so each port carries
a contiguous slice of the serialized stream; on the way back the per-port
buffers are concatenated in order (`output_reader.py::read_tensor_bytes`). The
kernel's Python `run()` / `forward()` are identical to the single-port case — the
port expansion is entirely spec-driven.

| Kernel | Interface | `array: dimensions` | Expanded ports (per side) | Protocol | Test class names |
|--------|-----------|---------------------|---------------------------|----------|------------------|
| `stream_array_pt` | `din` / `dout` | `[4]` | `din_0..din_3` / `dout_0..dout_3` | AXI4-Stream | `TestStreamArrayPt`, `TestStreamArrayPtProbe` |
| `stream_array_2d` | `din` / `dout` | `[2, 2]` | `din_0_0, din_0_1, din_1_0, din_1_1` (and `dout_*`) | AXI4-Stream | `TestStreamArray2d` |
| `mm_array_lb` | `mem_in` / `mem_out` | `[4]` | `m_axi_in` × 4 / `m_axi_out` × 4 | AXI4 mem-mapped + AXI4-Lite | `TestMmArrayLb` |

- **`stream_array_pt`** — 4-channel AXI4-Stream passthrough. Each of the four
  channels independently passes through its block of the tensor. Its probe
  variant (`TestStreamArrayPtProbe`, in `tests/test_stream_array_pt_probe.py`)
  sets `probe=True` on the PULL so **each of the four array-element BFMs**
  compares its received beats against golden independently.
- **`stream_array_2d`** — the same idea with a **2-D** `[2, 2]` array, showing how
  multi-dimensional `dimensions:` expand to nested flat names.
- **`mm_array_lb`** — the memory-mapped analogue: the `mem_in` / `mem_out` AXI4
  masters are each an `array: dimensions: [4]`, so the loopback runs across four
  parallel AXI4 channels while a single `ctrl` AXI4-Lite block (with `auto_bind`
  address/length splits, exactly like `mm_loopback`) drives the whole transfer.

> **Split vs. array.** An `array:` interface fans a tensor over a *homogeneous*
> set of identical ports declared by `dimensions:`. A **`split:`** interface
> (see [`../passthrough_regression/`](../passthrough_regression/README.md)'s
> `multi_port_scatter`) instead lists *named* ports explicitly under `split:
> ports:` — use `split:` when the ports are individually named/heterogeneous, and
> `array:` when they are a regular vector of identical ports.

Each of these kernels is a standalone, passing DUT — build and run them exactly
like `mm_loopback`:

```bash
vten build --kernel stream_array_pt --backend verilator
vten run   --kernel stream_array_pt --test TestStreamArrayPt --backend verilator --verify
```

## Build & run (open-source path)

The **verilator** and **cpu** backends need no Vivado. Run from inside this
directory, or pass `--project examples/mm_loopback` from the repo root.

```bash
# 1. Build the Verilator simulation
vten build --kernel mm_loopback --backend verilator

# 2. Run the loopback scenario with bit-exact verification
vten run --kernel mm_loopback --test TestMmLoopback --backend verilator --verify

# 3. Summarize results
vten report
```

Quick no-RTL smoke test (runs `forward()` only, no build needed):

```bash
vten run --kernel mm_loopback --test TestMmLoopback --backend cpu --verify
```

## Build & run (XRT / FPGA path)

`vten.toml` already contains a `[backend.xrt]` section targeting a Xilinx U280
with `target = "hw_emu"`. Building the `.xclbin` requires **Vivado + Vitis**, and
running it requires either a real **Alveo FPGA** (`target = "hw"`) or Vitis
**hardware emulation** (`target = "hw_emu"`):

```bash
# Build the xclbin for hardware emulation (needs Vitis)
vten build --kernel mm_loopback --backend xrt --target hw_emu

# Run on the emulated/real device
vten run --kernel mm_loopback --test TestMmLoopback --backend xrt --verify
```

The produced xclbin lands at the `xclbin_path` in `vten.toml`
(`kernels/mm_loopback/build/xrt/mm_loopback_hw_emu.xclbin`). The `emconfig.json`
in this directory is the emulation device config used by `hw_emu`.

## Expected result

A bit-exact **PASS**: `data_out` equals `data_in` (the loopback is the identity),
so `verification: 1/1 passed` and the summary reports `status: PASS`.

## Backend availability

| Backend | Needs | Notes |
|---------|-------|-------|
| `cpu` | nothing | Runs `forward()` only; no RTL, no DMA. |
| `verilator` | Verilator (open source) | No Vivado required. |
| `xsim` | Vivado | Set `vivado_path` in `vten.toml`. |
| `xrt` | Alveo FPGA (`hw`) **or** Vitis hardware emulation (`hw_emu`) + a built `.xclbin` | The only backend that runs on real silicon. |

## Inference API demo — [`infer.py`](infer.py)

`infer.py` is the reference for the **Level-3 Inference API**
(`vten.InferenceSession` / `vten.InferenceModule`), which lets you drive a
verified kernel from plain Python — passing `torch.Tensor` inputs and getting
device-resident `Tensor` outputs back, PyTorch-eager style.

It demonstrates, against `MmLoopbackKernel`:

1. `InferenceSession(...).run(MMLoopbackKernel, inputs={"data_in": x}, verify=True, N=...)`
2. `session.run_pipeline(...)` for a sequential chain
3. a small `InferenceModule` subclass used like an `nn.Module`
4. `session.cleanup()`

The script defaults to the **`cpu` backend** (no build, no FPGA) so it is
runnable out of the box, and documents how to point it at `xrt` / `hw_emu` once
you have built the xclbin. See the docstring at the top of
[`infer.py`](infer.py) for exact prerequisites.

## See also

- [`../README.md`](../README.md) — the examples index & feature→example map.
- [`../../docs/kernel_guide.md`](../../docs/kernel_guide.md) — `kernel_spec.yaml`, memory regions, `auto_bind`, array interfaces, and the DUT.
- [`../../docs/testing_guide.md`](../../docs/testing_guide.md) — `TestScenario` and the verification workflow.
- [`../../docs/cli_reference.md`](../../docs/cli_reference.md) — full CLI reference.
- [`../../docs/composite_guide.md`](../../docs/composite_guide.md) — composing multiple IPs (see also [`../scale_add`](../scale_add/README.md)).
