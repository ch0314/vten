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

- [`../../docs/kernel_guide.md`](../../docs/kernel_guide.md) — `kernel_spec.yaml`, memory regions, `auto_bind`, and the DUT.
- [`../../docs/testing_guide.md`](../../docs/testing_guide.md) — `TestScenario` and the verification workflow.
- [`../../docs/cli_reference.md`](../../docs/cli_reference.md) — full CLI reference.
- [`../../docs/composite_guide.md`](../../docs/composite_guide.md) — composing multiple IPs (see also [`../scale_add`](../scale_add/README.md)).
