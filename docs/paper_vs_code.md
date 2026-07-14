# Paper ↔ Code Reconciliation

A maintainer-facing note that maps the DAC paper's terminology and claims onto the
current vTen implementation, and notes documentation that still needs follow-up.

> **The source code is authoritative.** Where the paper, `README.md`, or any
> earlier design note disagrees with the source, follow the source. This note
> exists so future doc edits do not re-introduce outdated terminology.

See also: [architecture.md](architecture.md) ·
[cli_reference.md](cli_reference.md) ·
[testing_guide.md](testing_guide.md) ·
[kernel_guide.md](kernel_guide.md)

---

## Contents

- [1. DSL op names](#1-dsl-op-names)
- [2. "8-stage" vs. ~10 internal stages](#2-8-stage-vs-10-internal-stages)
- [3. Kernel Template / Kernel Object / Binding Table](#3-kernel-template--kernel-object--binding-table)
- [4. Transport mechanisms vs. shipped backends](#4-transport-mechanisms-vs-shipped-backends)
- [5. Known paper ↔ code drift](#5-known-paper--code-drift)
- [6. Evaluation gap](#6-evaluation-gap)

---

## 1. DSL op names

The paper names the tensor-transfer operations `write_tensor` / `read_tensor`.
The implementation renamed these to `push_tensor` / `pull_tensor` — the
memory-mapped case made "write/read" ambiguous (host↔memory vs.
accelerator↔memory), so a two-level Data/Control classification was adopted.
The control-plane ops kept their names.

| Paper op | Code op | Location |
|----------|---------|----------|
| `write_tensor` | **`push_tensor`** | [vten/runtime/context.py](../vten/runtime/context.py) · `OpKind.PUSH_TENSOR` |
| `read_tensor` | **`pull_tensor`** | [vten/runtime/context.py](../vten/runtime/context.py) · `OpKind.PULL_TENSOR` |
| `write_register` | `write_register` | `OpKind.WRITE_REGISTER` |
| `read_register` | `read_register` | `OpKind.READ_REGISTER` |
| `poll_register` | `poll_register` | `OpKind.POLL_REGISTER` |
| `barrier` | `barrier` | `OpKind.BARRIER` |

`OpKind` (record phase) and `OpCode` (SHM IR) live in
[vten/spec/models.py](../vten/spec/models.py). The record-phase `push_tensor`
lowers to `LOAD + PUSH` opcodes (and `pull_tensor` to `PULL + STORE`); the code
also adds `configure` (auto-bind register writes) which the paper's op table
does not list.

---

## 2. "8-stage" vs. ~10 internal stages

The paper, `README.md`, and earlier design notes all describe an **"8-stage
compile pipeline."** That is a *conceptual* summary. The real orchestrator,
[vten/runtime/engine.py](../vten/runtime/engine.py), enumerates **Stages 0–9**
(ten numbered stages, though not all are heavyweight, and Stage 7's
SHM-packing/backend split moved out to `backend/sim/`).

Correspondence (conceptual 8 → implementation 0–9):

| Impl stage (engine.py) | Conceptual role |
|------------------------|-----------------|
| Stage 0 — Flatten / wrap | Composite flatten (paper's "Stage 0: Composite Flatten") |
| Stage 1 — Parameter resolution | Parameter resolution |
| Stage 2 — Shape resolution & validation | Shape validation |
| Stage 3 — Direction refinement | (finer split; folded into resolution conceptually) |
| Stage 4 — Tensor serialization | Tensor serialization |
| Stage 5 — Probe golden serialization | (probe support; not in the 8-stage summary) |
| Stage 6 — Address allocation | Address allocation |
| Stage 7 — auto_bind resolution | `auto_bind` resolution |
| Stage 8 — IR lowering → Command[] | IR lowering |
| Stage 9 — BFM config synthesis | BFM config synthesis |

An older design diagram placed SHM packing as an in-engine "Stage 7"; in the
code the SHM packing + backend handshake lives in `vten/backend/sim/`
(`SimBackend`), not inside `RuntimeEngine`. Treat "8-stage" as a high-level
shorthand for "the ~10-stage `RuntimeEngine.compile()` pipeline."

**Action:** when rewriting user docs, either say "10-stage" and match
`engine.py`, or keep "8-stage" only as a high-level phrase and link to
`engine.py` for the authoritative stage list.

---

## 3. Kernel Template / Kernel Object / Binding Table

Paper §3.3 concepts and where they live in code:

- **Kernel Template**: the declarative class with `Tensor` descriptors. In code,
  this is `class Kernel` in [vten/kernel/base.py](../vten/kernel/base.py), which
  uses `__init_subclass__` to collect tensor descriptors.
- **Kernel Object**: an instantiated, parameter-resolved kernel. In code, this is
  `KernelInstance` in [vten/runtime/kernel_view.py](../vten/runtime/kernel_view.py),
  produced by `ExecutionContext.instantiate(...)`.
- **Binding Table**: the resolved register↔value mapping. In code, this is
  `RegisterBindingEntry` plus `resolve_registers()` in
  [vten/runtime/binder.py](../vten/runtime/binder.py), covering `auto_bind` values
  and param-name matching.

Related supporting types: `ExposedTensor` / flattened kernel views also in
[vten/runtime/kernel_view.py](../vten/runtime/kernel_view.py).

---

## 4. Transport mechanisms vs. shipped backends

Paper §4.2 discusses **five transport mechanisms**. Only some are *shipped
backends*; the rest are comparison points in the evaluation, not code paths in
this repo.

Status by transport:

- **Native C++**: comparison baseline only; no backend is shipped.
- **vTen Pure C++**: shipped as the `verilator` backend
  ([vten/backend/verilator.py](../vten/backend/verilator.py)). DPI is resolved via
  `verilated_dpi.h` in
  [vten/sv/vten_shm_bridge_verilator.cpp](../vten/sv/vten_shm_bridge_verilator.cpp).
- **vTen SV-DPI**: shipped as the `xsim` backend
  ([vten/backend/xsim.py](../vten/backend/xsim.py)), with the DPI-C bridge in
  [vten/sv/vten_shm_bridge.c](../vten/sv/vten_shm_bridge.c).
- **File I/O**: comparison point only; no backend is shipped.
- **VPI / Cocotb**: comparison baseline only; no backend is shipped.

Backends that exist in the registry but are **not** paper §4.2 transports:

- **`xrt`** ([vten/backend/xrt/](../vten/backend/xrt)): real-FPGA / `hw_emu`
  execution via XRT. It interprets the IR directly rather than using SHM.
- **`cpu`** ([vten/backend/cpu.py](../vten/backend/cpu.py)): reference-model
  execution. It runs the kernel's `forward()` with no RTL or FPGA.

The authoritative backend list is `_BACKEND_MAP` in
[vten/backend/registry.py](../vten/backend/registry.py): `xsim`, `verilator`,
`xrt`, `cpu`.

---

## 5. Known paper ↔ code drift

A few APIs and workflows described by the paper (and by earlier, unshipped
design notes) diverged from what the code actually does. The code is
authoritative in every case below.

The main differences are:

- Kernel entry point: scenarios are declarative; the executable DSL lives in the
  kernel's `run(ctx)`.
- Batch/run API: `InferenceSession` / `InferenceModule` replaced earlier
  `run_kernel()` / `KernelExecutor` sketches.
- RTL detection CLI: `vten spec --detect` was never implemented.

- **Kernel entry point.** Earlier design notes showed `forward(self)` and a
  `TestScenario.run(self, ctx, cfg)` method with inline `ctx.instantiate(...)` /
  `ctx.run(verify=True)`. Current code uses
  `forward(self, **inputs) -> dict[str, torch.Tensor]`
  ([vten/kernel/base.py](../vten/kernel/base.py)). `TestScenario` is declarative
  (`kernel` / `configs` / `probes` / `seed`); the DSL protocol lives in the
  **kernel's** `run(ctx)` and is invoked by
  [`execute_batch`](../vten/execution.py). The scenario-level `run(self, ctx, cfg)`
  shown in the older examples is **not** called by the CLI run path.
- **Batch/run API.** Earlier (unshipped) design notes documented `run_kernel()`
  and `KernelExecutor`. Both were removed; the code replaced them with
  `InferenceSession` / `InferenceModule` in
  [vten/inference.py](../vten/inference.py).
- **RTL detection CLI.** Earlier design notes described
  `vten spec --detect rtl/....sv`. That command was never implemented; there is
  no `spec` subcommand in [vten/cli/main.py](../vten/cli/main.py). Do not
  document it as a supported command.

When user-facing docs are the target, prefer [cli_reference.md](cli_reference.md)
and [testing_guide.md](testing_guide.md), which are derived from source.

---

## 6. Evaluation gap

The paper's headline evaluation is **not reproducible in-repo**:

- The **3D U-Net NPU** design-under-test is modeled in test fixtures
  ([tests/fixtures/npu_3d.py](../tests/fixtures/npu_3d.py)), but the full NPU RTL
  and the end-to-end U-Net run are not shipped as a runnable example.
- The **Cocotb baseline** used for comparison is not present — there is no
  Cocotb/VPI backend in the repo (see §4).
- The reported metrics — roughly **2×** speedup, **60.3%** LOC reduction
  (**261 vs. 658** lines), and **92.2%** effort reduction (**28 vs. 360**) — are
  paper results and have **no reproduction harness** in this repository. Do not
  cite them as if `vten report` or the examples produce them.

The runnable examples ([examples/passthrough](../examples/passthrough),
[examples/mm_loopback](../examples/mm_loopback),
[examples/scale_add](../examples/scale_add)) exercise the framework end-to-end
but are much smaller than the paper's U-Net case study.
