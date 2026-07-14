# Paper ↔ Code Reconciliation

A maintainer-facing note mapping the DAC paper's terminology and claims onto the
actual vTen implementation, and flagging documentation drift to fix later.

> **Ground truth is the code.** Where the paper, `README.md`, `CLAUDE.md`, or the
> `specs/` disagree with the source, the source wins. This note exists so future
> doc edits don't re-introduce stale terminology.

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
- [5. Stale specs to fix later](#5-stale-specs-to-fix-later)
- [6. Evaluation gap](#6-evaluation-gap)

---

## 1. DSL op names

The paper names the tensor-transfer operations `write_tensor` / `read_tensor`.
The implementation renamed these to `push_tensor` / `pull_tensor` — the
memory-mapped case made "write/read" ambiguous (host↔memory vs.
accelerator↔memory), so a two-level Data/Control classification was adopted
(rationale in `specs/01_kernel_and_dsl.md` §3.1). The control-plane ops kept
their names.

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

The paper, `README.md`, and `CLAUDE.md` all describe an **"8-stage compile
pipeline."** That is a *conceptual* summary. The real orchestrator,
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

The old `CLAUDE.md` diagram places SHM packing as an in-engine "Stage 7"; in the
code the SHM packing + backend handshake lives in `vten/backend/sim/`
(`SimBackend`), not inside `RuntimeEngine`. Treat "8-stage" as marketing
shorthand for "the ~10-stage `RuntimeEngine.compile()` pipeline."

**Action:** when rewriting user docs, either say "10-stage" and match
`engine.py`, or keep "8-stage" only as a high-level phrase and link to
`engine.py` for the authoritative stage list.

---

## 3. Kernel Template / Kernel Object / Binding Table

Paper §3.3 concepts and where they live in code:

| Paper concept | Code location |
|---------------|---------------|
| **Kernel Template** — the declarative class with `Tensor` descriptors | [vten/kernel/base.py](../vten/kernel/base.py) — `class Kernel` (uses `__init_subclass__` to collect tensor descriptors). |
| **Kernel Object** — an instantiated, parameter-resolved kernel | `KernelInstance` in [vten/runtime/kernel_view.py](../vten/runtime/kernel_view.py); produced by `ExecutionContext.instantiate(...)`. |
| **Binding Table** — the resolved register↔value mapping | `RegisterBindingEntry` + `resolve_registers()` in [vten/runtime/binder.py](../vten/runtime/binder.py) (auto_bind values and param-name matching). |

Related supporting types: `ExposedTensor` / flattened kernel views also in
[vten/runtime/kernel_view.py](../vten/runtime/kernel_view.py).

---

## 4. Transport mechanisms vs. shipped backends

Paper §4.2 discusses **five transport mechanisms**. Only some are *shipped
backends*; the rest are comparison points in the evaluation, not code paths in
this repo.

| Paper transport | Shipped in repo? | Backend / notes |
|-----------------|------------------|-----------------|
| Native C++ | No | Comparison baseline only — no backend. |
| vTen Pure C++ | **Yes** | `verilator` backend ([vten/backend/verilator.py](../vten/backend/verilator.py)); DPI resolved via `verilated_dpi.h` in [vten/sv/vten_shm_bridge_verilator.cpp](../vten/sv/vten_shm_bridge_verilator.cpp). |
| vTen SV-DPI | **Yes** | `xsim` backend ([vten/backend/xsim.py](../vten/backend/xsim.py)); DPI-C bridge in [vten/sv/vten_shm_bridge.c](../vten/sv/vten_shm_bridge.c). |
| File I/O | No | Comparison point only — no backend. |
| VPI / Cocotb | No | Comparison baseline only — no backend. |

Backends that exist in the registry but are **not** paper §4.2 transports:

| Backend | Location | Purpose |
|---------|----------|---------|
| `xrt` | [vten/backend/xrt/](../vten/backend/xrt) | Real-FPGA / `hw_emu` execution via XRT (IR interpreted directly, not SHM). |
| `cpu` | [vten/backend/cpu.py](../vten/backend/cpu.py) | Runs the kernel's `forward()` as a reference model — no RTL, no FPGA. |

The authoritative backend list is `_BACKEND_MAP` in
[vten/backend/registry.py](../vten/backend/registry.py): `xsim`, `verilator`,
`xrt`, `cpu`.

---

## 5. Stale specs to fix later

The `specs/` are partially stale. Known drift (code is ground truth):

| Spec | Stale content | Current reality |
|------|---------------|-----------------|
| `specs/07_e2e_examples.md` | `forward(self)` signature; `TestScenario.run(self, ctx, cfg)` with inline `ctx.instantiate(...)` / `ctx.run(verify=True)`. | Code uses `forward(self, **inputs) -> dict[str, torch.Tensor]` ([vten/kernel/base.py](../vten/kernel/base.py)). `TestScenario` is declarative (`kernel`/`configs`/`probes`/`seed`); the DSL protocol lives in the **kernel's** `run(ctx)`, invoked by [`execute_batch`](../vten/execution.py). The scenario `run(self, ctx, cfg)` shown in old examples is **not** called by the CLI run path. |
| `specs/09_user_api.md` | Documents `run_kernel()` and `KernelExecutor`. | Both **removed**. Replaced by `InferenceSession` / `InferenceModule` in [vten/inference.py](../vten/inference.py). |
| `specs/06_codegen_and_cli.md` §4.2 | Documents `vten spec --detect rtl/....sv`. | **Unimplemented** — there is no `spec` subcommand in [vten/cli/main.py](../vten/cli/main.py). Do not document it. |

These are already partially captured in `CLAUDE.md` under "Known Spec ↔ Code
Divergences"; keep that table and this section in sync. When user-facing docs
are the target, prefer [cli_reference.md](cli_reference.md) and
[testing_guide.md](testing_guide.md), which are derived from source.

---

## 6. Evaluation gap

The paper's headline evaluation is **not reproducible in-repo**:

- The **3D U-Net NPU** design-under-test is analyzed in
  `specs/npu_3d_analysis.md` and modeled in test fixtures
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
