# Architecture

This guide explains how vTen works internally: the layers it is built from, how a
Python test scenario becomes a running simulation, and how the same program drives
both an RTL simulator and a real FPGA. It is written for developers who want to
extend vTen and for advanced users who want a precise mental model of what happens
between `ctx.run()` and a bit-exact verdict.

The other guides ([Kernel Guide](kernel_guide.md), [CompositeKernel Guide](composite_guide.md),
[Testing Guide](testing_guide.md), [CLI Reference](cli_reference.md)) link back here for
the underlying mechanics.

---

## 1. Overview & Motivation

Hardware accelerators are designed against a *tensor-level* mental model — "a `(N, C, H, W)`
feature map streams in, a `(N, K, H, W)` feature map streams out" — but they are *verified*
against a *signal-level* testbench: AXI handshakes, byte-packed bus beats, control-register
pokes, ready/valid backpressure, and cycle timing. Every project re-crosses this **semantic gap**
by hand, writing bespoke SystemVerilog drivers that translate tensors into wiggles and back.
That glue is where verification time goes, and where bugs hide.

vTen closes the gap with a **data-centric** approach. You describe *what data crosses which
interface in which direction* — as PyTorch tensors bound to named interfaces — and vTen owns
the translation down to the wire. The tensor is the unit of thought; the framework compiles it
into a bit-accurate bus transaction, drives it through a protocol-correct Bus Functional Model
(BFM), captures the response, and compares it against a Python golden reference computed by the
same kernel definition. The signal-level testbench is *generated*, not authored.

Concretely, a user writes three things:

- a **kernel** — tensors, a `forward()` golden reference, and a `run()` DSL sequence,
- an **interface spec** (`kernel_spec.yaml`) — how each tensor maps to a bus protocol and packing,
- a **test** — parameter configurations to sweep.

Everything else (serialization, addressing, IR, BFM configuration, the SystemVerilog harness,
and the host↔simulator transport) is derived.

> This mirrors the DAC paper's framing of vTen as a tensor-centric verification framework.
> Where the paper's naming and the code diverge, **this document follows the code** — see
> [§10, How this maps to the paper](#10-how-this-maps-to-the-paper).

---

## 2. The Three-Layer Architecture

vTen is organized into three layers with a strict, one-directional dependency: the Frontend
produces a program, the Runtime compiles it into a backend-agnostic IR, and the Backend
executes that IR against a concrete target.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  FRONTEND  —  what to verify (tensor-level)                                │
│                                                                            │
│   Kernel / CompositeKernel   Tensor descriptors   kernel_spec.yaml         │
│   forward()  (golden)        run(ctx)  (DSL)      register()  (control)    │
│        vten/kernel/          vten/dsl/            vten/spec/                │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     │  ctx records an Operation[] list
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RUNTIME  —  how to make it wire-accurate (backend-agnostic)              │
│                                                                            │
│   ExecutionContext ──► RuntimeEngine._compile_ir  (record-then-compile)    │
│     flatten · resolve params · resolve shapes · serialize · allocate addr  │
│     · bind registers · lower to Command[] IR · synthesize BFM configs      │
│        vten/runtime/                                                        │
│                                                                            │
│   Output:  Command[]  +  tensor bytes  +  BFMConfig[]  +  Binding Table    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     │  the SAME Command[] IR
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
┌────────────────────────────────┐   ┌────────────────────────────────────┐
│  BACKEND (sim)                 │   │  BACKEND (hw / reference)           │
│                                │   │                                     │
│  pack Command[] → SHM image    │   │  interpret Command[] → XRT API      │
│  POSIX SHM + 2 semaphores      │   │  (DMA + MMIO on real FPGA)          │
│    ▼                           │   │  ── or ──                           │
│  SystemVerilog scheduler + BFM │   │  run forward() as CPU reference     │
│  drives the DUT                │   │                                     │
│    xsim · verilator            │   │    xrt · cpu                        │
│    vten/backend/sim/, sv/      │   │    vten/backend/xrt/, cpu.py        │
└────────────────────────────────┘   └────────────────────────────────────┘
```

The pivot is the **Command[] IR**: a flat, serializable list of low-level operations produced
by the Runtime and consumed by every backend. A simulation backend packs it into a shared-memory
image that a SystemVerilog scheduler reads; a hardware backend interprets it directly into XRT
calls. The Frontend never knows which backend it is running against, and the same DSL program is
bit-for-bit valid on all of them.

---

## 3. The Record-then-Compile Execution Model

vTen uses a **two-pass** model: your `run(self, ctx)` method does not execute anything — it
*records* a program. Compilation and execution happen afterward, all at once.

**Pass 1 — Record.** Each DSL method on the [`ExecutionContext`](../vten/runtime/context.py)
(`ctx`) appends an `Operation` to a pending list and returns an `OperationHandle` you can pass as
a `dep=` to a later call. Nothing touches a bus. The DSL surface is:

| `ctx` method       | Records            | Notes                                              |
|--------------------|--------------------|----------------------------------------------------|
| `instantiate`      | (kernel instance)  | Eagerly resolves params/shapes; registers a kernel.|
| `push_tensor`      | `PUSH_TENSOR`      | Host → DUT. Lowers to `LOAD` + `PUSH`.             |
| `pull_tensor`      | `PULL_TENSOR`      | DUT → host. Lowers to `PULL` (+ `STORE`).          |
| `write_register`   | `WRITE_REGISTER`   | Write named control-register fields.               |
| `read_register`    | `READ_REGISTER`    | Read one field back.                               |
| `poll_register`    | `POLL_REGISTER`    | Spin until a field matches `expected`.             |
| `configure`        | `CONFIGURE`        | Emit all `auto_bind` register writes for a kernel. |
| `barrier`          | `BARRIER`          | Fence: everything before must retire before after. |
| `run(verify=)`     | —                  | Ends Pass 1, triggers Pass 2.                      |

> **Naming note.** The paper describes the data-movement primitives as `write_tensor` /
> `read_tensor`. The **code** uses `push_tensor` / `pull_tensor`, and those are the real,
> callable names — this guide uses them throughout.

**Pass 2 — Compile & execute.** [`ExecutionContext.run`](../vten/runtime/context.py) hands the
recorded `Operation[]` to a [`RuntimeEngine`](../vten/runtime/engine.py), which compiles it into
IR (§4), submits the result to the attached backend (§8), reads output tensors back, and — when
`verify=True` — checks every device→host tensor against the golden `forward()`.

Dependencies between operations are explicit rather than positional: `dep=h` means "wait for the
commands that handle `h` before issuing mine." Because the IR carries an explicit dependency graph
(not just program order), the backend's scheduler is free to run independent transactions
concurrently — for example, streaming a large input on one interface while polling a status
register on another.

Recording before compiling is what lets one program target many backends: the `Operation[]` list
is pure intent, with no addresses, byte layouts, or protocol details baked in yet.

---

## 4. The Compile Pipeline

Compilation is orchestrated by [`RuntimeEngine._compile_ir`](../vten/runtime/engine.py), which
runs a sequence of internally-numbered stages (Stage 0 through Stage 9 in the code) to turn the
`Operation[]` list plus the kernel spec into a `CompiledResult`. Conceptually the pipeline does
seven things; the code splits a few of them into sub-stages, which is why you will see ten
`Stage N:` log lines. Each conceptual step below names its code entry point.

**1. Flatten / wrap** — [`vten/runtime/flatten.py`](../vten/runtime/flatten.py)
A [`CompositeKernel`](../vten/kernel/composite.py) is flattened into a single
`FlattenedKernelView`: its sub-kernel instances become one interface namespace, `>>` connections
become internal wires, and any tensor not consumed by a connection is auto-exposed as a top-level
host↔device port. A plain `Kernel` is wrapped into the same view type so downstream stages are
uniform. Composites have no `kernel_spec.yaml` of their own; the top spec is synthesized from the
exposed sub-kernel interfaces.

**2. Resolve parameters & shapes** — [`vten/runtime/resolver.py`](../vten/runtime/resolver.py)
Parametric dimensions like `("${N}",)` are resolved against the merged parameter namespace
(project params + runtime params + `compute_derived_params()`), then declared shapes are
validated against any attached data, and `>>` connection endpoints are checked for element-count
compatibility. (In the code this is Stages 1–2, plus a Stage 3 that refines each exposed tensor's
direction from how the DSL actually used it — `push_tensor` ⇒ host→device, `pull_tensor` ⇒
device→host.)

**3. Serialize host→device tensors** — [`vten/runtime/serializer.py`](../vten/runtime/serializer.py)
Each host→device tensor is turned into the exact bytes that will cross the bus:
- if the owning kernel defines a `layout_{name}()` hook, it is applied first
  ([`vten/runtime/layout.py`](../vten/runtime/layout.py)) to convert logical → physical layout;
- elements are quantized to the interface's `element_width`;
- elements are bit-packed into bus beats per the interface `packing` (respecting `bit_order`,
  `byte_order`, `elements_per_beat`, and `bus_width`).

Device→host tensors are not serialized — only their byte size is computed so a receive buffer can
be sized. Array and multi-port interfaces are split into per-port sub-buffers here. (Code Stage 4,
with a Stage 5 that serializes golden data for internal probe points.)

**4. Allocate physical addresses** — [`vten/runtime/address.py`](../vten/runtime/address.py)
For memory-mapped (AXI4) interfaces, each tensor buffer is placed into its declared memory region
with alignment and overflow checking, yielding a physical address. Pure streaming (AXI4-Stream)
tensors need no address. (Code Stage 6.)

**5. Resolve control registers** — [`vten/runtime/binder.py`](../vten/runtime/binder.py)
`auto_bind` register specs are evaluated into concrete values — a buffer's base address, its size,
a parameter, or an arithmetic expression over those — and registers whose name matches a parameter
get that parameter's value. This is the table that a `configure` op will later write. (Code
Stage 7.)

**6. Lower to `Command[]` IR** — [`vten/runtime/ir.py`](../vten/runtime/ir.py)
The `IRLowering` class walks the `Operation[]` list and emits low-level `Command`s (§5). Each op
expands to one or more commands: `push_tensor` → `LOAD` + `PUSH`, `pull_tensor` → `PULL`
(+ `STORE` for memory-mapped), `configure` → a batch of `WRITE_REG`. Operation-level `dep` handles
are resolved into concrete command IDs, and stable buffer IDs and interface IDs are assigned. (Code
Stage 8.)

**7. Synthesize BFM configurations** — [`vten/runtime/bfm_config.py`](../vten/runtime/bfm_config.py)
For each interface that carries traffic, a `BFMConfig` describes the BFM the harness must
instantiate: protocol, data/address width, master/slave role, and address ranges. This is what
tells the generated testbench which drivers to build and how to parameterize them. (Code Stage 9.)

The result of all of this is a [`CompiledResult`](../vten/runtime/engine.py): the `Command[]` IR,
the serialized `tensor_data` keyed by buffer ID, the `BFMConfig[]` list, the flattened view, and
the pieces of the Binding Table (§6). SHM packing is deliberately *not* done here — it belongs to
the simulation backend, so that a hardware backend can consume the same `CompiledResult` without
ever building an SHM image.

Multi-config sweeps compile each configuration group through the same stages and merge the results,
inserting a `BARRIER` command between groups; see
[`RuntimeEngine.compile_multi`](../vten/runtime/engine.py).

---

## 5. Execution IR & OpCodes

The IR is a flat list of [`Command`](../vten/runtime/command.py) dataclasses. A `Command` is a
fixed-shape record — opcode, IDs, protocol/role, an optional physical address and size, register
fields, and a dependency list — designed to pack into a 64-byte binary slot without reinterpretation.

The opcode set is small and closed ([`OpCode`](../vten/spec/models.py)):

| OpCode      | Value | Meaning                                               |
|-------------|:-----:|-------------------------------------------------------|
| `LOAD`      |  1    | Stage host tensor bytes into a device/host buffer.    |
| `PUSH`      |  2    | Drive a buffer onto an interface (host → DUT).         |
| `PULL`      |  3    | Capture data from an interface (DUT → host).           |
| `STORE`     |  4    | Read a captured buffer back to the host.              |
| `WRITE_REG` |  5    | Write a control register.                             |
| `READ_REG`  |  6    | Read a control register.                              |
| `POLL_REG`  |  7    | Poll a register field until it matches (mask/expected).|
| `BARRIER`   |  8    | Global fence across all interfaces.                   |

The DSL-to-IR mapping is intentionally direct:

- `push_tensor` → `LOAD` (stage bytes) then `PUSH` (drive the bus), the `PUSH` depending on the
  `LOAD`.
- `pull_tensor` → `PULL`, plus a `STORE` for memory-mapped protocols (AXI4-Stream `PULL` captures
  directly and needs no separate `STORE`).
- `configure` → one `WRITE_REG` per resolved `auto_bind` register.

Each command carries its own `dep` list of command IDs, so the executing scheduler sees an explicit
dependency DAG rather than a linear instruction stream. Whether that DAG is executed by a hardware
scheduler in SystemVerilog or by a host-side interpreter in Python, the semantics are identical —
which is the whole point of having one IR.

---

## 6. The Binding Table

The IR describes *operations*; the **Binding Table** describes the *identifiers* those operations
reference and what each resolves to on a concrete target. It is not a single object but a set of
maps produced across the pipeline, all carried on the `CompiledResult`:

| Component            | Produced by                                          | Maps                                            |
|----------------------|------------------------------------------------------|-------------------------------------------------|
| **Buffer IDs**       | [`ir.py`](../vten/runtime/ir.py) (`_allocate_buffer_ids`) | tensor / port name → integer buffer ID      |
| **Physical addresses** | [`address.py`](../vten/runtime/address.py)         | buffer → physical/device address                |
| **Register values**  | [`binder.py`](../vten/runtime/binder.py)             | `auto_bind` register → resolved value           |
| **Interface ID map** | [`ir.py`](../vten/runtime/ir.py) (`_iface_id_map`)   | interface name → integer interface ID           |

Commands reference buffers and interfaces by these integer IDs, keeping each 64-byte slot compact
and name-free. The interface ID map is aligned with the spec's declaration order so that it matches
the BFM instantiation order in the generated testbench. Buffer IDs are stable across a batch (and
offset per config group in a multi-config sweep) so the host can read the right bytes back out
afterward.

The register-value component is the subtle one, and it is what makes the **same DSL run
unmodified on sim and hardware** — see §7.

---

## 7. Shared-Memory Transport & the Host↔Simulator Handshake

Simulation backends and the host communicate through a POSIX shared-memory segment plus two named
semaphores. The host (Python) packs the `Command[]` IR and tensor bytes into a binary image; the
generated SystemVerilog scheduler reads that image, drives the DUT through the BFMs, and writes
per-command stats and captured output back into the same segment.

**Binary contract** ([`vten/backend/sim/shm_constants.py`](../vten/backend/sim/shm_constants.py)):

- **Magic** `0x5654454E` ("VTEN"), **protocol version** `3`.
- A **256-byte control header** (offset 0): magic, version, host/backend status words, command and
  buffer counts, region offsets, error code / cmd-id / message, and flags.
- **64-byte command slots** — one per `Command`, in the command region.
- **32-byte stats slots** — one per command, written back by the scheduler.
- **24-byte buffer descriptors** and a 64-byte-aligned data region holding the tensor bytes.
- Two **named semaphores** per session: `h2b` (host→backend) and `b2h` (backend→host).

**Handshake** ([`vten/backend/sim/base.py`](../vten/backend/sim/base.py), `SimBackend`):

```
Host (Python)                              Backend (SV scheduler via DPI-C bridge)
─────────────                              ────────────────────────────────────────
[1] shm_open + write image
    sem_open(h2b, b2h)
    launch simulator process ───────────►  vten_shm_init(): attach SHM
                                    ◄─────  sem_post(b2h)  "ready", backend_status=IDLE
[3] sem_wait(b2h)
    host_status = CMD_READY
    sem_post(h2b)               ─────────►  read commands, drive BFMs → DUT
                                            write stats + captured outputs to SHM
                                    ◄─────  sem_post(b2h)  "done" / "error"
[5] sem_wait(b2h)
    read stats + outputs
    host_status = ACK
    ...
[6] host_status = SHUTDOWN
    sem_post(h2b)               ─────────►  exit
```

The session stays alive across batches: subsequent `execute()` calls update the SHM image in place
(growing it via `ftruncate` when needed) and re-signal `CMD_READY`, so a multi-config sweep does not
pay simulator-startup cost per configuration. While waiting, the host polls the stats region to
report live per-command progress and to produce a structured diagnostic if the backend stalls or
times out.

**Two transports, one image.** The
[`XsimBackend`](../vten/backend/xsim.py) links the shared-memory bridge into Vivado xsim through
**SV-DPI** (a C bridge, `vten_shm_bridge.c`), while the
[`VerilatorBackend`](../vten/backend/verilator.py) uses a **pure C++** bridge
(`vten_shm_bridge_verilator.cpp`). Both subclass the same `SimBackend` and speak the identical SHM
protocol; they differ only in how the simulator process is built and launched. The fixed
SystemVerilog library — scheduler, controller, and the AXI4-Stream / AXI4 / AXI4-Lite BFMs — lives
in [`vten/sv/`](../vten/sv/).

**Why registers are SHM offsets in sim.** In simulation, an `auto_bind` "buffer address" register
resolves to the buffer's **offset inside the SHM data region** — because that is where the buffer
physically lives for the simulator. On hardware, the very same `WRITE_REG` command is rewritten so
its value becomes the buffer's real device address: the XRT interpreter substitutes
`bo.address() + offset` when it writes an address-bound control register
([`vten/backend/xrt/interpreter.py`](../vten/backend/xrt/interpreter.py)). This **address
substitution at write time** is precisely what lets an unmodified DSL program — and an unmodified
`Command[]` IR — be correct in both worlds.

---

## 8. The Four Backends

All backends implement the same [`Backend`](../vten/backend/base.py) ABC (`execute(compiled)` →
`BackendResult`, plus `cleanup()`) and consume the same `CompiledResult`. They split into two
compile targets: `SIM` (xsim, verilator, cpu) and `HW` (xrt).

| Backend       | Target | Transport            | Fidelity                    | Use it when…                                                        |
|---------------|--------|----------------------|-----------------------------|---------------------------------------------------------------------|
| **xsim**      | SIM    | SV-DPI (C bridge)    | Cycle-accurate RTL (Vivado) | You need Vivado-grade simulation and have a Vivado install.         |
| **verilator** | SIM    | Pure C++ bridge      | Cycle-accurate RTL (OSS)    | You want fast, open-source cycle-accurate sim with no Vivado.       |
| **cpu**       | SIM    | none (in-process)    | Functional only             | You want a ~100× faster smoke test of the config/pipeline & golden. |
| **xrt**       | HW     | PCIe DMA + MMIO      | Real silicon                | You are running verified kernels on an actual FPGA (or `hw_emu`).   |

**xsim** ([`vten/backend/xsim.py`](../vten/backend/xsim.py)) — launches Vivado `xsim` against a
pre-elaborated snapshot, linking the DPI-C shared-memory bridge during elaboration. Supports batch
waveform dumps and an interactive `--gui` mode that keeps the session alive across simulator
restarts.

**verilator** ([`vten/backend/verilator.py`](../vten/backend/verilator.py)) — launches a standalone
`Vtb_top` C++ binary produced by the Verilator build pipeline. Same SHM handshake, no proprietary
tools.

**cpu** ([`vten/backend/cpu.py`](../vten/backend/cpu.py)) — runs no RTL at all. It executes the
kernel's `forward()` and returns those tensors directly as the "DUT" output, skipping serialization
and the SHM round-trip entirely (which is where the ~100× speedup comes from). With `--verify` it
always passes, since the DUT output *is* the golden output — its value is exercising the DSL, the
compile pipeline, shape resolution, and golden computation without hardware.

**xrt** ([`vten/backend/xrt/`](../vten/backend/xrt/)) — executes on a real FPGA over PCIe. Instead
of packing an SHM image, a `CommandInterpreter`
([`interpreter.py`](../vten/backend/xrt/interpreter.py)) walks the same `Command[]` IR and calls
XRT: `LOAD` → `bo.write`, `PUSH` → `bo.sync(TO_DEVICE)` (DMA), `PULL` → `bo.sync(FROM_DEVICE)`,
`STORE` → `bo.read`, `WRITE_REG`/`READ_REG`/`POLL_REG` → `ip.write_register`/`read_register` (MMIO),
`BARRIER` → a host-side fence. This is the second consumer of the IR referenced throughout this
guide.

The [Inference API](../vten/inference.py) (`InferenceSession`, `InferenceModule`) sits on top of the
xrt backend for eager, kernel-granular deployment of already-verified kernels.

---

## 9. Project Layout: `VTEN_ROOT` vs `PROJECT_ROOT`

vTen is a **library**, and your accelerator is a separate **project**. RTL sources are large and
tool-specific, so vTen never copies them into its own tree; instead it resolves everything against
two roots.

- **`VTEN_ROOT`** — the installed `vten` package (resolved via `import vten`). Home of the fixed
  SystemVerilog library ([`vten/sv/`](../vten/sv/)), the code-generation templates, and the runtime.
  You do not edit this.
- **`PROJECT_ROOT`** — the directory containing your `vten.toml`. Home of your `rtl/`, `kernels/`,
  IP definitions, and the generated `build/` and `results/` trees.

```
$VTEN_ROOT/                     ← pip install -e .   (the vten package)
    vten/sv/                    fixed SystemVerilog library + DPI/C++ bridges
    vten/templates/             Jinja2 testbench + build-script templates
    vten/runtime/, backend/     the compile pipeline and backends

$PROJECT_ROOT/                  ← where vten.toml lives
    vten.toml                   project config: parameters, backends, rtl/ip paths
    rtl/                        your RTL sources (large; never moved)
    ip/                         Vivado IP (.xci)
    build/                      project-level build artifacts
    kernels/
      my_accel/
        kernel_spec.yaml        interface spec  (must live in kernels/<name>/)
        my_accel_kernel.py      kernel: tensors + forward() + run()
        tests/                  TestScenario definitions
        build/                  per-kernel generated artifacts (SV, xsim.dir, shm)
    results/                    test results
```

All paths in `kernel_spec.yaml` (`rtl_top`) and `vten.toml` (`[rtl].sources`) are resolved relative
to `PROJECT_ROOT`; the SV library and templates are resolved relative to `VTEN_ROOT`. Nothing is
hardcoded to an absolute path. This is the "Multi-Directory Setup" model described in the project's
`CLAUDE.md`, and it is what lets several kernels share one large RTL tree while keeping each kernel's
generated build output isolated under `kernels/<name>/build/`.

---

## 10. How This Maps to the Paper

The DAC paper and this codebase describe the same system; a few names and counts differ. The code
is authoritative.

| Paper                                   | Code                                                                 |
|-----------------------------------------|----------------------------------------------------------------------|
| `write_tensor` / `read_tensor` DSL ops  | `push_tensor` / `pull_tensor` ([`context.py`](../vten/runtime/context.py)) |
| "8-stage compile pipeline"              | The same pipeline, split into ~10 internally-numbered stages (Stage 0–9) in [`engine.py`](../vten/runtime/engine.py); the paper's eight conceptual stages ≈ §4's seven steps here, with a few code sub-stages. |
| Tensor-centric, data-centric framing    | Exactly as implemented — the tensor is the unit of intent; the wire-level testbench is generated. |
| One program, sim and hardware           | The single `Command[]` IR (§5) driving both the SHM/SV path and the XRT interpreter; `auto_bind` address substitution (§7) reconciles the two. |

If you are reading the paper alongside the source, treat the paper for *intent* and this document
(plus the code it links) for *ground truth*.

---

## See Also

- [Kernel Guide](kernel_guide.md) — writing a `Kernel`, `kernel_spec.yaml`, and `run()`.
- [CompositeKernel Guide](composite_guide.md) — multi-IP composition and `>>` wiring.
- [Testing Guide](testing_guide.md) — `TestScenario`, config sweeps, and verification.
- [CLI Reference](cli_reference.md) — `vten init` / `build` / `run` / `report`.
