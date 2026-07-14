# Kernel Guide

How to write a single **Kernel** — the verification unit for one RTL module (your DUT).

A Kernel binds a PyTorch tensor world (shapes, dtypes, a golden reference) to a
concrete RTL interface (AXI4-Stream / AXI4 / AXI4-Lite) described in a
`kernel_spec.yaml`. You never edit the RTL source — all binding lives in the
spec. This is vTen's **Zero RTL Intrusion** principle.

See also: [architecture.md](architecture.md) for the compile pipeline,
[testing_guide.md](testing_guide.md) for writing `TestScenario`s that run your
kernel, [cli_reference.md](cli_reference.md) for `vten build` / `vten run`, and
[composite_guide.md](composite_guide.md) for wiring multiple kernels together.

---

## The two files you write

For a kernel named `passthrough`, you author exactly two files under
`kernels/passthrough/`:

```
kernels/passthrough/
├── kernel_spec.yaml         # Interface spec — how tensors bind to RTL ports
└── passthrough_kernel.py    # Kernel class — tensors, golden model, protocol
```

- **`kernel_spec.yaml`** describes the DUT's ports: which RTL port, which
  protocol, how tensor elements pack onto the bus, what registers exist.
- **`<name>_kernel.py`** declares the tensors, the golden reference
  `forward()`, and the DSL `run(ctx)` protocol that drives the DUT.

The RTL itself (`rtl/passthrough.sv`) is referenced by `rtl_top:` and is never
modified.

---

## The canonical example: passthrough (AXI4-Stream)

`passthrough` is a streaming identity DUT: bytes go in on `s_axis`, the same
bytes come out on `m_axis`.

### `kernels/passthrough/kernel_spec.yaml`

```yaml
kernel: passthrough
rtl_top: rtl/passthrough.sv

interfaces:
  input_stream:
    rtl_port: s_axis
    protocol: axi4_stream
    tensor: data_in
    packing:
      element_width: 8
      elements_per_beat: 32
      bit_order: lsb_first

  output_stream:
    rtl_port: m_axis
    protocol: axi4_stream
    tensor: data_out
    packing:
      element_width: 8
      elements_per_beat: 32
      bit_order: lsb_first
```

- `kernel` — the kernel name (must match the directory).
- `rtl_top` — path to the RTL top, resolved relative to the project root.
- `interfaces` — a named map of DUT interfaces. Names (`input_stream`,
  `output_stream`) are *your* labels; the `.py` tensors reference them by name.
- `rtl_port` — the actual port prefix in the RTL (`s_axis`, `m_axis`).
- `tensor` — which Python tensor drives/reads this interface.
- `packing` — how tensor elements map onto AXI beats (see
  [Packing basics](#packing--layout-basics)).

### `kernels/passthrough/passthrough_kernel.py`

```python
import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor


class PassthroughKernel(Kernel):
    spec = "kernels/passthrough/kernel_spec.yaml"

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream",
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="output_stream",
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        return {"data_out": self.data_in.data.clone()}

    def run(self, ctx) -> None:
        h_push = ctx.push_tensor(self.data_in)
        ctx.pull_tensor(self.data_out, dep=h_push)
```

That's a complete, runnable kernel. The four moving parts:

| Part | Role |
|------|------|
| `spec = "..."` | Path to the `kernel_spec.yaml` (relative to project root). |
| `Tensor(...)` class attributes | Declare the data flowing in/out and which interface each binds to. |
| `generate_inputs(self, seed)` | Fill input tensors' `.data` with test data. |
| `forward(self, **inputs)` | Golden reference — what the DUT *should* produce. |
| `run(self, ctx)` | The DSL protocol — how to actually drive the DUT. |

Anatomy of the class body:

- `spec` and `default_params` are class-level configuration.
- Each `Tensor(...)` is a **descriptor**: on subclassing, vTen records it in
  `_tensor_descriptors` and sets its `.name` to the attribute name
  (`data_in`, `data_out`). See [vten/kernel/base.py](../vten/kernel/base.py).
- `register("...")` handles (shown later) declare AXI4-Lite control interfaces.

---

## Declaring tensors & interfaces

A `Tensor` is declared with a shape, dtype, interface name, and optional
direction ([vten/kernel/tensor.py](../vten/kernel/tensor.py)):

```python
Tensor(shape, dtype, interface, direction=None)
```

- **`shape`** — a tuple of ints and/or `"${PARAM}"` strings. Parametric dims are
  resolved from params at instantiation. `("${N}",)` is a length-`N` 1-D tensor;
  `("${OUT_CH}", "${H}", "${W}")` is 3-D.
- **`dtype`** — a `torch.dtype`. Common choices: `torch.int8`, `torch.uint8`,
  `torch.int16`, `torch.int32`, `torch.float32`. This is the *logical*
  (algorithmic) dtype; `packing` in the spec controls the physical bus layout.
- **`interface`** — the key in `kernel_spec.yaml`'s `interfaces:` map. This is
  the wiring: the tensor flows over the interface whose `tensor:` points back at
  it.
- **`direction`** — a `Direction` enum value. Import from
  `vten.spec.models`:

  ```python
  from vten.spec.models import Direction
  # Direction.HOST_TO_DEV   host provides data → DUT consumes (an input)
  # Direction.DEV_TO_HOST   DUT produces data → host reads (an output)
  # Direction.BIDIRECTIONAL
  ```

  If omitted (`direction=None`), it is inferred from the protocol/role at
  compile time — for a plain AXI4-Stream slave input this defaults to
  `HOST_TO_DEV`. For memory-mapped kernels, set it explicitly (see mm_loopback
  below) so the framework knows which tensors to read back and verify.

The spec side must agree: every `interface` you reference from a tensor must
exist in `interfaces:`, and that interface's `tensor:` (or `tensors:`) must name
the tensor back. A mismatch surfaces as a *missing interface* error at build
time.

---

## The `forward()` golden model

`forward()` is your bit-exact reference implementation. vTen calls it to compute
what the DUT *should* output, then compares against what the RTL actually
produced.

```python
def forward(self, **inputs) -> dict[str, torch.Tensor]:
    return {"data_out": self.data_in.data.clone()}
```

Rules:

- **Signature is `forward(self, **inputs)`** and it must return a
  `dict[str, torch.Tensor]` **keyed by output tensor name**. The key `"data_out"`
  must match the `data_out` tensor attribute.
- `inputs` is `{input_tensor_name: data}`. You can read inputs either from
  `inputs` or directly off `self.<tensor>.data`. The common idiom handles both:

  ```python
  def forward(self, **inputs):
      data = inputs.get("data_in", self.data_in.data)
      ...
  ```

  Reading from `inputs` first matters for composites, where an upstream kernel
  feeds this one (see [composite_guide.md](composite_guide.md)).
- Model the RTL's **exact** arithmetic, including width and saturation. The
  `scale` kernel multiplies int8 by a factor with signed saturation — note it
  widens to `int16` first, then clamps back:

  ```python
  def forward(self, **inputs):
      data = inputs.get("data_in", self.data_in.data)
      x = data.to(torch.int16) * self.scale_factor
      return {"data_out": x.clamp(-128, 127).to(torch.int8)}
  ```

  If your golden doesn't saturate identically to the hardware, you'll get a
  *verification mismatch* even though the RTL is correct.

`generate_inputs()` is the companion: it fills input tensors before `forward()`
and `run()` execute. Use `fill_random(generator=rng)` with a seeded
`torch.Generator` for reproducibility — `fill_random` picks a sensible range for
the dtype automatically (e.g. `[-128, 127]` for int8, `[0, 255]` for uint8).

---

## The `run(ctx)` DSL protocol

`run(self, ctx)` records the sequence of operations that drives the DUT. `ctx`
is an `ExecutionContext` ([vten/runtime/context.py](../vten/runtime/context.py));
its methods **record** operations (they don't execute immediately). The
recorded graph is compiled and dispatched to a backend.

### DSL methods on `ctx`

| Method | Purpose |
|--------|---------|
| `ctx.push_tensor(tensor, dep=)` | Host → device transfer. Feeds `tensor.data` into the DUT (LOAD + PUSH). |
| `ctx.pull_tensor(tensor, dep=, chunks=)` | Device → host read. Captures DUT output into `tensor` (PULL + STORE). |
| `ctx.configure(kernel, dep=)` | Emit a WRITE_REG for **every** `auto_bind` register of the kernel. |
| `ctx.write_register(reg, fields, dep=)` | Write field values, e.g. `{"start": 1}`. `reg` is a `register(...)` handle. |
| `ctx.read_register(reg, field_name, dep=)` | Read one register field. |
| `ctx.poll_register(reg, field_name, expected=, dep=)` | Poll a field until it equals `expected` (default `1`). |
| `ctx.barrier()` | Global fence: everything before must complete before anything after. |

Each returns an `OperationHandle`. Pass a handle as `dep=` to a later call to
create an ordering dependency. Handles also support
`handle.add_commit_dependency(other)`.

### Two kinds of dependency

- **`dep=handle`** — an **issue-ordering (data) dependency**. It says "don't
  *issue* this operation until that one has been issued." Use it to express
  dataflow: push before pull, configure before start.
- **`add_commit_dependency(handle)`** — a **completion/interrupt-ordering**
  dependency. It says "this operation must not *commit* (finish) until that one
  has committed." Use it to model "the read isn't valid until the DUT signals
  done."

For passthrough, one data dependency is enough — the output is valid as soon as
the input has streamed through:

```python
def run(self, ctx) -> None:
    h_push = ctx.push_tensor(self.data_in)      # feed input
    ctx.pull_tensor(self.data_out, dep=h_push)  # read output after push issued
```

For a DUT with an explicit start/done handshake, you configure, start, poll, and
gate the pull on the done poll (see mm_loopback next).

---

## Memory-mapped kernels: registers & `auto_bind`

`mm_loopback` is an AXI4 memory-mapped DUT controlled over AXI4-Lite. It reads a
buffer from DDR via an AXI4 master, copies it, and writes it back — an identity
loopback. This example covers three things the streaming case didn't: **AXI4
interfaces + `memory_regions`**, **AXI4-Lite `registers`**, and **`auto_bind`**.

### `kernels/mm_loopback/kernel_spec.yaml`

```yaml
kernel: mm_loopback
rtl_top: rtl/mm_loopback_core.sv

memory_regions:
  ddr:
    base: 0x10000000
    size: 0x10000000
    alignment: 4096

interfaces:
  ctrl:
    rtl_port: s_axilite
    protocol: axi4_lite
    addr_width: 12
    data_width: 32
    generate_controller: true
    registers:
      - name: ctrl
        offset: 0x00
        access: rw
        fields: { start: "0:0" }
      - name: status
        offset: 0x04
        access: ro
        fields: { done: "0:0" }
      - name: src_addr_lo
        offset: 0x10
        access: rw
        auto_bind: { tensor: data_in, value: address, bits: "31:0" }
      - name: src_addr_hi
        offset: 0x14
        access: rw
        auto_bind: { tensor: data_in, value: address, bits: "63:32" }
      - name: dst_addr_lo
        offset: 0x18
        access: rw
        auto_bind: { tensor: data_out, value: address, bits: "31:0" }
      - name: dst_addr_hi
        offset: 0x1C
        access: rw
        auto_bind: { tensor: data_out, value: address, bits: "63:32" }
      - name: length
        offset: 0x20
        access: rw
        auto_bind: { tensor: data_in, value: size_beats }

  mem_in:
    rtl_port: m_axi_in
    protocol: axi4
    data_width: 256
    addr_width: 64
    tensor: data_in
    memory_region: ddr
    packing: { element_width: 8, elements_per_beat: 32 }

  mem_out:
    rtl_port: m_axi_out
    protocol: axi4
    data_width: 256
    addr_width: 64
    tensor: data_out
    memory_region: ddr
    packing: { element_width: 8, elements_per_beat: 32 }
```

**`memory_regions`** — required for AXI4 masters. Each region (here `ddr`)
declares a physical `base`, `size`, and `alignment`. AXI4 interfaces reference a
region via `memory_region: ddr`; the framework allocates each tensor's buffer
inside that region and hands the DUT the resulting address.

**AXI4-Lite `registers`** — a list under a `axi4_lite` interface. Each register
has:

- `name`, `offset` (byte offset in the register map).
- `access` — `rw` (read/write), `ro` (read-only), `wo`, or `w1c`.
- `fields` — bit-field decomposition as `{ name: "hi:lo" }`. `{ start: "0:0" }`
  is a 1-bit field at bit 0. Register width is inferred from the highest bit.
- `pulse: true` — the field auto-clears after one cycle (a self-clearing start
  strobe). Used in the `scale` kernel's `ctrl` register.
- `auto_bind` — see below.
- `width: N` shorthand — `{ name: foo, width: 8 }` is equivalent to
  `fields: { foo: "7:0" }`.

`generate_controller: true` tells vTen to synthesize the AXI4-Lite controller
logic for this interface. It is only valid on `axi4_lite` interfaces.

### `auto_bind` — addresses without manual writes

`auto_bind` populates a register from a computed value at compile time
([vten/spec/models.py](../vten/spec/models.py), `AutoBindSpec`), so you don't
write DMA addresses by hand. Fields:

- `tensor` — which tensor this value comes from.
- `value` — `address` (the tensor's allocated device address) or `size_beats`
  (its length in AXI beats).
- `bits` — which slice of the value goes into this register, e.g. `"31:0"` /
  `"63:32"` to split a 64-bit address across two 32-bit registers (the classic
  `_lo`/`_hi` split).
- `param` / `expr` — bind to a parameter or a computed expression instead.
- `offset` — a byte offset added to the address.

In the spec above, `src_addr_lo`/`src_addr_hi` capture the low/high 32 bits of
`data_in`'s address; `dst_addr_lo`/`dst_addr_hi` do the same for `data_out`;
`length` gets `data_in`'s size in beats. **`ctx.configure(self)` expands into one
WRITE_REG per `auto_bind` register** — so a single `configure` call programs all
of them.

### `kernels/mm_loopback/mm_loopback_kernel.py`

```python
import torch

from vten.kernel.base import Kernel
from vten.kernel.register import register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class MmLoopbackKernel(Kernel):
    spec = "kernels/mm_loopback/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",), dtype=torch.uint8,
        interface="mem_in", direction=Direction.HOST_TO_DEV,
    )
    data_out = Tensor(
        shape=("${N}",), dtype=torch.uint8,
        interface="mem_out", direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        h_push = ctx.push_tensor(self.data_in)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_push)

        h_cfg = ctx.configure(self, dep=h_push)
        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)
        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden reference: loopback is identity."""
        return {"data_out": self.data_in.data.clone()}
```

Two new pieces vs. passthrough:

1. **`ctrl = register("ctrl")`** declares a handle to the AXI4-Lite `ctrl`
   interface. Its argument is the interface name in the spec. `register` comes
   from `vten.kernel.register` (also re-exported as `vten.register`). You pass
   the handle to `write_register` / `poll_register`.

2. **The start/done handshake** in `run`:
   - `configure(self, dep=h_push)` programs the address & length registers
     (after the input buffer exists).
   - `write_register(self.ctrl, {"start": 1}, dep=h_cfg)` kicks the DUT.
   - `poll_register(self.ctrl, "done", dep=h_start)` waits for completion. The
     field name `"done"` matches the `status` register's `{ done: "0:0" }`.
   - `h_pull.add_commit_dependency(h_poll)` makes the read-back *commit* only
     after `done` is observed — the output isn't valid until the DUT finishes.

Note the ordering intent: the pull is *issued* early (`dep=h_push`) so the
device-to-host path is set up, but it *commits* late (after `h_poll`). That
separation — issue order via `dep`, completion order via commit dependency — is
the core of the DSL.

---

## Parameters

Kernels are parametric. Parameters resolve `${PARAM}` shapes and feed
`forward()` / registers.

### `default_params`

A class-level dict of defaults:

```python
class ScaleKernel(Kernel):
    default_params = {"N": 1024, "scale_factor": 1}
```

Resolved params become instance attributes, so `forward()` can read
`self.scale_factor`, `self.N`, etc. A `TestScenario` config (or
`ctx.instantiate(Kernel, N=32)`) overrides these per run.

### `${PARAM}` shapes

Any shape dim written as a `"${NAME}"` string is resolved from the parameter
namespace at instantiation. `Tensor(shape=("${N}",), ...)` with `N=1024`
resolves to a length-1024 tensor.

### `compute_derived_params()`

Override this to compute params that depend on other params — e.g. a length in
beats derived from `N`:

```python
def compute_derived_params(self) -> dict:
    N = getattr(self, "N", 1024)
    return {"length": N // 32}
```

It runs after base params are set as instance attributes and its returned dict
is merged into the namespace, so derived values are available to shapes,
`auto_bind` `param:`/`expr:`, and `forward()`. Use it whenever a value is a pure
function of the base params rather than something the user sets directly.

### Params → registers (`runtime_params`)

A spec can map a param straight onto a register via a top-level `runtime_params`
block (from `scale`'s spec):

```yaml
runtime_params:
  scale_factor:
    default: 2
    register: ctrl.scale_factor
  length:
    register: ctrl.length
```

`ctx.configure(self)` writes these register values in addition to `auto_bind`
registers, so `scale`'s `run()` doesn't need explicit `write_register` calls for
`scale_factor` or `length` — a single `configure` covers both param-mapped and
auto-bound registers.

---

## Packing & layout basics

`packing` in the spec describes how logical tensor elements map onto physical
AXI beats ([vten/spec/models.py](../vten/spec/models.py), `PackingScheme`):

```yaml
packing:
  element_width: 8       # bits per logical element
  elements_per_beat: 32  # elements packed into one bus beat
  bit_order: lsb_first   # first element in the least-significant bits
  byte_order: little     # endianness within a beat
  alignment: packed      # packed = no per-element padding
```

The key relationship:

```
bus_width = element_width × elements_per_beat
```

So `8 × 32 = 256` bits — matching the AXI4 `data_width: 256` in mm_loopback. For
AXI4 the parser **enforces** `bus_width ≤ data_width`: exceeding it is an error;
being smaller warns that upper bits are zero-padded. For AXI4-Stream, if you omit
`data_width` it's derived from the packing; the `beat13` example uses
`element_width: 8, elements_per_beat: 13` with an explicit `data_width: 104`
(a 13-byte, non-power-of-two beat).

Other knobs:
- `bit_order: lsb_first | msb_first` — where element 0 sits within a beat.
- `alignment: packed` (dense) vs. per-element byte alignment.
- `mode: custom` with a `fields:` list for non-standard bit layouts.

**Layout hook.** If your tensor's declared shape is *logical* (algorithmic) and
differs from the physical HW/DDR layout, define a `layout_<tensorname>(self,
data)` method on the kernel. When present, the framework treats the declared
shape as logical and auto-calls `layout_<name>()` to produce the physical
buffer before serialization; otherwise the declared shape is taken as physical
and data is serialized as-is (see the note at the top of
[vten/kernel/tensor.py](../vten/kernel/tensor.py)). Most simple kernels don't
need it — packing alone handles element-to-beat mapping.

---

## Running your kernel

You drive a kernel through a `TestScenario` (see
[testing_guide.md](testing_guide.md)) and the CLI (see
[cli_reference.md](cli_reference.md)). The minimal scenario just names the
kernel:

```python
from vten import TestScenario

class TestPassthrough(TestScenario):
    kernel = "passthrough"
```

Then:

```bash
vten build --kernel passthrough
vten run --kernel passthrough --test TestPassthrough --verify
```

With `--verify`, vTen compares each `DEV_TO_HOST` tensor against your
`forward()` golden and reports pass/fail per tensor.

---

## Troubleshooting

**Verification mismatch (`max_diff > 0`).** The RTL output disagrees with
`forward()`. Most often the golden model doesn't match the hardware's arithmetic
— check width promotion and saturation (e.g. int8 multiply must widen to int16
then clamp to `[-128, 127]`), and signed/unsigned handling of register values
(the `offset` kernel converts a uint8 register value ≥ 128 to its signed
equivalent). Also confirm `generate_inputs()` and `forward()` use the same
input data. A failing run writes `results/<kernel>/<Test>/mismatches.jsonl` with
the diverging elements.

**Shape mismatch.** The resolved tensor shape doesn't match what the DUT
produced/consumed. Check that every `${PARAM}` in a shape has a value (in
`default_params`, `compute_derived_params()`, or the config), and that the
element count squares with `length`/`size_beats` registers. `Tensor` raises
`"shape not resolved"` if a param is missing at instantiation.

**Missing interface.** A tensor's `interface="..."` names an interface not in
the spec, or an interface's `tensor:` names a tensor that doesn't exist. The
names must match on both sides. Also ensure the spec path in `spec = "..."` is
correct (relative to the project root) and the file parses — `kernel`,
`rtl_top`, and a non-empty `interfaces` block are all required.

**Packing width mismatch.** For AXI4, `element_width × elements_per_beat` must
not exceed `data_width` (hard error) and should equal it to avoid zero-padding
(warning). Recompute `bus_width` and adjust `elements_per_beat`, `element_width`,
or `data_width` so they agree. `tensor:` and `tensors:` are mutually exclusive
on one interface — set only one.

**`run()` / `forward()` not implemented.** Both raise `NotImplementedError` by
default. Every runnable kernel must override `forward()` (for verification) and
`run()` (for the protocol). If a kernel is only ever used as a composite
sub-kernel with a pre-existing internal wire, `generate_inputs` can be
auto-supplied by the composite (see [composite_guide.md](composite_guide.md)),
but a standalone kernel needs its own.
