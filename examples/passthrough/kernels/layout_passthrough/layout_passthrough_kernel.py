"""layout_passthrough — demonstrates the layout_{name}() / unlayout_{name}() hook.

This kernel REUSES the already-passing physical DUT ``rtl/passthrough.sv`` (a
byte-verbatim AXI4-Stream copy). Nothing about the RTL changes; the whole point
is to show the *host-side* layout hook mechanism on a DUT we already trust.

Why verification still holds (symmetric round-trip)
---------------------------------------------------
The declared shape is the *logical* shape. When a ``layout_<tensor>()`` method
exists, vTen treats the declared shape as logical and auto-calls the hook to
produce the *physical* buffer before serialization (see
``vten/runtime/layout.py::apply_layout`` and the note atop
``vten/kernel/tensor.py``). On the output side it auto-calls
``unlayout_<tensor>()`` after deserialization
(``vten/runtime/output_reader.py``).

We use ``torch.flip`` along axis 0, which is its OWN inverse:

    logical x
      --layout_data_in-->  flip(x)                       (physical, sent to DUT)
      --DUT (byte-verbatim passthrough)-->  flip(x)      (physical, from DUT)
      --unlayout_data_out--> flip(flip(x)) == x          (logical, returned)

For the golden path, ``forward()`` receives the layout-applied input
(``vten/runtime/golden.py::run_forward`` applies ``layout_data_in`` before
calling ``forward``), so it returns ``flip(x)`` — the identity in *physical*
space. ``compute_golden_outputs`` then applies ``unlayout_data_out`` to the
golden as well (``golden.py`` line ~137), yielding ``x``. Both the HW output and
the golden are un-layouted to ``x``, so ``--verify`` passes bit-for-bit.

VERIFY THIS EXAMPLE with a real backend:
    vten build --kernel layout_passthrough --backend verilator
    vten run   --kernel layout_passthrough --test TestLayoutPassthrough \
               --backend verilator --verify
(cpu backend also works and needs no build.)
"""

import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor


class LayoutPassthroughKernel(Kernel):
    spec = "kernels/layout_passthrough/kernel_spec.yaml"

    # Declared shape is LOGICAL because layout_data_in() is defined below.
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

    # ── Layout hooks (symmetric, self-inverse) ──

    def layout_data_in(self, data: torch.Tensor) -> torch.Tensor:
        """Logical → physical: reverse element order along axis 0."""
        return torch.flip(data, [0])

    def unlayout_data_out(self, data: torch.Tensor) -> torch.Tensor:
        """Physical → logical: reverse again (flip is its own inverse)."""
        return torch.flip(data, [0])

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden in PHYSICAL space.

        forward() receives the layout-applied (physical) input, and the DUT is a
        byte-verbatim passthrough, so the physical golden is the identity of the
        physical input. vTen applies unlayout_data_out() to this golden before
        comparison, recovering the logical input.
        """
        data = inputs.get("data_in", self.data_in.data)
        return {"data_out": data.clone()}

    def run(self, ctx) -> None:
        # Combinational DUT (s_axis_tready = m_axis_tready): issue PUSH and PULL
        # CONCURRENTLY — sequencing pull after push (an issue-dep waits for the dep
        # to COMMIT while the slave BFM only asserts tready during a PULL) deadlocks
        # on a real simulator. The host-side layout_data_in / unlayout_data_out
        # hooks are unaffected. Previously masked by cpu-only testing.
        ctx.push_tensor(self.data_in)
        ctx.pull_tensor(self.data_out)
