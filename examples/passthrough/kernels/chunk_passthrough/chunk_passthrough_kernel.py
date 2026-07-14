"""chunk_passthrough — demonstrates chunks= on pull_tensor (host-side splitting).

This kernel REUSES the already-passing physical DUT ``rtl/passthrough.sv`` (a
byte-verbatim AXI4-Stream copy). Nothing about the RTL changes — the point is to
show the *host-side* chunked-read feature on a DUT we already trust.

Why chunks= needs NO RTL support
--------------------------------
``chunks=`` is a pure host-side read-splitting feature (confirmed by reading
``vten/runtime/context.py::pull_tensor`` and
``vten/runtime/ir.py::_lower_pull_chunk``): a single PULL is lowered into N
command groups, each draining a fraction (``serialized_size // N``) of the SAME
output stream into ``data_out:chunk_0..N-1`` buffers. On readback,
``vten/runtime/output_reader.py::read_all_chunk_bytes`` concatenates the chunk
buffers back in order. The DUT just emits its normal output stream; the host
decides how to drain it. So verification against ``forward()`` still holds.

Note: the split must land on whole beats. With N=1024 int8 packed 32 elements /
beat = 32 beats, ``chunks=4`` → 8 beats (256 bytes) per chunk — a clean split.
The default N=1024 is chosen so 4 divides the beat count evenly.

The kernel-level ``run()`` (not a TestScenario custom run) uses ``chunks=`` so it
is actually exercised by ``vten run`` — the CLI executes the kernel's ``run()``.

VERIFY THIS EXAMPLE with a real backend:
    vten build --kernel chunk_passthrough --backend verilator
    vten run   --kernel chunk_passthrough --test TestChunkPassthrough \
               --backend verilator --verify
"""

import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor


class ChunkPassthroughKernel(Kernel):
    spec = "kernels/chunk_passthrough/kernel_spec.yaml"

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
        # chunks=4: drain data_out in 4 host-side chunk groups. Returns a list of
        # OperationHandles (one per chunk); reassembled on host. No RTL change.
        ctx.pull_tensor(self.data_out, dep=h_push, chunks=4)
