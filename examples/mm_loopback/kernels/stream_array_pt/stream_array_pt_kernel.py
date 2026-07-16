"""stream_array_pt — 4-channel AXI4-Stream Passthrough with Array Interface.

Tests tensor block-split distribution across 4 AXI4-Stream channels.
Each channel independently passes through its portion of the tensor.
"""

import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class StreamArrayPtKernel(Kernel):
    spec = "kernels/stream_array_pt/kernel_spec.yaml"

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.uint8,
        interface="din",
        direction=Direction.HOST_TO_DEV,
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.uint8,
        interface="dout",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        # Combinational DUT (s_axis_tready = m_axis_tready): PUSH and PULL must be
        # issued CONCURRENTLY — sequencing pull after push (an issue-dep waits for
        # the dep to COMMIT, and the slave BFM only asserts tready while a PULL is
        # active) deadlocks on a real simulator. Previously masked by cpu-only testing.
        ctx.push_tensor(self.data_in)
        ctx.pull_tensor(self.data_out)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden reference: passthrough is identity."""
        return {"data_out": self.data_in.data.clone()}
