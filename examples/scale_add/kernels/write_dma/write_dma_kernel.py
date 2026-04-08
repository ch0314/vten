"""WriteDMA — AXI4-Stream Input to AXI4 Memory Write.

Accepts data from AXI4-Stream, writes to memory via AXI4 master.
Identity DMA: forward() returns input unchanged.
"""

import torch

from vten.kernel.base import Kernel
from vten.kernel.register import register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class WriteDMAKernel(Kernel):
    spec = "kernels/write_dma/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream",
        direction=Direction.HOST_TO_DEV,
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="mem_port",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        h_push = ctx.push_tensor(self.data_in)

        h_cfg = ctx.configure(self, dep=h_push)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_push)

        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)
        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden reference: DMA is identity."""
        data = inputs.get("data_in", self.data_in.data)
        return {"data_out": data.clone()}
