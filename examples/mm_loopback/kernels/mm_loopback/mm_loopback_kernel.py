"""mm_loopback — AXI4 Memory-Mapped Loopback Kernel.

Reads data from DDR via AXI4 master, writes back to DDR.
Controlled via AXI4-Lite registers (address, length, start/done).
"""

import torch

from vten.kernel.base import Kernel
from vten.kernel.register import register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class MmLoopbackKernel(Kernel):
    spec = "kernels/mm_loopback/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.uint8,
        interface="mem_in",
        direction=Direction.HOST_TO_DEV,
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.uint8,
        interface="mem_out",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        h_load = ctx.load_tensor(self.data_in)
        h_push = ctx.push_tensor(self.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_load)

        h_cfg = ctx.configure(self, dep=h_load)
        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)
        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)

        h_store = ctx.store_tensor(self.data_out, dep=[h_pull, h_poll])
        ctx.verify(h_store, self.forward()["data_out"])

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden reference: loopback is identity."""
        return {"data_out": self.data_in.data.clone()}
