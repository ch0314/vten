import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class MultiPortScatterKernel(Kernel):
    spec = "kernels/multi_port_scatter/kernel_spec.yaml"

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="din",
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="dout",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        h_send = ctx.send_tensor(self.data_in)
        h_recv = ctx.recv_tensor(self.data_out, dep=h_send)

        ctx.verify(h_recv, self.forward()["data_out"])

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden: passthrough (output == input)."""
        return {"data_out": self.data_in.data.clone()}
