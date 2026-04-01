import torch

from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class StreamDmaKernel(Kernel):
    spec = "kernels/stream_dma/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream",
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="dma_port",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        h_load = ctx.load_tensor(self.data_in)
        h_cfg = ctx.configure(self, dep=h_load)
        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)

        h_push = ctx.push_tensor(self.data_in, dep=h_start)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_start)
        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)

        ctx.verify(h_pull, self.forward()["data_out"])

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        return {"data_out": self.data_in.data.clone()}
