import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor


class Narrow8Kernel(Kernel):
    spec = "kernels/narrow8/kernel_spec.yaml"

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
        h_load = ctx.load_tensor(self.data_in)
        h_push = ctx.push_tensor(self.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_load)
        ctx.verify(h_pull, self.forward()["data_out"])
