import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction
from vten.kernel.register import register


class OffsetKernel(Kernel):
    """Add signed offset_value to each int8 element with saturation."""

    spec = "kernels/offset/kernel_spec.yaml"

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
        interface="output_stream",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, offset_value: int = 1) -> torch.Tensor:
        x = self.data_in.data.to(torch.int16) + offset_value
        return x.clamp(-128, 127).to(torch.int8)
