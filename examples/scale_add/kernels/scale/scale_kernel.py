import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction
from vten.kernel.register import register


class ScaleKernel(Kernel):
    """Multiply each signed int8 element by scale_factor with saturation."""

    spec = "kernels/scale/kernel_spec.yaml"

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

    def forward(self, scale_factor: int = 2) -> torch.Tensor:
        x = self.data_in.data.to(torch.int16) * scale_factor
        return x.clamp(-128, 127).to(torch.int8)
