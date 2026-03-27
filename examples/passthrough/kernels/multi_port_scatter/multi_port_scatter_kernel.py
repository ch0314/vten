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

    def forward(self):
        """Golden: passthrough (output == input)."""
        return self.data_in.data.clone()
