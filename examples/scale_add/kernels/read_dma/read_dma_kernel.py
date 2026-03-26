"""ReadDMA — AXI4 Memory Read to AXI4-Stream Output.

Reads data from memory via AXI4 master, outputs as AXI4-Stream.
Identity DMA: forward() returns input unchanged.
"""

import torch

from vten.kernel.base import Kernel
from vten.kernel.register import register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class ReadDMAKernel(Kernel):
    spec = "kernels/read_dma/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="mem_port",
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

    def forward(self) -> torch.Tensor:
        """Golden reference: DMA is identity."""
        return self.data_in.data.clone()
