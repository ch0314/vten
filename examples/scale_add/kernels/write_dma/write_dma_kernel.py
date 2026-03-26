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

    def forward(self) -> torch.Tensor:
        """Golden reference: DMA is identity."""
        return self.data_in.data.clone()
