import torch

from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor


class StreamDmaV2Kernel(Kernel):
    spec = "kernels/stream_dma_v2/kernel_spec.yaml"

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
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self):
        return self.data_in.data.clone()
