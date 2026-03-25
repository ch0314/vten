import torch
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor

class UnalignedKernel(Kernel):
    """N=100 on 256-bit bus: 100 bytes is not a multiple of 32 bytes/beat.
    Tests padding behavior at the last partial beat."""
    spec = "kernels/unaligned/kernel_spec.yaml"
    data_in = Tensor(shape=("${N}",), dtype=torch.int8, interface="input_stream")
    data_out = Tensor(shape=("${N}",), dtype=torch.int8, interface="output_stream")

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self):
        return self.data_in.data.clone()
