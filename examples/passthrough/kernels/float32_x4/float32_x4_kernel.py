import torch
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor

class Float32X4Kernel(Kernel):
    spec = "kernels/float32_x4/kernel_spec.yaml"
    data_in = Tensor(shape=("${N}",), dtype=torch.float32, interface="input_stream")
    data_out = Tensor(shape=("${N}",), dtype=torch.float32, interface="output_stream")

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self):
        return self.data_in.data.clone()
