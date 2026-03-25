import torch

from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class StreamScatterKernel(Kernel):
    spec = "kernels/stream_scatter/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream",
    )
    result_0 = Tensor(
        shape=("${N}//2",),
        dtype=torch.int8,
        interface="hbm_0",
        direction=Direction.DEV_TO_HOST,
    )
    result_1 = Tensor(
        shape=("${N}//2",),
        dtype=torch.int8,
        interface="hbm_1",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Golden reference: scale by 2 (saturating), scatter to 2 ports."""
        data = self.data_in.data.to(torch.int16) * 2
        data = data.clamp(-128, 127).to(torch.int8)

        # Reshape into beats (32 elements per beat), then scatter
        elems_per_beat = 32
        beats = data.reshape(-1, elems_per_beat)
        even_beats = beats[0::2].reshape(-1)  # port 0
        odd_beats = beats[1::2].reshape(-1)   # port 1
        return even_beats, odd_beats
