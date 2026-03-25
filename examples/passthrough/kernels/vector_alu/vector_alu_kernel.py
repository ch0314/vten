import torch

from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class VectorAluKernel(Kernel):
    spec = "kernels/vector_alu/kernel_spec.yaml"

    ctrl = register("ctrl")

    operand_a = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="mem_port",
        direction=Direction.HOST_TO_DEV,
    )
    operand_b = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="mem_port",
        direction=Direction.HOST_TO_DEV,
    )
    result = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="mem_port",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.operand_a.fill_random(generator=rng)
        self.operand_b.fill_random(generator=rng)

    def forward(self, op_mode: int = 0) -> torch.Tensor:
        a = self.operand_a.data.to(torch.int16)
        b = self.operand_b.data.to(torch.int16)
        if op_mode == 0:
            raw = a + b
        elif op_mode == 1:
            raw = a - b
        elif op_mode == 2:
            raw = a * b
            return raw.to(torch.int8)  # MUL: low 8 bits (no saturation)
        else:
            raw = a + b
        # Saturating clamp for ADD/SUB
        return raw.clamp(-128, 127).to(torch.int8)
