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

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        a = inputs.get("operand_a", self.operand_a.data).to(torch.int16)
        b = inputs.get("operand_b", self.operand_b.data).to(torch.int16)
        op_mode = self.op_mode
        if op_mode == 0:
            raw = a + b
        elif op_mode == 1:
            raw = a - b
        elif op_mode == 2:
            raw = a * b
            return {"result": raw.to(torch.int8)}  # MUL: low 8 bits (no saturation)
        else:
            raw = a + b
        # Saturating clamp for ADD/SUB
        return {"result": raw.clamp(-128, 127).to(torch.int8)}

    def run(self, ctx) -> None:
        h_load_a = ctx.load_tensor(self.operand_a)
        h_load_b = ctx.load_tensor(self.operand_b)

        h_cfg = ctx.configure(self, dep=[h_load_a, h_load_b])

        h_push_a = ctx.push_tensor(self.operand_a, dep=h_cfg)
        h_push_b = ctx.push_tensor(self.operand_b, dep=h_cfg)
        h_pull = ctx.pull_tensor(self.result, dep=h_cfg)

        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)

        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

        ctx.verify(h_pull, self.forward()["result"])
