import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction
from vten.kernel.register import register


class OffsetKernel(Kernel):
    """Add signed offset_value to each int8 element with saturation."""

    spec = "kernels/offset/kernel_spec.yaml"

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

    default_params = {"N": 1024, "offset_value": 0}

    def compute_derived_params(self) -> dict:
        N = getattr(self, "N", 1024)
        return {"length": N // 32}

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        data = inputs.get("data_in", self.data_in.data)
        # offset_value is register-width (8-bit): handle uint8→signed wrap
        ov = self.offset_value
        if ov >= 128:
            ov = ov - 256
        x = data.to(torch.int16) + ov
        return {"data_out": x.clamp(-128, 127).to(torch.int8)}

    def run(self, ctx) -> None:
        h_load = ctx.load_tensor(self.data_in)
        h_cfg = ctx.configure(self, dep=h_load)

        h_push = ctx.push_tensor(self.data_in, dep=h_cfg)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_cfg)

        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)
        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

        ctx.verify(h_pull, self.forward())
