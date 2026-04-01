import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction
from vten.kernel.register import register


class ScaleKernel(Kernel):
    """Multiply each signed int8 element by scale_factor with saturation."""

    spec = "kernels/scale/kernel_spec.yaml"

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

    default_params = {"N": 1024, "scale_factor": 1}

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
        x = data.to(torch.int16) * self.scale_factor
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
