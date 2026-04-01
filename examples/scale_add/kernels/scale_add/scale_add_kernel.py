import sys
from pathlib import Path

import torch

from vten.kernel.composite import CompositeKernel
from vten.kernel.register import register

# Import sub-kernels
_kernel_base = str(Path(__file__).resolve().parent.parent)
if _kernel_base not in sys.path:
    sys.path.insert(0, _kernel_base)

from scale.scale_kernel import ScaleKernel
from offset.offset_kernel import OffsetKernel


class ScaleAddKernel(CompositeKernel):
    """Composite: scale(x factor) → offset(+ value) pipeline.

    v2 API: sub-kernels as instances, >> connections, auto-expose.
    """

    # Sub-kernels
    scale = ScaleKernel()
    offset = OffsetKernel()

    # Register proxies for each sub-kernel's ctrl
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")

    # Internal connection: scale output → offset input
    connections = [scale.data_out >> offset.data_in]

    # Auto-exposed tensors (not in connections):
    #   scale.data_in  → data_in
    #   offset.data_out → data_out
    #   scale.ctrl, offset.ctrl → ctrl registers

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden: scale then offset, both with signed int8 saturation."""
        data = inputs.get("data_in", self.data_in.data)
        x = data.to(torch.int16) * self.scale_factor
        x = x.clamp(-128, 127)
        ov = self.offset_value
        if ov >= 128:
            ov = ov - 256
        x = x + ov
        return {"data_out": x.clamp(-128, 127).to(torch.int8)}

    def run(self, ctx) -> None:
        h_load = ctx.load_tensor(self.data_in)

        # ctx.configure auto-writes runtime_params with register mappings
        h_cfg = ctx.configure(self, dep=h_load)

        h_push = ctx.push_tensor(self.data_in, dep=h_cfg)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_cfg)

        # Start both sub-kernels
        h_start_s = ctx.write_register(self.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_o = ctx.write_register(self.offset_ctrl, {"start": 1}, dep=h_cfg)

        # Poll both done flags
        h_poll_s = ctx.poll_register(self.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(self.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)

        ctx.verify(h_pull, self.forward()["data_out"])
