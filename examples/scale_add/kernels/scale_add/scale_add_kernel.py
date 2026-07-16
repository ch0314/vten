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

    # Composite-level params used by the golden forward() below (and propagated
    # to each sub-kernel's control register: scale_factor → scale, offset_value
    # → offset). Any config may override these; when a config omits them these
    # defaults apply, so every sweep entry resolves self.scale_factor /
    # self.offset_value.
    default_params = {"N": 1024, "scale_factor": 1, "offset_value": 0}

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
        # Internal (dotted) probe support: if a TestScenario declared
        # probes=["scale.data_out"], seed the golden for that internal wire.
        # This is a no-op unless the probe was requested. It is required because
        # this composite defines a *custom* forward() (no auto-chained
        # _golden_pool), so the framework cannot auto-extract the internal-wire
        # golden — see vten/runtime/probe_manager.py::resolve_internal_probe_golden.
        # The scale→offset wire carries the scale stage output (saturating int8
        # multiply), i.e. the data BEFORE offset is applied.
        if "scale.data_out" in getattr(ctx, "_declarative_probes", []):
            scaled = (self.data_in.data.to(torch.int16) * self.scale_factor)
            scaled = scaled.clamp(-128, 127).to(torch.int8)
            ctx.set_internal_probe_golden("scale", "data_out", scaled)

        # Streaming DUTs: each sub-core only accepts input once STARTED and
        # while its output is being drained (input tready gated on state==S_RUN
        # and downstream ready). Configure + start FIRST, then push (input) and
        # pull (output) run CONCURRENTLY, then poll done — same ordering fix as
        # the scale/offset single kernels. (Pushing before start deadlocks on a
        # real simulator; previously masked by cpu-only testing.)
        # ctx.configure auto-writes runtime_params with register mappings
        h_cfg = ctx.configure(self)

        # Start both sub-kernels
        h_start_s = ctx.write_register(self.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_o = ctx.write_register(self.offset_ctrl, {"start": 1}, dep=h_cfg)

        h_push = ctx.push_tensor(self.data_in, dep=h_start_s)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_start_o)

        # Poll both done flags
        h_poll_s = ctx.poll_register(self.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(self.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)
