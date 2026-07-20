import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.runtime.quant import qmul
from vten.spec.models import Direction, QuantSpec
from vten.kernel.register import register

# Mirrors the quant: blocks in kernel_spec.yaml — int8 integer codes
# (Q-format, frac_bits=0), saturating output.
_DATA_QS = QuantSpec(bits=8, signed=True, frac_bits=0, overflow="saturate")
# scale_factor register: 8-bit UNSIGNED code (RTL: $signed({1'b0, reg})).
_SCALE_QS = QuantSpec(bits=8, signed=False, frac_bits=0)


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
        # Exact-width multiply + saturation driven by the declared QuantSpecs
        # (widen, multiply, clamp to [-128, 127]) — no hand-written clamps.
        out = qmul(data, _DATA_QS, self.scale_factor, _SCALE_QS, _DATA_QS)
        return {"data_out": out.to(torch.int8)}

    def run(self, ctx) -> None:
        # Streaming DUT: input_stream.tready = (state==S_RUN) & output_stream.tready,
        # so the core only accepts input once STARTED and while its output is being
        # drained. Configure + start FIRST, then push (input) and pull (output) run
        # CONCURRENTLY, then poll done. (Ordering push before start deadlocks on a
        # real simulator — it was previously masked by cpu-only testing.)
        h_cfg = ctx.configure(self)
        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)

        h_push = ctx.push_tensor(self.data_in, dep=h_start)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_start)

        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)
