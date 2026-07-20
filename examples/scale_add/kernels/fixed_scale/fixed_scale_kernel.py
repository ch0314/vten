import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.runtime.quant import qmul
from vten.spec.models import Direction, QuantSpec
from vten.kernel.register import register

# Mirrors the quant: blocks in kernel_spec.yaml — Q1.7 signed codes
# (real = code / 128), round-half-up, saturating.
_DATA_QS = QuantSpec(
    bits=8, signed=True, frac_bits=7, rounding="half_up", overflow="saturate"
)
# coeff register: SIGNED 16-bit Q8.8 fixed-point (real = code / 256).
_COEFF_QS = QuantSpec(bits=16, signed=True, frac_bits=8)


class FixedScaleKernel(Kernel):
    """Q8.8 coefficient multiply on a Q1.7 stream: round-half-up + saturate.

    The reference showcase for QuantSpec-driven golden models: the entire
    fixed-point datapath — widen, multiply, round-half-up by 8 fractional
    bits, saturate — is ONE qmul call parameterized by the declared formats.
    No hand-written shifts, rounding constants, or clamps.

    RTL per lane: prod24 = x8 * coeff16; (prod24 + 128) >>> 8; clamp.
    """

    spec = "kernels/fixed_scale/kernel_spec.yaml"

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

    default_params = {"N": 1024, "coeff": 256}  # 256 = 1.0 in Q8.8

    def compute_derived_params(self) -> dict:
        N = getattr(self, "N", 1024)
        return {"length": N // 32}

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)
        # Pin the numerically interesting codes: qmin/qmax (saturation) and
        # +-1 / 0 (exact half-LSB ties at odd codes when coeff = 0.5).
        pinned = torch.tensor([-128, 127, -1, 1, 0], dtype=torch.int8)
        self.data_in.data[: pinned.numel()] = pinned

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        data = inputs.get("data_in", self.data_in.data)
        # Q1.7 x Q8.8 → Q1.7: qmul aligns the fractional bits (shift by
        # 7 + 8 - 7 = 8), rounds half-up, and saturates — exactly the RTL's
        # (prod + 128) >>> 8 then clamp, derived from the specs alone.
        out = qmul(data, _DATA_QS, self.coeff, _COEFF_QS, _DATA_QS)
        return {"data_out": out.to(torch.int8)}

    def run(self, ctx) -> None:
        # Streaming DUT (same protocol as scale): input_stream.tready is
        # gated on (state==S_RUN) & output_stream.tready, so configure +
        # start FIRST, then push (input) and pull (output) run CONCURRENTLY,
        # then poll done.
        h_cfg = ctx.configure(self)
        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)

        h_push = ctx.push_tensor(self.data_in, dep=h_start)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_start)

        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)
