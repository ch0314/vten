import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.runtime.quant import apply_overflow, qadd
from vten.spec.models import Direction, QuantSpec
from vten.kernel.register import register

# Mirrors the quant: blocks in kernel_spec.yaml — int8 integer codes
# (Q-format, frac_bits=0), saturating output.
_DATA_QS = QuantSpec(bits=8, signed=True, frac_bits=0, overflow="saturate")
# offset_value register: arrives as a raw UNSIGNED 8-bit code that the RTL
# reinterprets two's-complement ($signed(reg)) — a wrap fold, not a clamp.
_OFFSET_REG_QS = QuantSpec(bits=8, signed=True, frac_bits=0, overflow="wrap")


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
        # uint8 register code → signed value via two's-complement wrap
        # (e.g. 251 → -5), then saturating add — both driven by QuantSpecs.
        ov = apply_overflow(torch.tensor(self.offset_value), _OFFSET_REG_QS)
        out = qadd(data, _DATA_QS, ov, _DATA_QS, _DATA_QS)
        return {"data_out": out.to(torch.int8)}

    def run(self, ctx) -> None:
        # Streaming DUT: configure + start FIRST, then push (input) and pull
        # (output) run CONCURRENTLY, then poll done. (Ordering push before start
        # deadlocks on a real simulator; masked previously by cpu-only testing.)
        h_cfg = ctx.configure(self)
        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)

        h_push = ctx.push_tensor(self.data_in, dep=h_start)
        h_pull = ctx.pull_tensor(self.data_out, dep=h_start)

        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)
