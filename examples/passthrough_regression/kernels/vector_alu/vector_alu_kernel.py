import torch

from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.runtime.quant import apply_overflow, qadd, qmul
from vten.spec.models import Direction, QuantSpec

# Operand format mirrors the mem_port quant: block — int8 integer codes
# (Q-format, frac_bits=0). Overflow is PER-OP in this ALU, which one
# interface-level quant block cannot express, so the op-specific choice is
# made here with local QuantSpecs:
#   ADD/SUB → widened result CLAMPED to [-128, 127]  (saturate)
#   MUL     → raw low 8 bits of the product          (wrap)
_DATA_QS = QuantSpec(bits=8, signed=True, frac_bits=0)
_SAT_QS = QuantSpec(bits=8, signed=True, frac_bits=0, overflow="saturate")
_WRAP_QS = QuantSpec(bits=8, signed=True, frac_bits=0, overflow="wrap")


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
        a = inputs.get("operand_a", self.operand_a.data)
        b = inputs.get("operand_b", self.operand_b.data)
        # Fall back to the runtime_params default (0 = ADD) so the golden is
        # computable when a scenario config doesn't set op_mode explicitly.
        op_mode = getattr(self, "op_mode", 0)
        if op_mode == 1:
            # SUB: no qsub helper — the widened int64 difference is exact;
            # the saturation semantics come from the declared spec.
            res = apply_overflow(a.to(torch.int64) - b.to(torch.int64), _SAT_QS)
        elif op_mode == 2:
            # MUL: RTL keeps the raw low 8 bits of the product (wrap).
            res = qmul(a, _DATA_QS, b, _DATA_QS, _WRAP_QS)
        else:
            # ADD (op_mode 0 and the RTL's default case): saturating.
            res = qadd(a, _DATA_QS, b, _DATA_QS, _SAT_QS)
        return {"result": res.to(torch.int8)}

    def run(self, ctx) -> None:
        # Memory-mapped DUT: PUSH/PULL only register the DDR buffers with the
        # passive AXI4 slave BFM — they complete when the DUT (the AXI master) has
        # read/written every byte, which happens only after start. Register both
        # operands and the result buffer FIRST (no deps), then configure + start.
        # (Gating configure/pull on PUSH completion deadlocks on a real simulator;
        # masked previously by cpu-only testing.)
        h_push_a = ctx.push_tensor(self.operand_a)
        h_push_b = ctx.push_tensor(self.operand_b)
        h_pull = ctx.pull_tensor(self.result)

        h_cfg = ctx.configure(self)

        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)

        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)
