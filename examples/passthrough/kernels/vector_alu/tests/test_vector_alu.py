import sys
from pathlib import Path

from vten.cli.run import TestScenario


def _run_vector_alu(ctx, cfg, op_mode: int):
    """Common flow for all vector ALU operations."""
    kernel_dir = str(Path(__file__).resolve().parent.parent)
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

    from vector_alu_kernel import VectorAluKernel

    N = cfg.get("N", 1024)
    k = ctx.instantiate(VectorAluKernel, N=N)
    k.generate_inputs(seed=42)

    # 1. Load operands to SHM
    h_load_a = ctx.load_tensor(k.operand_a)
    h_load_b = ctx.load_tensor(k.operand_b)

    # 2. Configure DUT registers (auto_bind: addr, length)
    h_cfg = ctx.configure(k, dep=[h_load_a, h_load_b])

    # 3. Set operation mode
    h_op = ctx.write_register(k.ctrl, {"op_mode": op_mode}, dep=h_cfg)

    # 4. Activate BFMs BEFORE start — they wait passively for DUT transactions
    h_push_a = ctx.push_tensor(k.operand_a, dep=h_op)
    h_push_b = ctx.push_tensor(k.operand_b, dep=h_op)
    h_pull = ctx.pull_tensor(k.result, dep=h_op)

    # 5. Start DUT — now BFMs are ready to serve/capture data
    h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_op)

    # 6. Poll for completion — PULL commits only after done
    h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)
    h_pull.add_commit_dependency(h_poll)

    # 7. Verify
    ctx.verify(h_pull, k.forward(op_mode=op_mode))


class TestVectorAluAdd(TestScenario):
    """Vector ALU: element-wise ADD (op_mode=0)."""

    kernel = "vector_alu"

    def run(self, ctx, cfg):
        _run_vector_alu(ctx, cfg, op_mode=0)


class TestVectorAluSub(TestScenario):
    """Vector ALU: element-wise SUB (op_mode=1)."""

    kernel = "vector_alu"

    def run(self, ctx, cfg):
        _run_vector_alu(ctx, cfg, op_mode=1)


class TestVectorAluMul(TestScenario):
    """Vector ALU: element-wise MUL (op_mode=2)."""

    kernel = "vector_alu"

    def run(self, ctx, cfg):
        _run_vector_alu(ctx, cfg, op_mode=2)


class TestVectorAluProbe(TestScenario):
    """Vector ALU ADD with probe=True on PULL.

    golden_buf_id=0 = operand_a buffer.
    result = A + B != A (when B != 0), so BFM MUST report probe mismatches.
    This test verifies that mismatch detection works.
    Host-side verify still passes (correct golden provided).
    """

    kernel = "vector_alu"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from vector_alu_kernel import VectorAluKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(VectorAluKernel, N=N)
        k.generate_inputs(seed=42)

        h_load_a = ctx.load_tensor(k.operand_a)
        h_load_b = ctx.load_tensor(k.operand_b)
        h_cfg = ctx.configure(k, dep=[h_load_a, h_load_b])
        h_op = ctx.write_register(k.ctrl, {"op_mode": 0}, dep=h_cfg)

        h_push_a = ctx.push_tensor(k.operand_a, dep=h_op)
        h_push_b = ctx.push_tensor(k.operand_b, dep=h_op)
        # probe=True: BFM compares result against buf 0 (operand_a)
        # Since result = A + B != A, this WILL produce mismatches in xsim log
        h_pull = ctx.pull_tensor(k.result, dep=h_op, probe=True)

        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_op)
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

        # Host-side verify uses correct golden (A+B)
        ctx.verify(h_pull, k.forward(op_mode=0))
