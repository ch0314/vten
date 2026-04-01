from vten.cli.run import TestScenario


class TestVectorAluAdd(TestScenario):
    """Vector ALU: element-wise ADD (op_mode=0)."""

    kernel = "vector_alu"


class TestVectorAluSub(TestScenario):
    """Vector ALU: element-wise SUB (op_mode=1)."""

    kernel = "vector_alu"
    configs = [{"name": "sub", "op_mode": 1}]


class TestVectorAluMul(TestScenario):
    """Vector ALU: element-wise MUL (op_mode=2)."""

    kernel = "vector_alu"
    configs = [{"name": "mul", "op_mode": 2}]


class TestVectorAluProbe(TestScenario):
    """Vector ALU ADD with probe=True on PULL.

    golden_buf_id=0 = operand_a buffer.
    result = A + B != A (when B != 0), so BFM MUST report probe mismatches.
    This test verifies that mismatch detection works.
    Host-side verify still passes (correct golden provided).
    """

    kernel = "vector_alu"

    def run(self, ctx, cfg):
        import sys
        from pathlib import Path

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

        h_push_a = ctx.push_tensor(k.operand_a, dep=h_cfg)
        h_push_b = ctx.push_tensor(k.operand_b, dep=h_cfg)
        # probe=True: BFM compares result against buf 0 (operand_a)
        h_pull = ctx.pull_tensor(k.result, dep=h_cfg, probe=True)

        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_cfg)
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

        # Host-side verify uses correct golden (A+B)
        ctx.verify(h_pull, k.forward()["result"])
