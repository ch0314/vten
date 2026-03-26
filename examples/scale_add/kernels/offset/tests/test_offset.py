import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestOffset(TestScenario):
    """Offset kernel standalone: add offset_value=1 to each byte."""

    kernel = "offset"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from offset_kernel import OffsetKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(OffsetKernel, N=N)
        k.generate_inputs(seed=42)

        total_beats = N // 32

        h_load = ctx.load_tensor(k.data_in)
        h_ov = ctx.write_register(k.ctrl, {"offset_value": 1}, dep=h_load)
        h_len = ctx.write_register(k.ctrl, {"length": total_beats}, dep=h_ov)

        # Activate BFMs before start
        h_push = ctx.push_tensor(k.data_in, dep=h_len)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_len)

        # Start DUT
        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_len)

        # Poll for completion
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

        ctx.verify(h_pull, k.forward(offset_value=1))
