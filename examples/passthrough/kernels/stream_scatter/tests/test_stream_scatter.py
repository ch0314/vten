import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestStreamScatter(TestScenario):
    """Stream scatter: AXI4-Stream → scale×2 → dual HBM ports.

    Tests: mixed protocol, multi-port AXI4, BARRIER, READ_REG, poll_register.
    """

    kernel = "stream_scatter"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from stream_scatter_kernel import StreamScatterKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(StreamScatterKernel, N=N)
        k.generate_inputs(seed=42)

        # 1. Load input to SHM
        h_load = ctx.load_tensor(k.data_in)

        # 2. Configure (auto_bind: dst0/dst1 addr)
        h_cfg = ctx.configure(k, dep=h_load)

        # 3. Write length manually (total input beats = N / elements_per_beat)
        total_beats = N // 32
        h_len = ctx.write_register(k.ctrl, {"length": total_beats}, dep=h_cfg)

        # 4. BARRIER — ensure all register writes complete before data ops
        h_barrier = ctx.barrier()

        # 4. Activate BFMs before start
        h_push = ctx.push_tensor(k.data_in, dep=h_barrier)
        h_pull_0 = ctx.pull_tensor(k.result_0, dep=h_barrier)
        h_pull_1 = ctx.pull_tensor(k.result_1, dep=h_barrier)

        # 5. Start DUT
        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_barrier)

        # 6. READ_REG — read beat counter (for testing the feature)
        h_read = ctx.read_register(k.ctrl, "count", dep=h_start)

        # 7. Poll for completion — PULL commits only after done
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)
        h_pull_0.add_commit_dependency(h_poll)
        h_pull_1.add_commit_dependency(h_poll)

        # 8. Verify both output ports
        golden_0, golden_1 = k.forward()
        ctx.verify(h_pull_0, golden_0)
        ctx.verify(h_pull_1, golden_1)
