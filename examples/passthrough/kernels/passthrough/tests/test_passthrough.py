import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestPassthrough(TestScenario):
    kernel = "passthrough"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from passthrough_kernel import PassthroughKernel

        k = ctx.instantiate(PassthroughKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)
        ctx.verify(h_pull, k.forward())


class TestPassthroughProbe(TestScenario):
    """Passthrough with probe mode: BFM-level beat-by-beat golden comparison.

    For passthrough, output == input, so golden_buf=0 (data_in buffer)
    is the correct reference for BFM probe comparison.
    """

    kernel = "passthrough"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from passthrough_kernel import PassthroughKernel

        k = ctx.instantiate(PassthroughKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        # probe=True: BFM compares each output beat against golden buffer (buf 0)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load, probe=True)
        # Host-side verify confirms data integrity
        ctx.verify(h_pull, k.forward())


