import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestBrokenPassthrough(TestScenario):
    kernel = "broken_passthrough"


class TestBrokenPassthroughProbe(TestScenario):
    """Test broken passthrough WITH probe mode.

    The BFM compares each output beat against the golden buffer in real-time.
    Since the RTL corrupts every byte, the BFM should report beat-by-beat
    mismatches in the xsim simulation log.

    Host-side verify will also FAIL for the same reason.
    """

    kernel = "broken_passthrough"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from broken_passthrough_kernel import BrokenPassthroughKernel

        k = ctx.instantiate(BrokenPassthroughKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        # probe=True: BFM compares each output beat against golden buffer
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load, probe=True)
        # Host-side verify also confirms mismatch
        ctx.verify(h_pull, k.forward()["data_out"])
