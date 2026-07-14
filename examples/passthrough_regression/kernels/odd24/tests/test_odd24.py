import sys
from pathlib import Path

from vten.cli.scenario import TestScenario


class TestOdd24(TestScenario):
    """24-bit bus width passthrough: 3 bytes per beat, non-power-of-2 alignment."""

    kernel = "odd24"


class TestOdd24Probe(TestScenario):
    """24-bit bus with probe: beat-by-beat golden comparison at 3-byte granularity."""

    kernel = "odd24"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from odd24_kernel import Odd24Kernel

        k = ctx.instantiate(Odd24Kernel, N=cfg.get("N", 768))
        k.generate_inputs(seed=42)

        h_push = ctx.push_tensor(k.data_in)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_push, probe=True)
