import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestNarrow8(TestScenario):
    """8-bit bus width passthrough: 1 byte per beat, tests minimum bulk memcpy size."""

    kernel = "narrow8"


class TestNarrow8Probe(TestScenario):
    """8-bit bus with probe: beat-by-beat golden comparison at 1 byte granularity."""

    kernel = "narrow8"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from narrow8_kernel import Narrow8Kernel

        k = ctx.instantiate(Narrow8Kernel, N=cfg.get("N", 256))
        k.generate_inputs(seed=42)

        h_push = ctx.push_tensor(k.data_in)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_push, probe=True)
