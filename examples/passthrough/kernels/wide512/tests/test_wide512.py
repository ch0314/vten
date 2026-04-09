import sys
from pathlib import Path

from vten.cli.scenario import TestScenario


class TestWide512(TestScenario):
    """512-bit bus width passthrough: 64 bytes per beat, tests large bulk memcpy."""

    kernel = "wide512"


class TestWide512Probe(TestScenario):
    """512-bit bus with probe: beat-by-beat golden comparison at 64-byte granularity."""

    kernel = "wide512"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from wide512_kernel import Wide512Kernel

        k = ctx.instantiate(Wide512Kernel, N=cfg.get("N", 4096))
        k.generate_inputs(seed=42)

        h_push = ctx.push_tensor(k.data_in)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_push, probe=True)
