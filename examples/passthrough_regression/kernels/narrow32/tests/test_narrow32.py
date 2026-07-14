import sys
from pathlib import Path

from vten.cli.scenario import TestScenario


class TestNarrow32(TestScenario):
    """32-bit bus width passthrough: 4 bytes per beat."""

    kernel = "narrow32"


class TestNarrow32Probe(TestScenario):
    """32-bit bus with probe: beat-by-beat golden comparison."""

    kernel = "narrow32"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from narrow32_kernel import Narrow32Kernel

        k = ctx.instantiate(Narrow32Kernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_push = ctx.push_tensor(k.data_in)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_push, probe=True)
