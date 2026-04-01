import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestBeat7(TestScenario):
    """56-bit bus: 7 bytes/beat, prime-ish element count."""

    kernel = "beat7"


class TestBeat7Probe(TestScenario):
    kernel = "beat7"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)
        from beat7_kernel import Beat7Kernel
        k = ctx.instantiate(Beat7Kernel, N=cfg.get("N", 700))
        k.generate_inputs(seed=42)
        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load, probe=True)
        ctx.verify(h_pull, k.forward()["data_out"])
