import sys
from pathlib import Path
from vten.cli.run import TestScenario

class TestInt16X7(TestScenario):
    """112-bit bus: 7 int16 per beat (14 bytes, odd element count)."""
    kernel = "int16_x7"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)
        from int16_x7_kernel import Int16X7Kernel
        k = ctx.instantiate(Int16X7Kernel, N=cfg.get("N", 700))
        k.generate_inputs(seed=42)
        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)
        ctx.verify(h_pull, k.forward())
