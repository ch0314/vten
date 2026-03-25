import sys
from pathlib import Path
from vten.cli.run import TestScenario

class TestFloat32X4(TestScenario):
    """128-bit bus: 4 float32 per beat."""
    kernel = "float32_x4"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)
        from float32_x4_kernel import Float32X4Kernel
        k = ctx.instantiate(Float32X4Kernel, N=cfg.get("N", 512))
        k.generate_inputs(seed=42)
        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)
        ctx.verify(h_pull, k.forward())
