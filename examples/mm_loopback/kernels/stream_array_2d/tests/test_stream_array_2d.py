"""TestScenario for stream_array_2d kernel.

2x2 AXI4-Stream passthrough with 2D array interface.
Tensor data is block-split across 4 streams (dimensions [2,2]).
Validates multi-dimensional ArraySpec flat_names and codegen.
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestStreamArray2d(TestScenario):
    kernel = "stream_array_2d"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from stream_array_2d_kernel import StreamArray2dKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(StreamArray2dKernel, N=N)
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)
        h_store = ctx.store_tensor(k.data_out, dep=h_pull)
        ctx.verify(h_store, k.forward())
