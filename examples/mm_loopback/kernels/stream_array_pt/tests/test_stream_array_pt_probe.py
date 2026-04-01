"""TestScenario for stream_array_pt with probe=True.

Same DUT and kernel as test_stream_array_pt, but with probe=True on PULL.
Each of the 4 array element BFMs independently compares received output
beats against golden data at the BFM level (not just Python-side verify).
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestStreamArrayPtProbe(TestScenario):
    kernel = "stream_array_pt"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from stream_array_pt_kernel import StreamArrayPtKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(StreamArrayPtKernel, N=N)
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)

        # probe=True: each array element BFM compares output against golden
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load, probe=True)

        h_store = ctx.store_tensor(k.data_out, dep=h_pull)
        ctx.verify(h_store, k.forward()["data_out"])
