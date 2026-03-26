"""TestScenario for stream_array_pt kernel.

4-channel AXI4-Stream passthrough with array interface.
Tensor data is block-split across 4 streams by the runtime.

AXI4-Stream dependency model:
  - PUSH: BFM drives tdata/tvalid to DUT slave port
  - PULL: BFM captures tdata from DUT master port
  - No address registers needed — pure streaming
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestStreamArrayPt(TestScenario):
    kernel = "stream_array_pt"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from stream_array_pt_kernel import StreamArrayPtKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(StreamArrayPtKernel, N=N)
        k.generate_inputs(seed=42)

        # 1. Host → SHM buffer
        h_load = ctx.load_tensor(k.data_in)

        # 2. PUSH input (split across 4 streams) + PULL output (4 streams)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)

        # 3. Store output — wait for PULL completion
        h_store = ctx.store_tensor(k.data_out, dep=h_pull)

        # 4. Verify
        ctx.verify(h_store, k.forward())
