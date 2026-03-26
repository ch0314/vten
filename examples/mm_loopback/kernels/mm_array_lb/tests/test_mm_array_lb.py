"""TestScenario for mm_array_lb kernel.

4-channel AXI4 MM loopback with array interface.
Tensor data is block-split across 4 AXI4 master ports by the runtime.

AXI4 MM dependency model (DUT is master, BFM is slave):
  - PUSH: registers data in BFM memory; completes when DUT reads it.
  - PULL: registers capture region in BFM; completes when DUT writes it.
  - Both PUSH and PULL must be dispatched BEFORE the core starts,
    so the BFM slave has entries in its active_table to match
    DUT master AR/AW requests. If missing → DECERR.
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestMmArrayLb(TestScenario):
    kernel = "mm_array_lb"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from mm_array_lb_kernel import MmArrayLbKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(MmArrayLbKernel, N=N)
        k.generate_inputs(seed=42)

        # 1. Host → SHM buffer
        h_load = ctx.load_tensor(k.data_in)

        # 2. Register data regions in BFMs — must be active before core starts
        #    PUSH splits across 4 mem_in ports, PULL splits across 4 mem_out ports
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)

        # 3. Configure: auto_bind writes src/dst addr + length registers
        h_cfg = ctx.configure(k, dep=h_load)

        # 4. Start kernel — after configure; push/pull already dispatched
        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_cfg)

        # 5. Poll for completion
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)

        # 6. Store output — wait for both PULL (data captured) and POLL (core done)
        h_store = ctx.store_tensor(k.data_out, dep=[h_pull, h_poll])

        # 7. Verify
        ctx.verify(h_store, k.forward())
