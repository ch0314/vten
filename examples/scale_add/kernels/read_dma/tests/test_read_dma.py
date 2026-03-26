"""TestScenario for read_dma kernel.

ReadDMA reads from memory via AXI4 master, outputs as AXI4-Stream.
  1. Load input tensor to host buffer
  2. Push (register data in BFM memory for DUT reads)
  3. Configure registers (auto_bind fills src address + length)
  4. Start kernel — DUT reads memory via AXI4, outputs to stream
  5. Poll for done
  6. Pull output tensor (capture stream output from BFM)
  7. Verify: output == input (identity DMA)

Note on AXI4 MM dependency model:
  - PUSH registers data in BFM but completes only when DUT reads it.
  - Configure and start must NOT depend on PUSH completion (deadlock).
  - PUSH must be dispatched before start so BFM has data ready.
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestReadDMA(TestScenario):
    kernel = "read_dma"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from read_dma_kernel import ReadDMAKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(ReadDMAKernel, N=N)
        k.generate_inputs(seed=42)

        # 1. Host → BFM memory
        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)

        # 2. Configure: auto_bind writes src addr + length registers
        #    Depends on load (not push!) to avoid deadlock.
        h_cfg = ctx.configure(k, dep=h_load)

        # 3. Pull output stream — dispatch before start so BFM is ready
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)

        # 4. Start kernel
        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_cfg)

        # 5. Poll for completion
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

        # 6. Verify
        ctx.verify(h_pull, k.forward())
