"""TestScenario for write_dma kernel.

WriteDMA accepts AXI4-Stream input, writes to memory via AXI4 master.
  1. Load input tensor to host buffer
  2. Push input (stream data into BFM for DUT to consume)
  3. Configure registers (auto_bind fills dst address + length)
  4. Start kernel — DUT accepts stream, writes to memory
  5. Poll for done
  6. Pull output tensor (read DUT writes from BFM memory)
  7. Store output
  8. Verify: output == input (identity DMA)
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestWriteDMA(TestScenario):
    kernel = "write_dma"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from write_dma_kernel import WriteDMAKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(WriteDMAKernel, N=N)
        k.generate_inputs(seed=42)

        # 1. Host → BFM
        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)

        # 2. Configure: auto_bind writes dst addr + length registers
        #    Depends on load (not push!) to avoid deadlock.
        h_cfg = ctx.configure(k, dep=h_load)

        # 3. Pull output — dispatch early so AXI4 BFM has the PULL entry
        #    before DUT starts writing. Commit delayed until after done.
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)

        # 4. Start kernel — after configure, push already dispatched
        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_cfg)

        # 5. Poll for completion
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

        # 6. Store output after pull commits
        h_store = ctx.store_tensor(k.data_out, dep=h_pull)

        # 6. Verify
        ctx.verify(h_store, k.forward())
