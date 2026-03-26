"""TestScenario for dma_pipeline composite kernel.

Full DMA pipeline: ReadDMA → Scale → Offset → WriteDMA.
  ReadDMA reads from DDR via AXI4, outputs stream.
  Scale multiplies by factor, Offset adds value (both stream-to-stream).
  WriteDMA accepts stream, writes to DDR via AXI4.

Dependency model for AXI4 MM:
  - PUSH (mem_in): dispatched before start, completes when DUT reads
  - PULL (mem_out): dispatched before start, commit delayed until all done
  - Configure and start must NOT depend on PUSH completion (deadlock)
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestDmaPipeline(TestScenario):
    kernel = "dma_pipeline"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from dma_pipeline_kernel import DmaPipelineKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(DmaPipelineKernel, N=N)
        k.generate_inputs(seed=42)

        total_beats = N // 32

        # 1. Load input tensor
        h_load = ctx.load_tensor(k.data_in)

        # 2. Push input to AXI4 BFM (ReadDMA will read from here)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)

        # 3. Pull output — dispatch early so AXI4 BFM has PULL entry
        #    before WriteDMA starts writing
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)

        # 4. Configure ReadDMA (auto_bind: src_addr + length)
        h_cfg_rdma = ctx.configure(k, dep=h_load)

        # 5. Configure Scale sub-kernel
        h_sf = ctx.write_register(k.scale_ctrl, {"scale_factor": 2}, dep=h_load)
        h_len_s = ctx.write_register(k.scale_ctrl, {"length": total_beats}, dep=h_sf)

        # 6. Configure Offset sub-kernel
        h_ov = ctx.write_register(k.offset_ctrl, {"offset_value": 1}, dep=h_load)
        h_len_o = ctx.write_register(k.offset_ctrl, {"length": total_beats}, dep=h_ov)

        # 7. Configure WriteDMA (auto_bind: dst_addr + length)
        #    Note: configure() already generated writes for ALL auto_bind regs

        # 8. Start all sub-kernels (after all config is done)
        all_cfg = [h_cfg_rdma, h_len_s, h_len_o]
        h_start_rdma = ctx.write_register(k.rdma_ctrl, {"start": 1}, dep=all_cfg)
        h_start_scale = ctx.write_register(k.scale_ctrl, {"start": 1}, dep=all_cfg)
        h_start_offset = ctx.write_register(k.offset_ctrl, {"start": 1}, dep=all_cfg)
        h_start_wdma = ctx.write_register(k.wdma_ctrl, {"start": 1}, dep=all_cfg)

        # 9. Poll all done flags
        h_poll_rdma = ctx.poll_register(k.rdma_ctrl, "done", dep=h_start_rdma)
        h_poll_scale = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_scale)
        h_poll_offset = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_offset)
        h_poll_wdma = ctx.poll_register(k.wdma_ctrl, "done", dep=h_start_wdma)

        # 10. Pull commits after all sub-kernels done
        h_pull.add_commit_dependency(h_poll_rdma)
        h_pull.add_commit_dependency(h_poll_scale)
        h_pull.add_commit_dependency(h_poll_offset)
        h_pull.add_commit_dependency(h_poll_wdma)

        # 11. Verify
        ctx.verify(h_pull, k.forward(scale_factor=2, offset_value=1))
