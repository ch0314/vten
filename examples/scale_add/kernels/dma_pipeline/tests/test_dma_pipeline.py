"""TestScenarios for dma_pipeline composite kernel.

Full DMA pipeline: ReadDMA → Scale → Offset → WriteDMA.
  ReadDMA reads from DDR via AXI4, outputs stream.
  Scale multiplies by factor, Offset adds value (both stream-to-stream).
  WriteDMA accepts stream, writes to DDR via AXI4.

Dependency model for AXI4 MM:
  - PUSH (mem_in): dispatched before start, completes when DUT reads
  - PULL (mem_out): dispatched before start, commit delayed until all done
  - Configure and start must NOT depend on PUSH completion (deadlock)

TestDmaPipeline: parametrized via configs for parameter sweep.
TestDmaPipelineStore: store_tensor() readback after pull.
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestDmaPipeline(TestScenario):
    """4-stage DMA pipeline with parameter sweep."""

    kernel = "dma_pipeline"

    configs = [
        {"name": "default"},                                         # N=1024, scale=2, off=1
        {"name": "identity", "scale_factor": 1, "offset_value": 0},  # DMA round-trip
        {"name": "small_n", "N": 32},                                # minimal DMA
        {"name": "overflow", "scale_factor": 10, "offset_value": 50},
    ]

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from dma_pipeline_kernel import DmaPipelineKernel

        N = cfg.get("N", 1024)
        scale_factor = cfg.get("scale_factor", 2)
        offset_value = cfg.get("offset_value", 1)

        k = ctx.instantiate(DmaPipelineKernel, N=N)
        k.generate_inputs(seed=42)

        total_beats = N // 32

        # 1. Load input tensor
        h_load = ctx.load_tensor(k.data_in)

        # 2. Push input to AXI4 BFM (ReadDMA will read from here)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)

        # 3. Pull output — dispatch early so AXI4 BFM has PULL entry
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)

        # 4. Configure ReadDMA and WriteDMA (auto_bind: addresses + length)
        h_cfg_rdma = ctx.configure(k, dep=h_load)

        # 5. Configure Scale sub-kernel
        h_sf = ctx.write_register(
            k.scale_ctrl, {"scale_factor": scale_factor}, dep=h_load,
        )
        h_len_s = ctx.write_register(
            k.scale_ctrl, {"length": total_beats}, dep=h_sf,
        )

        # 6. Configure Offset sub-kernel
        h_ov = ctx.write_register(
            k.offset_ctrl, {"offset_value": offset_value}, dep=h_load,
        )
        h_len_o = ctx.write_register(
            k.offset_ctrl, {"length": total_beats}, dep=h_ov,
        )

        # 7. Start all sub-kernels (after all config is done)
        all_cfg = [h_cfg_rdma, h_len_s, h_len_o]
        h_start_rdma = ctx.write_register(k.rdma_ctrl, {"start": 1}, dep=all_cfg)
        h_start_scale = ctx.write_register(k.scale_ctrl, {"start": 1}, dep=all_cfg)
        h_start_offset = ctx.write_register(k.offset_ctrl, {"start": 1}, dep=all_cfg)
        h_start_wdma = ctx.write_register(k.wdma_ctrl, {"start": 1}, dep=all_cfg)

        # 8. Poll all done flags
        h_poll_rdma = ctx.poll_register(k.rdma_ctrl, "done", dep=h_start_rdma)
        h_poll_scale = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_scale)
        h_poll_offset = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_offset)
        h_poll_wdma = ctx.poll_register(k.wdma_ctrl, "done", dep=h_start_wdma)

        # 9. Pull commits after all sub-kernels done
        h_pull.add_commit_dependency(h_poll_rdma)
        h_pull.add_commit_dependency(h_poll_scale)
        h_pull.add_commit_dependency(h_poll_offset)
        h_pull.add_commit_dependency(h_poll_wdma)

        # 10. Verify
        golden_off = offset_value if offset_value < 128 else offset_value - 256
        ctx.verify(h_pull, k.forward(
            scale_factor=scale_factor, offset_value=golden_off,
        ))


class TestDmaPipelineStore(TestScenario):
    """DmaPipeline with store_tensor() readback after pull."""

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

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)

        h_cfg_rdma = ctx.configure(k, dep=h_load)

        h_sf = ctx.write_register(k.scale_ctrl, {"scale_factor": 2}, dep=h_load)
        h_len_s = ctx.write_register(k.scale_ctrl, {"length": total_beats}, dep=h_sf)

        h_ov = ctx.write_register(k.offset_ctrl, {"offset_value": 1}, dep=h_load)
        h_len_o = ctx.write_register(k.offset_ctrl, {"length": total_beats}, dep=h_ov)

        all_cfg = [h_cfg_rdma, h_len_s, h_len_o]
        h_start_rdma = ctx.write_register(k.rdma_ctrl, {"start": 1}, dep=all_cfg)
        h_start_scale = ctx.write_register(k.scale_ctrl, {"start": 1}, dep=all_cfg)
        h_start_offset = ctx.write_register(k.offset_ctrl, {"start": 1}, dep=all_cfg)
        h_start_wdma = ctx.write_register(k.wdma_ctrl, {"start": 1}, dep=all_cfg)

        h_poll_rdma = ctx.poll_register(k.rdma_ctrl, "done", dep=h_start_rdma)
        h_poll_scale = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_scale)
        h_poll_offset = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_offset)
        h_poll_wdma = ctx.poll_register(k.wdma_ctrl, "done", dep=h_start_wdma)

        h_pull.add_commit_dependency(h_poll_rdma)
        h_pull.add_commit_dependency(h_poll_scale)
        h_pull.add_commit_dependency(h_poll_offset)
        h_pull.add_commit_dependency(h_poll_wdma)

        # Store output tensor back to host memory
        h_store = ctx.store_tensor(k.data_out, dep=h_pull)

        ctx.verify(h_pull, k.forward(scale_factor=2, offset_value=1))
