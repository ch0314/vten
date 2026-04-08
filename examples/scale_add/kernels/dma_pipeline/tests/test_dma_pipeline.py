"""TestScenarios for dma_pipeline composite kernel.

Full DMA pipeline: ReadDMA → Scale → Offset → WriteDMA.
  ReadDMA reads from DDR via AXI4, outputs stream.
  Scale multiplies by factor, Offset adds value (both stream-to-stream).
  WriteDMA accepts stream, writes to DDR via AXI4.

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

        h_push = ctx.push_tensor(k.data_in)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_push)

        h_cfg = ctx.configure(k, dep=h_push)

        h_start_rdma = ctx.write_register(k.rdma_ctrl, {"start": 1}, dep=h_cfg)
        h_start_scale = ctx.write_register(k.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_offset = ctx.write_register(k.offset_ctrl, {"start": 1}, dep=h_cfg)
        h_start_wdma = ctx.write_register(k.wdma_ctrl, {"start": 1}, dep=h_cfg)

        h_poll_rdma = ctx.poll_register(k.rdma_ctrl, "done", dep=h_start_rdma)
        h_poll_scale = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_scale)
        h_poll_offset = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_offset)
        h_poll_wdma = ctx.poll_register(k.wdma_ctrl, "done", dep=h_start_wdma)

        h_pull.add_commit_dependency(h_poll_rdma)
        h_pull.add_commit_dependency(h_poll_scale)
        h_pull.add_commit_dependency(h_poll_offset)
        h_pull.add_commit_dependency(h_poll_wdma)
