"""DMA Pipeline — ReadDMA → Scale → Offset → WriteDMA.

Full memory-mapped pipeline: reads from DDR, applies scale + offset,
writes back to DDR. Like a Vitis multi-IP dataflow.
"""

import sys
from pathlib import Path

import torch

from vten.kernel.composite import CompositeKernel
from vten.kernel.register import register

# Import sub-kernels
_kernel_base = str(Path(__file__).resolve().parent.parent)
if _kernel_base not in sys.path:
    sys.path.insert(0, _kernel_base)

from read_dma.read_dma_kernel import ReadDMAKernel
from scale.scale_kernel import ScaleKernel
from offset.offset_kernel import OffsetKernel
from write_dma.write_dma_kernel import WriteDMAKernel


class DmaPipelineKernel(CompositeKernel):
    """Composite: ReadDMA → Scale → Offset → WriteDMA.

    v2 API: sub-kernels as instances, >> connections, auto-expose.
    """

    # Sub-kernels
    read_dma = ReadDMAKernel()
    scale = ScaleKernel()
    offset = OffsetKernel()
    write_dma = WriteDMAKernel()

    # Register proxies
    rdma_ctrl = register("rdma_ctrl")
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")
    wdma_ctrl = register("wdma_ctrl")

    # Internal connections: ReadDMA → Scale → Offset → WriteDMA
    connections = [
        read_dma.data_out >> scale.data_in,
        scale.data_out >> offset.data_in,
        offset.data_out >> write_dma.data_in,
    ]

    # Auto-exposed tensors (not in connections):
    #   read_dma.data_in  → data_in
    #   write_dma.data_out → data_out
    #   all ctrl registers

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden: read → scale → offset → write (identity DMA on both ends)."""
        data = inputs.get("data_in", self.data_in.data)
        x = data.to(torch.int16) * self.scale_factor
        x = x.clamp(-128, 127)
        ov = self.offset_value
        if ov >= 128:
            ov = ov - 256
        x = x + ov
        return {"data_out": x.clamp(-128, 127).to(torch.int8)}

    def run(self, ctx) -> None:
        # Push input to AXI4 BFM (ReadDMA reads from here)
        h_push = ctx.push_tensor(self.data_in)

        # Pull output — dispatch early so AXI4 BFM has PULL entry
        h_pull = ctx.pull_tensor(self.data_out, dep=h_push)

        # Configure all sub-kernels (auto_bind + runtime_params registers)
        h_cfg = ctx.configure(self, dep=h_push)

        # Start all sub-kernels
        h_start_rdma = ctx.write_register(self.rdma_ctrl, {"start": 1}, dep=h_cfg)
        h_start_scale = ctx.write_register(self.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_offset = ctx.write_register(self.offset_ctrl, {"start": 1}, dep=h_cfg)
        h_start_wdma = ctx.write_register(self.wdma_ctrl, {"start": 1}, dep=h_cfg)

        # Poll all done flags
        h_poll_rdma = ctx.poll_register(self.rdma_ctrl, "done", dep=h_start_rdma)
        h_poll_scale = ctx.poll_register(self.scale_ctrl, "done", dep=h_start_scale)
        h_poll_offset = ctx.poll_register(self.offset_ctrl, "done", dep=h_start_offset)
        h_poll_wdma = ctx.poll_register(self.wdma_ctrl, "done", dep=h_start_wdma)

        # Pull commits after all sub-kernels done
        h_pull.add_commit_dependency(h_poll_rdma)
        h_pull.add_commit_dependency(h_poll_scale)
        h_pull.add_commit_dependency(h_poll_offset)
        h_pull.add_commit_dependency(h_poll_wdma)
