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

    # Register proxies — names must match the auto-inferred {ref_name}_{iface}
    # convention (read_dma/write_dma, not rdma/wdma), else the interface has no
    # register map and write_register/poll_register fail to resolve fields.
    read_dma_ctrl = register("read_dma_ctrl")
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")
    write_dma_ctrl = register("write_dma_ctrl")

    # Golden params read by forward(). Must match the sub-kernel hardware defaults
    # (Scale: scale_factor=1, Offset: offset_value=0) so the golden matches the RTL
    # when a config does not override them — a composite class default does NOT
    # propagate to the sub-kernel registers (only config/project params do), so a
    # non-identity default here would make the golden disagree with the hardware.
    # Configs that set scale_factor/offset_value flow to both golden and RTL.
    default_params = {"N": 1024, "scale_factor": 1, "offset_value": 0}

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
        # Memory-mapped pipeline: PUSH registers the DDR source buffer and PULL the
        # DDR destination buffer with the passive AXI4 slave BFM — both complete
        # only after the DUT masters move the bytes (post-start). Register both
        # buffers FIRST (no deps), then configure + start. (Gating pull/configure on
        # PUSH completion deadlocks on a real simulator; masked by cpu-only testing.)
        h_push = ctx.push_tensor(self.data_in)

        # Pull output — dispatch early so AXI4 BFM has PULL entry
        h_pull = ctx.pull_tensor(self.data_out)

        # Configure all sub-kernels (auto_bind + runtime_params registers)
        h_cfg = ctx.configure(self)

        # Start all sub-kernels
        h_start_rdma = ctx.write_register(self.read_dma_ctrl, {"start": 1}, dep=h_cfg)
        h_start_scale = ctx.write_register(self.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_offset = ctx.write_register(self.offset_ctrl, {"start": 1}, dep=h_cfg)
        h_start_wdma = ctx.write_register(self.write_dma_ctrl, {"start": 1}, dep=h_cfg)

        # Poll all done flags
        h_poll_rdma = ctx.poll_register(self.read_dma_ctrl, "done", dep=h_start_rdma)
        h_poll_scale = ctx.poll_register(self.scale_ctrl, "done", dep=h_start_scale)
        h_poll_offset = ctx.poll_register(self.offset_ctrl, "done", dep=h_start_offset)
        h_poll_wdma = ctx.poll_register(self.write_dma_ctrl, "done", dep=h_start_wdma)

        # Pull commits after all sub-kernels done
        h_pull.add_commit_dependency(h_poll_rdma)
        h_pull.add_commit_dependency(h_poll_scale)
        h_pull.add_commit_dependency(h_poll_offset)
        h_pull.add_commit_dependency(h_poll_wdma)
