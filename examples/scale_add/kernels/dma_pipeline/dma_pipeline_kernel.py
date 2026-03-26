"""DMA Pipeline — ReadDMA → Scale → Offset → WriteDMA.

Full memory-mapped pipeline: reads from DDR, applies scale + offset,
writes back to DDR. Like a Vitis multi-IP dataflow.
"""

import sys
from pathlib import Path

import torch

from vten.kernel.composite import CompositeKernel, Connect, Internal
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

    Vitis-flow: no kernel_spec.yaml. Each sub-kernel keeps its own
    AXI-Lite ctrl port (independent ctrl pattern).
    Memory interfaces: ReadDMA reads from DDR, WriteDMA writes to DDR.
    Stream interfaces are all internal wires between stages.
    """

    # Sub-kernel bindings
    read_dma = ReadDMAKernel.bind(interface_map={
        "ctrl": ("rdma_ctrl", "read_dma"),
        "mem_port": "mem_in",               # External AXI4 read port
        "output_stream": Internal(),         # Internal → scale
    })
    scale = ScaleKernel.bind(interface_map={
        "ctrl": ("scale_ctrl", "scale"),
        "input_stream": Internal(),          # Internal ← read_dma
        "output_stream": Internal(),         # Internal → offset
    })
    offset = OffsetKernel.bind(interface_map={
        "ctrl": ("offset_ctrl", "offset"),
        "input_stream": Internal(),          # Internal ← scale
        "output_stream": Internal(),         # Internal → write_dma
    })
    write_dma = WriteDMAKernel.bind(interface_map={
        "ctrl": ("wdma_ctrl", "write_dma"),
        "input_stream": Internal(),          # Internal ← offset
        "mem_port": "mem_out",              # External AXI4 write port
    })

    # Register proxies
    rdma_ctrl = register("rdma_ctrl")
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")
    wdma_ctrl = register("wdma_ctrl")

    # Expose tensors: ReadDMA input from DDR, WriteDMA output to DDR
    data_in = read_dma.data_in.expose("mem_in")
    data_out = write_dma.data_out.expose("mem_out")

    # Internal connections: ReadDMA → Scale → Offset → WriteDMA
    connections = [
        Connect(read_dma.data_out, scale.data_in),
        Connect(scale.data_out, offset.data_in),
        Connect(offset.data_out, write_dma.data_in),
    ]

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, scale_factor: int = 2, offset_value: int = 1) -> torch.Tensor:
        """Golden: read → scale → offset → write (identity DMA on both ends)."""
        x = self.data_in.data.to(torch.int16) * scale_factor
        x = x.clamp(-128, 127)
        x = x + offset_value
        return x.clamp(-128, 127).to(torch.int8)
