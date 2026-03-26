import sys
from pathlib import Path

import torch

from vten.kernel.composite import CompositeKernel, Connect, Internal
from vten.kernel.register import register

# Import sub-kernels
_kernel_base = str(Path(__file__).resolve().parent.parent)
if _kernel_base not in sys.path:
    sys.path.insert(0, _kernel_base)

from scale.scale_kernel import ScaleKernel
from offset.offset_kernel import OffsetKernel


class ScaleAddKernel(CompositeKernel):
    """Composite: scale(x factor) → offset(+ value) pipeline.

    Vitis-flow: no kernel_spec.yaml. Sub-kernels are independently verified.
    Framework auto-generates wrapper-of-wrappers from connectivity description.
    Each sub-kernel keeps its own AXI-Lite ctrl port (independent ctrl pattern).
    """

    # No spec — synthesized from sub-kernel specs

    # Sub-kernel bindings with interface mapping
    # Independent ctrl: each sub-kernel exposes its own AXI-Lite interface
    scale = ScaleKernel.bind(interface_map={
        "ctrl": ("scale_ctrl", "scale"),     # Independent AXI-Lite port
        "input_stream": "input_stream",      # External
        "output_stream": Internal(),         # Internal wire to offset
    })
    offset = OffsetKernel.bind(interface_map={
        "ctrl": ("offset_ctrl", "offset"),   # Independent AXI-Lite port
        "input_stream": Internal(),          # Internal wire from scale
        "output_stream": "output_stream",    # External
    })

    # Register proxies for each sub-kernel's ctrl
    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")

    # Expose sub-kernel tensors as top-level I/O
    data_in = scale.data_in.expose("input_stream")
    data_out = offset.data_out.expose("output_stream")

    # Internal RTL connection: scale output → offset input
    connections = [Connect(scale.data_out, offset.data_in)]

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, scale_factor: int = 2, offset_value: int = 1) -> torch.Tensor:
        """Golden: scale then offset, both with signed int8 saturation."""
        x = self.data_in.data.to(torch.int16) * scale_factor
        x = x.clamp(-128, 127)
        x = x + offset_value
        return x.clamp(-128, 127).to(torch.int8)
