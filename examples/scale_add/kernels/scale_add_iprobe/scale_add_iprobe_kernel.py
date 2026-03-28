"""ScaleAdd with Internal(probe=True) for internal wire golden comparison.

Same as ScaleAddKernel but scale's output_stream is Internal(probe=True),
causing a passive probe BFM to be instantiated on the internal wire.
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

from scale.scale_kernel import ScaleKernel
from offset.offset_kernel import OffsetKernel


class ScaleAddIProbeKernel(CompositeKernel):
    """Composite with internal probe: scale output monitored by passive BFM."""

    scale = ScaleKernel.bind(interface_map={
        "ctrl": ("scale_ctrl", "scale"),
        "input_stream": "input_stream",
        "output_stream": Internal(probe=True),   # ← probe on internal wire
    })
    offset = OffsetKernel.bind(interface_map={
        "ctrl": ("offset_ctrl", "offset"),
        "input_stream": Internal(),
        "output_stream": "output_stream",
    })

    scale_ctrl = register("scale_ctrl")
    offset_ctrl = register("offset_ctrl")

    data_in = scale.data_in.expose("input_stream")
    data_out = offset.data_out.expose("output_stream")

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

    def forward_scale_only(self, scale_factor: int = 2) -> torch.Tensor:
        """Golden for the internal wire (scale output before offset)."""
        x = self.data_in.data.to(torch.int16) * scale_factor
        return x.clamp(-128, 127).to(torch.int8)
