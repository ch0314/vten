import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class MultiPortScatterKernel(Kernel):
    spec = "kernels/multi_port_scatter/kernel_spec.yaml"

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="din",
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="dout",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        # Combinational multi-port DUT: each split port wires s_axis_tready =
        # m_axis_tready. PUSH and PULL must be issued CONCURRENTLY (the framework
        # fans them out per block_split port) — sequencing pull after push (an
        # issue-dep waits for the dep to COMMIT while the slave BFM only asserts
        # tready during a PULL) deadlocks on a real simulator. Masked by cpu-only.
        ctx.push_tensor(self.data_in)
        ctx.pull_tensor(self.data_out)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden: passthrough (output == input)."""
        return {"data_out": self.data_in.data.clone()}
