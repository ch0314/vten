import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor


class Narrow32Kernel(Kernel):
    spec = "kernels/narrow32/kernel_spec.yaml"

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream",
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="output_stream",
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        return {"data_out": self.data_in.data.clone()}

    def run(self, ctx) -> None:
        # Combinational DUT (s_axis_tready = m_axis_tready): PUSH and PULL must be
        # issued CONCURRENTLY — sequencing pull after push (an issue-dep waits for
        # the dep to COMMIT, and the slave BFM only asserts tready while a PULL is
        # active) deadlocks on a real simulator. Previously masked by cpu-only testing.
        ctx.push_tensor(self.data_in)
        ctx.pull_tensor(self.data_out)
