import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor


class BrokenPassthroughKernel(Kernel):
    """Kernel for a passthrough DUT that intentionally corrupts data.

    The RTL XORs every byte with 0x01, but forward() returns the
    CORRECT (uncorrupted) expected output so that mismatches are detected
    both at host-side verify and at BFM probe level.
    """

    spec = "kernels/broken_passthrough/kernel_spec.yaml"

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
        # Return correct passthrough (data unchanged).
        # The RTL will corrupt data, so verify/probe will detect mismatches.
        return {"data_out": self.data_in.data.clone()}

    def run(self, ctx) -> None:
        # Combinational DUT (s_axis_tready = m_axis_tready): PUSH and PULL must
        # be issued CONCURRENTLY — sequencing pull after push (an issue-dep
        # waits for the dep to COMMIT) deadlocks on a real simulator, and the
        # intended verify/probe mismatch is never reached.
        ctx.push_tensor(self.data_in)
        ctx.pull_tensor(self.data_out)
