import torch

from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class StreamScatterKernel(Kernel):
    spec = "kernels/stream_scatter/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream",
    )
    result_0 = Tensor(
        shape=("${N}//2",),
        dtype=torch.int8,
        interface="hbm_0",
        direction=Direction.DEV_TO_HOST,
    )
    result_1 = Tensor(
        shape=("${N}//2",),
        dtype=torch.int8,
        interface="hbm_1",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        # Memory-mapped scatter (stream in -> two HBM ports via AXI masters): PUSH
        # only arms the input-stream source and completes after the DUT drains it
        # (post-start), so configure must NOT gate on PUSH completion or the graph
        # deadlocks on a real simulator (masked previously by cpu-only testing).
        # The per-port pulls stay barrier-gated before start with commit-deps on
        # done, preserving the two-port scatter ordering.
        h_push = ctx.push_tensor(self.data_in)
        h_cfg = ctx.configure(self)

        total_beats = self.N // 32
        h_len = ctx.write_register(self.ctrl, {"length": total_beats}, dep=h_cfg)

        h_barrier = ctx.barrier()

        h_pull_0 = ctx.pull_tensor(self.result_0, dep=h_barrier)
        h_pull_1 = ctx.pull_tensor(self.result_1, dep=h_barrier)

        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_barrier)
        h_read = ctx.read_register(self.ctrl, "count", dep=h_start)

        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull_0.add_commit_dependency(h_poll)
        h_pull_1.add_commit_dependency(h_poll)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Golden reference: scale by 2 (saturating), scatter to 2 ports."""
        data = inputs.get("data_in", self.data_in.data).to(torch.int16) * 2
        data = data.clamp(-128, 127).to(torch.int8)

        # Reshape into beats (32 elements per beat), then scatter
        elems_per_beat = 32
        beats = data.reshape(-1, elems_per_beat)
        even_beats = beats[0::2].reshape(-1)  # port 0
        odd_beats = beats[1::2].reshape(-1)   # port 1
        return {"result_0": even_beats, "result_1": odd_beats}
