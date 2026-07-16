import torch

from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.spec.models import Direction


class StreamDmaV2Kernel(Kernel):
    spec = "kernels/stream_dma_v2/kernel_spec.yaml"

    ctrl = register("ctrl")

    data_in = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="input_stream",
    )
    data_out = Tensor(
        shape=("${N}",),
        dtype=torch.int8,
        interface="dma_port",
        direction=Direction.DEV_TO_HOST,
    )

    def generate_inputs(self, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.data_in.fill_random(generator=rng)

    def run(self, ctx) -> None:
        # Memory-mapped DMA (stream in -> DDR out via AXI master): PUSH arms the
        # input-stream source and PULL registers the DDR destination buffer with
        # the passive AXI4 slave BFM — both complete only after the DUT moves the
        # bytes, which happens after start. Register both buffers FIRST (no deps),
        # then configure + start. (Gating configure on PUSH completion deadlocks on
        # a real simulator; masked previously by cpu-only testing.)
        h_push = ctx.push_tensor(self.data_in)
        h_pull = ctx.pull_tensor(self.data_out)

        h_cfg = ctx.configure(self)
        h_start = ctx.write_register(self.ctrl, {"start": 1}, dep=h_cfg)
        h_poll = ctx.poll_register(self.ctrl, "done", dep=h_start)
        h_pull.add_commit_dependency(h_poll)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        return {"data_out": self.data_in.data.clone()}
