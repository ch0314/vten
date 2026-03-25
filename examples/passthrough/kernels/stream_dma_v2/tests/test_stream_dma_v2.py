import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestStreamDmaV2(TestScenario):
    kernel = "stream_dma_v2"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from stream_dma_v2_kernel import StreamDmaV2Kernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(StreamDmaV2Kernel, N=N)
        k.generate_inputs(seed=42)

        # 1. Load input data to SHM
        h_load = ctx.load_tensor(k.data_in)

        # 2. Configure DUT via AXI-Lite (auto_bind handles addr/length)
        h_cfg = ctx.configure(k, dep=h_load)

        # 3. Start DMA engine
        h_start = ctx.write_register(k.ctrl, {"start": 1}, dep=h_cfg)

        # 4. Stream data + DMA write + poll — all concurrent after start
        h_push = ctx.push_tensor(k.data_in, dep=h_start)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_start)
        h_poll = ctx.poll_register(k.ctrl, "done", dep=h_start)

        # 5. Verify: DMA output == input (passthrough)
        ctx.verify(h_pull, k.forward())