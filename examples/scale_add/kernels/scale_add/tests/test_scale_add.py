import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestScaleAdd(TestScenario):
    """Composite kernel: scale(x2) then offset(+1).

    Independent ctrl: each sub-kernel has its own AXI-Lite interface.
    """

    kernel = "scale_add"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from scale_add_kernel import ScaleAddKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(ScaleAddKernel, N=N)
        k.generate_inputs(seed=42)

        total_beats = N // 32

        h_load = ctx.load_tensor(k.data_in)

        # Configure scale sub-kernel
        h_sf = ctx.write_register(k.scale_ctrl, {"scale_factor": 2}, dep=h_load)
        h_len_s = ctx.write_register(k.scale_ctrl, {"length": total_beats}, dep=h_sf)

        # Configure offset sub-kernel
        h_ov = ctx.write_register(k.offset_ctrl, {"offset_value": 1}, dep=h_load)
        h_len_o = ctx.write_register(k.offset_ctrl, {"length": total_beats}, dep=h_ov)

        # All config done — activate BFMs
        h_push = ctx.push_tensor(k.data_in, dep=[h_len_s, h_len_o])
        h_pull = ctx.pull_tensor(k.data_out, dep=[h_len_s, h_len_o])

        # Start both sub-kernels
        h_start_s = ctx.write_register(k.scale_ctrl, {"start": 1}, dep=[h_len_s, h_len_o])
        h_start_o = ctx.write_register(k.offset_ctrl, {"start": 1}, dep=[h_len_s, h_len_o])

        # Poll both done flags
        h_poll_s = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)

        ctx.verify(h_pull, k.forward(scale_factor=2, offset_value=1))
