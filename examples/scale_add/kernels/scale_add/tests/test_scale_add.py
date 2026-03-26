"""TestScenarios for scale_add composite kernel.

ScaleAdd: input → Scale(×factor) → Offset(+value) → output
Each sub-kernel has independent AXI-Lite ctrl.

TestScaleAdd: parametrized via configs for parameter sweep.
TestScaleAddProbe: probe=True on pull for beat-level BFM verification.
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestScaleAdd(TestScenario):
    """Composite kernel: scale then offset, with parameter sweep."""

    kernel = "scale_add"

    configs = [
        {"name": "default"},                                        # N=1024, scale=2, off=1
        {"name": "identity", "scale_factor": 1, "offset_value": 0}, # pass-through
        {"name": "big_scale", "scale_factor": 5, "offset_value": 3},
        {"name": "small_n", "N": 32},                               # 1 beat
        {"name": "large_n", "N": 4096},                             # 128 beats
        {"name": "negative_off", "offset_value": 251},              # -5 as uint8 (0xFB)
    ]

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from scale_add_kernel import ScaleAddKernel

        N = cfg.get("N", 1024)
        scale_factor = cfg.get("scale_factor", 2)
        offset_value = cfg.get("offset_value", 1)

        k = ctx.instantiate(ScaleAddKernel, N=N)
        k.generate_inputs(seed=42)

        total_beats = N // 32

        h_load = ctx.load_tensor(k.data_in)

        # Configure scale sub-kernel
        h_sf = ctx.write_register(
            k.scale_ctrl, {"scale_factor": scale_factor}, dep=h_load,
        )
        h_len_s = ctx.write_register(
            k.scale_ctrl, {"length": total_beats}, dep=h_sf,
        )

        # Configure offset sub-kernel
        h_ov = ctx.write_register(
            k.offset_ctrl, {"offset_value": offset_value}, dep=h_load,
        )
        h_len_o = ctx.write_register(
            k.offset_ctrl, {"length": total_beats}, dep=h_ov,
        )

        # All config done — activate BFMs
        h_push = ctx.push_tensor(k.data_in, dep=[h_len_s, h_len_o])
        h_pull = ctx.pull_tensor(k.data_out, dep=[h_len_s, h_len_o])

        # Start both sub-kernels
        h_start_s = ctx.write_register(
            k.scale_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
        )
        h_start_o = ctx.write_register(
            k.offset_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
        )

        # Poll both done flags
        h_poll_s = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)

        # Golden: offset_value in register is 8-bit → handle wrap for negative
        golden_off = offset_value if offset_value < 128 else offset_value - 256
        ctx.verify(h_pull, k.forward(
            scale_factor=scale_factor, offset_value=golden_off,
        ))


class TestScaleAddProbe(TestScenario):
    """ScaleAdd with probe=True on pull for beat-level BFM verification."""

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

        h_sf = ctx.write_register(k.scale_ctrl, {"scale_factor": 2}, dep=h_load)
        h_len_s = ctx.write_register(k.scale_ctrl, {"length": total_beats}, dep=h_sf)

        h_ov = ctx.write_register(k.offset_ctrl, {"offset_value": 1}, dep=h_load)
        h_len_o = ctx.write_register(k.offset_ctrl, {"length": total_beats}, dep=h_ov)

        h_push = ctx.push_tensor(k.data_in, dep=[h_len_s, h_len_o])
        h_pull = ctx.pull_tensor(k.data_out, dep=[h_len_s, h_len_o], probe=True)

        h_start_s = ctx.write_register(k.scale_ctrl, {"start": 1}, dep=[h_len_s, h_len_o])
        h_start_o = ctx.write_register(k.offset_ctrl, {"start": 1}, dep=[h_len_s, h_len_o])

        h_poll_s = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)

        ctx.verify(h_pull, k.forward(scale_factor=2, offset_value=1))
