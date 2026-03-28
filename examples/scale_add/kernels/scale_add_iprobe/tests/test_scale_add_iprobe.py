"""TestScenario for ScaleAdd with internal probe.

Tests that a passive probe BFM on the internal wire (scale→offset)
correctly compares RTL data against golden reference.
"""

import sys
from pathlib import Path

import torch

from vten.cli.run import TestScenario


class TestScaleAddIProbe(TestScenario):
    """Internal probe: golden comparison on scale→offset internal wire."""

    kernel = "scale_add_iprobe"

    configs = [
        {"name": "default"},
        {"name": "identity", "scale_factor": 1},
    ]

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from scale_add_iprobe_kernel import ScaleAddIProbeKernel

        N = cfg.get("N", 1024)
        scale_factor = cfg.get("scale_factor", 2)
        offset_value = cfg.get("offset_value", 1)

        k = ctx.instantiate(ScaleAddIProbeKernel, N=N)
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

        # Activate BFMs
        h_push = ctx.push_tensor(k.data_in, dep=[h_len_s, h_len_o])
        h_pull = ctx.pull_tensor(k.data_out, dep=[h_len_s, h_len_o])

        # Start both sub-kernels
        h_start_s = ctx.write_register(
            k.scale_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
        )
        h_start_o = ctx.write_register(
            k.offset_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
        )

        # Poll done
        h_poll_s = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)

        # Register internal probe golden: scale sub-kernel's output
        # This is the data that flows on the internal wire (before offset)
        scale_golden = k.forward_scale_only(scale_factor=scale_factor)
        ctx.set_internal_probe_golden("scale", "data_out", scale_golden)

        # Verify final output
        golden_off = offset_value if offset_value < 128 else offset_value - 256
        ctx.verify(h_pull, k.forward(
            scale_factor=scale_factor, offset_value=golden_off,
        ))


class TestScaleAddIProbeAbort(TestScenario):
    """Intentionally wrong golden → probe should trigger early abort."""

    kernel = "scale_add_iprobe"

    configs = [{"name": "broken_golden"}]

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from scale_add_iprobe_kernel import ScaleAddIProbeKernel

        N = cfg.get("N", 1024)
        scale_factor = 2
        offset_value = 1

        k = ctx.instantiate(ScaleAddIProbeKernel, N=N)
        k.generate_inputs(seed=42)

        total_beats = N // 32

        h_load = ctx.load_tensor(k.data_in)
        h_sf = ctx.write_register(
            k.scale_ctrl, {"scale_factor": scale_factor}, dep=h_load,
        )
        h_len_s = ctx.write_register(
            k.scale_ctrl, {"length": total_beats}, dep=h_sf,
        )
        h_ov = ctx.write_register(
            k.offset_ctrl, {"offset_value": offset_value}, dep=h_load,
        )
        h_len_o = ctx.write_register(
            k.offset_ctrl, {"length": total_beats}, dep=h_ov,
        )
        h_push = ctx.push_tensor(k.data_in, dep=[h_len_s, h_len_o])
        h_pull = ctx.pull_tensor(k.data_out, dep=[h_len_s, h_len_o])
        h_start_s = ctx.write_register(
            k.scale_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
        )
        h_start_o = ctx.write_register(
            k.offset_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
        )
        h_poll_s = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)

        # WRONG golden: all zeros instead of actual scale output
        wrong_golden = torch.zeros(N, dtype=torch.int8)
        ctx.set_internal_probe_golden("scale", "data_out", wrong_golden)

        ctx.verify(h_pull, k.forward(
            scale_factor=scale_factor, offset_value=offset_value,
        ))
