"""TestScenarios for scale_add composite kernel.

ScaleAdd: input → Scale(×factor) → Offset(+value) → output
Each sub-kernel has independent AXI-Lite ctrl.

TestScaleAdd: parametrized via configs for parameter sweep.
TestScaleAddProbe: probe=True on pull for beat-level BFM verification.
"""

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


class TestScaleAddProbe(TestScenario):
    """ScaleAdd with probe=True on pull for beat-level BFM verification."""

    kernel = "scale_add"

    def run(self, ctx, cfg):
        import sys
        from pathlib import Path

        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from scale_add_kernel import ScaleAddKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(ScaleAddKernel, N=N)
        k.generate_inputs(seed=42)

        h_push = ctx.push_tensor(k.data_in)
        h_cfg = ctx.configure(k, dep=h_push)

        h_pull = ctx.pull_tensor(k.data_out, dep=h_cfg, probe=True)

        h_start_s = ctx.write_register(k.scale_ctrl, {"start": 1}, dep=h_cfg)
        h_start_o = ctx.write_register(k.offset_ctrl, {"start": 1}, dep=h_cfg)

        h_poll_s = ctx.poll_register(k.scale_ctrl, "done", dep=h_start_s)
        h_poll_o = ctx.poll_register(k.offset_ctrl, "done", dep=h_start_o)
        h_pull.add_commit_dependency(h_poll_s)
        h_pull.add_commit_dependency(h_poll_o)
