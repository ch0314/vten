"""TestScenarios for scale_add composite kernel.

ScaleAdd: input → Scale(×factor) → Offset(+value) → output
Each sub-kernel has independent AXI-Lite ctrl.

TestScaleAdd: parametrized via configs for parameter sweep.
TestScaleAddProbe: probe=True on pull for beat-level BFM verification.
TestScaleAddInternalProbe: dotted probe on the internal scale→offset wire.
"""

from vten.cli.scenario import TestScenario


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


class TestScaleAddInternalProbe(TestScenario):
    """Probe the INTERNAL scale→offset wire (a dotted/internal probe).

    Contrast with TestScaleAddProbe: that probes the *exposed output*
    (``data_out``) — the offset stage's result. This scenario instead probes
    ``scale.data_out``, the hidden internal tensor consumed by the
    ``scale.data_out >> offset.data_in`` connection. A passive probe BFM taps
    that internal wire and compares each beat against golden mid-pipeline data
    (the scale stage's output, BEFORE offset is applied).

    NO new RTL: this reuses the existing scale + offset DUTs of the scale_add
    composite. The dotted probe is declared via the ``probes`` field below.

    How the golden is supplied
    --------------------------
    A TestScenario is *pure declarative config* — the CLI executes the KERNEL's
    run(), not a scenario method (see vten/cli/scenario.py and
    vten/execution.py::execute_batch). The declarative ``probes`` field IS
    applied by the CLI (execution.py registers it before inst.run()), but the
    scale_add composite defines a *custom* forward() with no auto-chained
    ``_golden_pool``, so the framework cannot auto-extract the internal-wire
    golden (vten/runtime/probe_manager.py::resolve_internal_probe_golden is a
    no-op without a pool). ScaleAddKernel.run() therefore seeds the golden with
    ctx.set_internal_probe_golden("scale", "data_out", scaled) *whenever this
    probe is requested* (it inspects ctx._declarative_probes). That call is what
    upgrades the INTERNAL mapping to INTERNAL_PROBE in engine.compile().

    VERIFY with a real backend:
        vten run --kernel scale_add --test TestScaleAddInternalProbe \
                 --backend verilator --verify
    """

    kernel = "scale_add"

    # Declarative dotted probe: "<sub_kernel>.<tensor>" taps the internal wire.
    # ScaleAddKernel.run() seeds the matching golden when it sees this request.
    probes = ["scale.data_out"]
