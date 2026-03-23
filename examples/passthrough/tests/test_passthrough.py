"""Passthrough E2E test scenario.

Spec reference: 07_e2e_examples.md §1.4
"""

import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestPassthrough(TestScenario):
    kernel = "passthrough"

    def run(self, ctx, cfg):
        # Ensure project root's kernels/ is importable
        project_dir = str(Path(__file__).resolve().parent.parent)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        from kernels.passthrough_kernel import PassthroughKernel

        k = ctx.instantiate(PassthroughKernel, N=cfg.get("N", 1024))
        k.generate_inputs(seed=42)

        h_load = ctx.load_tensor(k.data_in)
        h_push = ctx.push_tensor(k.data_in, dep=h_load)
        h_pull = ctx.pull_tensor(k.data_out, dep=h_load)
        ctx.verify(h_pull, k.forward())
