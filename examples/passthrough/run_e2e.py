#!/usr/bin/env python3
"""Passthrough E2E: compile → SHM → xsim → verify."""

import os
import sys
import struct
import time

# Add vten root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from vten.backend.xsim import XsimBackend
from vten.runtime.context import ExecutionContext

# Change to passthrough project dir for spec loading
os.chdir(os.path.dirname(__file__))

from kernels.passthrough_kernel import PassthroughKernel

VTEN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def main():
    project_config = {
        "project": {"name": "passthrough"},
        "rtl": {"top_module": "tb_top"},
        "backend": {
            "xsim": {
                "vivado_path": "/tools/Xilinx/Vivado/2023.2",
                "submit_timeout_s": 60,
                "timeout_ms": 10000,
            },
        },
        # xsim must run from the dir containing xsim.dir/
        "_xsim_dir": VTEN_ROOT,
        "_project_dir": os.path.dirname(__file__),
    }

    backend = XsimBackend(project_config)

    ctx = ExecutionContext(
        backend=backend,
        project_params={"N": 64},  # 64 bytes = 2 beats of 256-bit
    )

    ki = ctx.instantiate(PassthroughKernel, N=64)
    ki.generate_inputs(seed=42)

    # Record ops: load → push + pull (concurrent — passthrough needs both sides active)
    h_load = ctx.load_tensor(ki.data_in)
    h_push = ctx.push_tensor(ki.data_in, dep=h_load)
    h_pull = ctx.pull_tensor(ki.data_out, dep=h_load)  # depends on LOAD, not PUSH

    print("[E2E] Compiling and submitting to xsim...")
    try:
        result = ctx.run()
        print(f"[E2E] Result: status={result.status}, total_cycles={result.total_cycles}")
        print(f"[E2E] Per-command stats: {len(result.per_command_stats)} commands")
        for s in result.per_command_stats:
            print(f"  cmd {s.cmd_id}: status={s.status} "
                  f"issue={s.issue_cycle} commit={s.commit_cycle} "
                  f"active={s.active_cycles} stalls={s.stall_cycles}")
    except Exception as e:
        print(f"[E2E] Error: {type(e).__name__}: {e}")
        # Try to get xsim output
        if hasattr(backend, '_process') and backend._process:
            try:
                backend._process.terminate()
                stdout, stderr = backend._process.communicate(timeout=10)
                out = stdout.decode(errors='replace')
                print(f"[xsim stdout] ({len(out)} chars, last 4000):")
                print(out[-4000:])
                if stderr:
                    print("[xsim stderr]:", stderr.decode(errors='replace')[-1000:])
            except Exception as e2:
                print(f"[xsim output err]: {e2}")
        raise
    finally:
        try:
            backend.shutdown()
        except Exception:
            pass
        backend.cleanup()


if __name__ == "__main__":
    main()
