"""xsim E2E tests for Multi-Config and Multi-Batch Session.

Category E: Multi-invocation (single-batch multi-config + multi-batch session)
Category F: Edge cases (broken passthrough)

Requires xsim (Vivado simulator) to be available.
Run with: pytest tests/test_xsim_e2e.py -v -m xsim
Skip with: pytest tests/ -m "not xsim"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from vten.backend.registry import get_backend, resolve_backend_name
from vten.cli.config import load_project_config
from vten.errors import VerificationError
from vten.runtime.context import ExecutionContext
from vten.spec.parser import parse_kernel_spec

# ── Helpers ──


def _add_kernel_path(kernel_dir: str | Path) -> None:
    """Add kernel directory to sys.path for local imports."""
    kernel_dir = str(Path(kernel_dir).resolve())
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)


def _load_passthrough_env(kernel_name: str = "passthrough",
                          project_name: str = "passthrough"):
    """Load a passthrough-style project environment with the given kernel build dir.

    ``project_name`` selects the example project under ``examples/``. The
    default ``passthrough`` project holds the working kernel; the
    ``broken_passthrough`` kernel lives in the ``passthrough_regression``
    project.
    """
    from vten.backend.base import RunContext
    project = Path("examples") / project_name
    project = project.resolve()
    config = load_project_config(project)
    backend_name = resolve_backend_name(config)
    backend = get_backend(backend_name, config)
    backend.set_run_context(RunContext(
        project_dir=project,
        kernel_build_dir=project / "kernels" / kernel_name / "build",
    ))
    return project, config, backend


def _load_scale_add_env(kernel_name: str = "scale_add"):
    """Load scale_add project environment with specified kernel build dir."""
    from vten.backend.base import RunContext
    project = Path("examples/scale_add").resolve()
    config = load_project_config(project)
    backend_name = resolve_backend_name(config)
    backend = get_backend(backend_name, config)
    backend.set_run_context(RunContext(
        project_dir=project,
        kernel_build_dir=project / "kernels" / kernel_name / "build",
    ))
    return project, config, backend


# ── Kernel imports (lazy, after sys.path setup) ──


def _get_passthrough_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "passthrough")
    from passthrough_kernel import PassthroughKernel
    return PassthroughKernel



def _get_broken_passthrough_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "broken_passthrough")
    from broken_passthrough_kernel import BrokenPassthroughKernel
    return BrokenPassthroughKernel


def _get_scale_add_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "scale_add")
    from scale_add_kernel import ScaleAddKernel
    return ScaleAddKernel


# ═══════════════════════════════════════════════════════════════════
# Category E: Multi-Invocation xsim E2E
# ═══════════════════════════════════════════════════════════════════


class TestMultiConfigE2E:
    """E1: Single-batch multi-config with real xsim backend."""

    @pytest.mark.xsim
    def test_multi_config_passthrough_3x(self):
        """E1.1: Same passthrough config ×3 in single batch with BARRIERs."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            ctx = ExecutionContext(backend=backend, project_params={"N": 1024})

            goldens = []
            for i in range(3):
                ki = ctx.instantiate(KernelClass, spec=spec, N=1024)
                ki.generate_inputs(seed=42 + i)
                h_push = ctx.push_tensor(ki.data_in)
                ctx.pull_tensor(ki.data_out, dep=h_push)
                goldens.append(ki.forward()["data_out"])
                ctx.config_boundary()

            result = ctx.run(verify=True)
            assert result.status == "DONE"
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

    @pytest.mark.xsim
    def test_multi_config_different_sizes(self):
        """E1.2: Different tensor sizes (N=32, N=64, N=128) in one batch."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            ctx = ExecutionContext(backend=backend, project_params={"N": 32})

            for n_val in [32, 64, 128]:
                ctx._project_params = {"N": n_val}
                ki = ctx.instantiate(KernelClass, spec=spec, N=n_val)
                ki.generate_inputs(seed=42)
                h_push = ctx.push_tensor(ki.data_in)
                ctx.pull_tensor(ki.data_out, dep=h_push)
                ctx.config_boundary()

            result = ctx.run(verify=True)
            assert result.status == "DONE"
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

    @pytest.mark.xsim
    def test_multi_config_scale_add_sweep(self):
        """E1.3: ScaleAdd composite with 3 different param configs in 1 batch."""
        project, config, backend = _load_scale_add_env()
        KernelClass = _get_scale_add_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            configs = [
                {"scale_factor": 1, "offset_value": 0},   # identity
                {"scale_factor": 2, "offset_value": 1},   # default
                {"scale_factor": 3, "offset_value": -2},   # larger scale + neg offset
            ]
            N = 1024
            total_beats = N // 32

            ctx = ExecutionContext(backend=backend, project_params={"N": N})

            for cfg in configs:
                off_val = cfg["offset_value"] & 0xFF  # 8-bit register
                ki = ctx.instantiate(
                    KernelClass, N=N,
                    scale_factor=cfg["scale_factor"],
                    offset_value=off_val,
                )
                ki.generate_inputs(seed=42)

                h_push = ctx.push_tensor(ki.data_in)

                # ctx.configure auto-writes runtime_params with register mappings
                h_cfg = ctx.configure(ki, dep=h_push)

                h_pull = ctx.pull_tensor(ki.data_out, dep=h_cfg)

                h_start_s = ctx.write_register(
                    ki.scale_ctrl, {"start": 1}, dep=h_cfg,
                )
                h_start_o = ctx.write_register(
                    ki.offset_ctrl, {"start": 1}, dep=h_cfg,
                )

                h_poll_s = ctx.poll_register(ki.scale_ctrl, "done", dep=h_start_s)
                h_poll_o = ctx.poll_register(ki.offset_ctrl, "done", dep=h_start_o)
                h_pull.add_commit_dependency(h_poll_s)
                h_pull.add_commit_dependency(h_poll_o)

                ctx.config_boundary()

            result = ctx.run(verify=True)
            assert result.status == "DONE"
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()


class TestMultiBatchSessionE2E:
    """E2: Multi-batch session with real xsim backend."""

    @pytest.mark.xsim
    def test_session_passthrough_3batch(self):
        """E2.1: 3 sequential batches — open, submit×2, close."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            with backend:
                for i in range(3):
                    ctx = ExecutionContext(backend=backend, project_params={"N": 1024})
                    ki = ctx.instantiate(KernelClass, spec=spec, N=1024)
                    ki.generate_inputs(seed=42 + i)
                    h_push = ctx.push_tensor(ki.data_in)
                    ctx.pull_tensor(ki.data_out, dep=h_push)
                    result = ctx.run(verify=True)
                    assert result.status == "DONE"
        finally:
            os.chdir(prev_cwd)

    @pytest.mark.xsim
    def test_session_growing_tensor_sizes(self):
        """E2.6: Multi-batch with increasing tensor sizes triggers SHM resize.

        Batch 1: N=256 (small SHM), Batch 2: N=1024 (4x larger SHM),
        Batch 3: N=4096 (16x larger SHM). Each batch must trigger
        ftruncate + remap on the C bridge side.
        """
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            with backend:
                for N in [256, 1024, 4096]:
                    ctx = ExecutionContext(backend=backend, project_params={"N": N})
                    ki = ctx.instantiate(KernelClass, spec=spec, N=N)
                    ki.generate_inputs(seed=N)
                    h_push = ctx.push_tensor(ki.data_in)
                    ctx.pull_tensor(ki.data_out, dep=h_push)
                    result = ctx.run(verify=True)
                    assert result.status == "DONE", f"Failed at N={N}"
        finally:
            os.chdir(prev_cwd)



# ═══════════════════════════════════════════════════════════════════
# Category F: Edge Cases xsim E2E
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """F: Edge cases — broken passthrough, tensor size boundaries."""

    @pytest.mark.xsim
    def test_broken_passthrough_verify_fail(self):
        """F1: broken_passthrough detects host-side verification failure."""
        project, config, backend = _load_passthrough_env(
            "broken_passthrough", project_name="passthrough_regression"
        )
        spec = parse_kernel_spec(
            project / "kernels" / "broken_passthrough" / "kernel_spec.yaml"
        )
        KernelClass = _get_broken_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            ctx = ExecutionContext(backend=backend, project_params={"N": 1024})
            ki = ctx.instantiate(KernelClass, spec=spec, N=1024)
            ki.generate_inputs(seed=42)
            h_push = ctx.push_tensor(ki.data_in)
            ctx.pull_tensor(ki.data_out, dep=h_push)

            with pytest.raises(VerificationError):
                # forward() returns correct data, but RTL XORs with 0x01
                ctx.run(verify=True)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

    @pytest.mark.xsim
    def test_broken_passthrough_probe_fail(self):
        """F2: broken_passthrough with probe=True detects BFM-level mismatch."""
        project, config, backend = _load_passthrough_env(
            "broken_passthrough", project_name="passthrough_regression"
        )
        spec = parse_kernel_spec(
            project / "kernels" / "broken_passthrough" / "kernel_spec.yaml"
        )
        KernelClass = _get_broken_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            ctx = ExecutionContext(backend=backend, project_params={"N": 1024})
            ki = ctx.instantiate(KernelClass, spec=spec, N=1024)
            ki.generate_inputs(seed=42)
            h_push = ctx.push_tensor(ki.data_in)
            # probe=True triggers golden comparison in BFM
            ctx.pull_tensor(ki.data_out, dep=h_push, probe=True)

            with pytest.raises(Exception):
                # Either VerificationError from host or BackendError
                # from PROBE_MISMATCH — both indicate failure detection
                ctx.run(verify=True)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

