"""xsim E2E tests for Functional API, Multi-Config, Multi-Batch Session.

Category D: Functional API (run_kernel / KernelExecutor) with real xsim backend
Category E: Multi-invocation (single-batch multi-config + multi-batch session)
Category F: Edge cases (broken passthrough, tensor size boundaries)

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
from vten.functional import KernelExecutor, run_kernel
from vten.runtime.context import ExecutionContext
from vten.spec.parser import parse_kernel_spec

# ── Helpers ──


def _add_kernel_path(kernel_dir: str | Path) -> None:
    """Add kernel directory to sys.path for local imports."""
    kernel_dir = str(Path(kernel_dir).resolve())
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)


def _load_passthrough_env(kernel_name: str = "passthrough"):
    """Load passthrough project environment with specified kernel build dir."""
    project = Path("examples/passthrough").resolve()
    config = load_project_config(project)
    config["_project_dir"] = str(project)
    config["_kernel_build_dir"] = str(project / "kernels" / kernel_name / "build")
    backend_name = resolve_backend_name(config)
    backend = get_backend(backend_name, config)
    return project, config, backend


def _load_scale_add_env(kernel_name: str = "scale_add"):
    """Load scale_add project environment with specified kernel build dir."""
    project = Path("examples/scale_add").resolve()
    config = load_project_config(project)
    config["_project_dir"] = str(project)
    config["_kernel_build_dir"] = str(project / "kernels" / kernel_name / "build")
    backend_name = resolve_backend_name(config)
    backend = get_backend(backend_name, config)
    return project, config, backend


# ── Kernel imports (lazy, after sys.path setup) ──


def _get_passthrough_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "passthrough")
    from passthrough_kernel import PassthroughKernel
    return PassthroughKernel


def _get_narrow8_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "narrow8")
    from narrow8_kernel import Narrow8Kernel
    return Narrow8Kernel


def _get_wide512_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "wide512")
    from wide512_kernel import Wide512Kernel
    return Wide512Kernel


def _get_unaligned_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "unaligned")
    from unaligned_kernel import UnalignedKernel
    return UnalignedKernel


def _get_broken_passthrough_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "broken_passthrough")
    from broken_passthrough_kernel import BrokenPassthroughKernel
    return BrokenPassthroughKernel


def _get_scale_add_kernel(project: Path):
    _add_kernel_path(project / "kernels" / "scale_add")
    from scale_add_kernel import ScaleAddKernel
    return ScaleAddKernel


# ═══════════════════════════════════════════════════════════════════
# Category D: Functional API xsim E2E
# ═══════════════════════════════════════════════════════════════════


class TestRunKernel:
    """D1: run_kernel() one-shot with real xsim backend."""

    @pytest.mark.xsim
    def test_run_kernel_passthrough(self):
        """D1.1: Basic passthrough via run_kernel() — 256-bit bus."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
            outputs = run_kernel(
                KernelClass, {"data_in": x},
                backend=backend, spec=spec, params={"N": 1024},
            )
            assert torch.equal(outputs["data_out"], x)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

    @pytest.mark.xsim
    def test_run_kernel_narrow8(self):
        """D1.2: Narrowest bus width (8-bit, 1 element/beat)."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "narrow8" / "kernel_spec.yaml")
        KernelClass = _get_narrow8_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            x = torch.randint(-128, 127, (512,), dtype=torch.int8)
            outputs = run_kernel(
                KernelClass, {"data_in": x},
                backend=backend, spec=spec, params={"N": 512},
            )
            assert torch.equal(outputs["data_out"], x)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

    @pytest.mark.xsim
    def test_run_kernel_wide512(self):
        """D1.3: Widest bus (512-bit, 64 elements/beat)."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "wide512" / "kernel_spec.yaml")
        KernelClass = _get_wide512_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
            outputs = run_kernel(
                KernelClass, {"data_in": x},
                backend=backend, spec=spec, params={"N": 1024},
            )
            assert torch.equal(outputs["data_out"], x)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

    @pytest.mark.xsim
    def test_run_kernel_unaligned(self):
        """D1.4: Non-beat-aligned tensor (N=100 on 256-bit bus)."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "unaligned" / "kernel_spec.yaml")
        KernelClass = _get_unaligned_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            x = torch.randint(-128, 127, (100,), dtype=torch.int8)
            outputs = run_kernel(
                KernelClass, {"data_in": x},
                backend=backend, spec=spec, params={"N": 100},
            )
            assert torch.equal(outputs["data_out"], x)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()


class TestKernelExecutorSingle:
    """D3: KernelExecutor single-call with real xsim backend."""

    @pytest.mark.xsim
    def test_executor_single_passthrough(self):
        """D3.1: KernelExecutor single call — same as run_kernel but via executor."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            npu = KernelExecutor(
                KernelClass, backend=backend, spec=spec, params={"N": 1024},
            )
            x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
            outputs = npu(data_in=x)
            assert torch.equal(outputs["data_out"], x)
            npu.close()
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()


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
                h_load = ctx.load_tensor(ki.data_in)
                ctx.push_tensor(ki.data_in, dep=h_load)
                h_pull = ctx.pull_tensor(ki.data_out, dep=h_load)
                goldens.append(ki.forward())
                ctx.verify(h_pull, goldens[-1])
                ctx.config_boundary()

            result = ctx.run()
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
                h_load = ctx.load_tensor(ki.data_in)
                ctx.push_tensor(ki.data_in, dep=h_load)
                h_pull = ctx.pull_tensor(ki.data_out, dep=h_load)
                ctx.verify(h_pull, ki.forward())
                ctx.config_boundary()

            result = ctx.run()
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
                ki = ctx.instantiate(KernelClass, N=N)
                ki.generate_inputs(seed=42)

                h_load = ctx.load_tensor(ki.data_in)

                # Configure scale sub-kernel
                h_sf = ctx.write_register(
                    ki.scale_ctrl, {"scale_factor": cfg["scale_factor"]}, dep=h_load,
                )
                h_len_s = ctx.write_register(
                    ki.scale_ctrl, {"length": total_beats}, dep=h_sf,
                )

                # Configure offset sub-kernel
                off_val = cfg["offset_value"] & 0xFF  # 8-bit register
                h_ov = ctx.write_register(
                    ki.offset_ctrl, {"offset_value": off_val}, dep=h_load,
                )
                h_len_o = ctx.write_register(
                    ki.offset_ctrl, {"length": total_beats}, dep=h_ov,
                )

                h_push = ctx.push_tensor(ki.data_in, dep=[h_len_s, h_len_o])
                h_pull = ctx.pull_tensor(ki.data_out, dep=[h_len_s, h_len_o])

                h_start_s = ctx.write_register(
                    ki.scale_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
                )
                h_start_o = ctx.write_register(
                    ki.offset_ctrl, {"start": 1}, dep=[h_len_s, h_len_o],
                )

                h_poll_s = ctx.poll_register(ki.scale_ctrl, "done", dep=h_start_s)
                h_poll_o = ctx.poll_register(ki.offset_ctrl, "done", dep=h_start_o)
                h_pull.add_commit_dependency(h_poll_s)
                h_pull.add_commit_dependency(h_poll_o)

                ctx.verify(h_pull, ki.forward(
                    scale_factor=cfg["scale_factor"],
                    offset_value=cfg["offset_value"],
                ))
                ctx.config_boundary()

            result = ctx.run()
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
            session_open = False
            for i in range(3):
                ctx = ExecutionContext(backend=backend, project_params={"N": 1024})
                ctx._session_open = session_open
                ki = ctx.instantiate(KernelClass, spec=spec, N=1024)
                ki.generate_inputs(seed=42 + i)
                h_load = ctx.load_tensor(ki.data_in)
                ctx.push_tensor(ki.data_in, dep=h_load)
                h_pull = ctx.pull_tensor(ki.data_out, dep=h_load)
                ctx.verify(h_pull, ki.forward())
                result = ctx.run()
                session_open = ctx._session_open
                assert result.status == "DONE"

            # Close session
            assert session_open is True
            backend.close_session()
        finally:
            os.chdir(prev_cwd)
            try:
                backend.cleanup()
            except Exception:
                pass

    @pytest.mark.xsim
    def test_session_executor_2call(self):
        """E2.2: KernelExecutor context manager with 2 sequential calls."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            with KernelExecutor(
                KernelClass, backend=backend, spec=spec, params={"N": 1024},
            ) as npu:
                x1 = torch.randint(-128, 127, (1024,), dtype=torch.int8)
                y1 = npu(data_in=x1)["data_out"]
                assert torch.equal(y1, x1)

                x2 = torch.randint(-128, 127, (1024,), dtype=torch.int8)
                y2 = npu(data_in=x2)["data_out"]
                assert torch.equal(y2, x2)
        finally:
            os.chdir(prev_cwd)
            try:
                backend.cleanup()
            except Exception:
                pass

    @pytest.mark.xsim
    def test_session_different_data(self):
        """E2.3: Multi-batch with different random data each batch."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            with KernelExecutor(
                KernelClass, backend=backend, spec=spec, params={"N": 1024},
            ) as npu:
                for seed in [1, 2, 3, 4, 5]:
                    rng = torch.Generator().manual_seed(seed)
                    x = torch.randint(-128, 127, (1024,), dtype=torch.int8, generator=rng)
                    y = npu(data_in=x)["data_out"]
                    assert torch.equal(y, x), f"Mismatch at seed={seed}"
        finally:
            os.chdir(prev_cwd)
            try:
                backend.cleanup()
            except Exception:
                pass

    @pytest.mark.xsim
    def test_session_cross_batch_alias(self):
        """E2.4: Output from batch 1 passed as input to batch 2 (auto-alias)."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            with KernelExecutor(
                KernelClass, backend=backend, spec=spec, params={"N": 1024},
            ) as npu:
                x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
                y1 = npu(data_in=x)["data_out"]
                assert torch.equal(y1, x)

                # Pass output as input — triggers auto-alias (buffer reuse)
                y2 = npu(data_in=y1)["data_out"]
                assert torch.equal(y2, x)  # passthrough: output == input
        finally:
            os.chdir(prev_cwd)
            try:
                backend.cleanup()
            except Exception:
                pass

    @pytest.mark.xsim
    def test_session_executor_close(self):
        """E2.5: Open session, execute once, close cleanly."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            npu = KernelExecutor(
                KernelClass, backend=backend, spec=spec, params={"N": 1024},
            )
            x = torch.randint(-128, 127, (1024,), dtype=torch.int8)
            y = npu(data_in=x)["data_out"]
            assert torch.equal(y, x)
            assert npu._session_open is True
            npu.close()
            assert npu._session_open is False
        finally:
            os.chdir(prev_cwd)
            try:
                backend.cleanup()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# Category F: Edge Cases xsim E2E
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """F: Edge cases — broken passthrough, tensor size boundaries."""

    @pytest.mark.xsim
    def test_broken_passthrough_verify_fail(self):
        """F1: broken_passthrough detects host-side verification failure."""
        project, config, backend = _load_passthrough_env("broken_passthrough")
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
            h_load = ctx.load_tensor(ki.data_in)
            ctx.push_tensor(ki.data_in, dep=h_load)
            h_pull = ctx.pull_tensor(ki.data_out, dep=h_load)
            # forward() returns correct data, but RTL XORs with 0x01
            ctx.verify(h_pull, ki.forward())

            with pytest.raises(VerificationError):
                ctx.run()
        finally:
            os.chdir(prev_cwd)
            try:
                backend.close_session()
            except Exception:
                pass
            try:
                backend.cleanup()
            except Exception:
                pass

    @pytest.mark.xsim
    def test_broken_passthrough_probe_fail(self):
        """F2: broken_passthrough with probe=True detects BFM-level mismatch."""
        project, config, backend = _load_passthrough_env("broken_passthrough")
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
            h_load = ctx.load_tensor(ki.data_in)
            ctx.push_tensor(ki.data_in, dep=h_load)
            # probe=True triggers COMPARE command in BFM
            h_pull = ctx.pull_tensor(ki.data_out, dep=h_load, probe=True)
            ctx.verify(h_pull, ki.forward())

            with pytest.raises(Exception):
                # Either VerificationError from host or BackendError
                # from PROBE_MISMATCH — both indicate failure detection
                ctx.run()
        finally:
            os.chdir(prev_cwd)
            try:
                backend.close_session()
            except Exception:
                pass
            try:
                backend.cleanup()
            except Exception:
                pass

    @pytest.mark.xsim
    def test_small_tensor_n32(self):
        """F3: Minimum tensor size (N=32, exactly 1 beat on 256-bit bus)."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            x = torch.randint(-128, 127, (32,), dtype=torch.int8)
            outputs = run_kernel(
                KernelClass, {"data_in": x},
                backend=backend, spec=spec, params={"N": 32},
            )
            assert torch.equal(outputs["data_out"], x)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()

    @pytest.mark.xsim
    def test_large_tensor_n4096(self):
        """F4: Large tensor (N=4096, 128 beats on 256-bit bus)."""
        project, config, backend = _load_passthrough_env()
        spec = parse_kernel_spec(project / "kernels" / "passthrough" / "kernel_spec.yaml")
        KernelClass = _get_passthrough_kernel(project)

        prev_cwd = os.getcwd()
        os.chdir(str(project))
        try:
            x = torch.randint(-128, 127, (4096,), dtype=torch.int8)
            outputs = run_kernel(
                KernelClass, {"data_in": x},
                backend=backend, spec=spec, params={"N": 4096},
            )
            assert torch.equal(outputs["data_out"], x)
        finally:
            os.chdir(prev_cwd)
            backend.cleanup()
