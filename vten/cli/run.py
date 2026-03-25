"""vten run: TestScenario base class and test discovery.

Spec reference: 00_data_models.md §14, 06_codegen_and_cli.md §4.4
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from vten.backend.xsim import XsimBackend
from vten.cli.config import load_project_config
from vten.errors import VTenError, VerificationError


class TestScenario:
    """Base class for user-defined test scenarios."""

    kernel: str = ""
    configs: list[dict] | None = None

    def run(self, ctx, cfg) -> None:
        raise NotImplementedError


def discover_test(name: str, tests_dir: str | Path) -> TestScenario:
    """Find and instantiate a TestScenario by name.

    Matches by: exact class name, case-insensitive class name,
    snake_case name, or filename stem.
    """
    tests_path = Path(tests_dir)
    test_files = sorted(tests_path.glob("test_*.py"))

    candidates: list[tuple[str, type]] = []

    for test_file in test_files:
        mod_name = f"_vten_discover_{test_file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, test_file)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, TestScenario)
                and obj is not TestScenario
            ):
                candidates.append((test_file.stem, obj))

    name_lower = name.lower()
    matches: list[type] = []

    for file_stem, cls in candidates:
        cls_name = cls.__name__
        # Match by exact class name
        if cls_name == name:
            matches.append(cls)
        # Match by case-insensitive class name
        elif cls_name.lower() == name_lower:
            matches.append(cls)
        # Match by snake_case (e.g., "test_conv3d" matches TestConv3D via file stem)
        elif file_stem == name or file_stem == f"test_{name}":
            matches.append(cls)
        # Match by filename stem without test_ prefix
        elif file_stem.removeprefix("test_") == name:
            matches.append(cls)

    if not matches:
        raise ValueError(f"Not found: no test scenario matching '{name}'")

    if len(matches) > 1:
        # Deduplicate by class identity
        unique = list({id(c): c for c in matches}.values())
        if len(unique) > 1:
            names = [c.__name__ for c in unique]
            raise ValueError(f"Ambiguous: multiple matches for '{name}': {names}")
        matches = unique

    return matches[0]()


def merge_configs(base: dict, override: dict | None) -> dict:
    """Merge base config with override. Override wins on conflicts."""
    if not override:
        return dict(base)
    return {**base, **override}


def _build_shm_image(kernel_dir: Path) -> bytes | None:
    """Try to load pre-built SHM image from kernel build/shm/."""
    shm_path = kernel_dir / "build" / "shm" / "kernel_task.bin"
    if shm_path.exists():
        return shm_path.read_bytes()
    return None


def _build_bfm_configs(kernel_dir: Path) -> list:
    """Try to derive BFM configs from kernel_spec.yaml."""
    from vten.runtime.ir import BFMConfig
    from vten.spec.parser import parse_kernel_spec

    spec_path = kernel_dir / "kernel_spec.yaml"
    if not spec_path.exists():
        return []

    try:
        spec = parse_kernel_spec(spec_path)
        configs = []
        for name, iface in spec.interfaces.items():
            configs.append(BFMConfig(
                interface_name=name,
                protocol=iface.protocol,
                data_width=iface.data_width or 256,
                addr_width=iface.addr_width or 64,
                role="slave",
            ))
        return configs
    except Exception:
        return []


def _compile_from_context(ctx) -> tuple[bytes | None, list]:
    """If ctx has pending ops, compile them into SHM image and BFM configs.

    Returns (shm_image, bfm_configs) or (None, []) if no ops recorded.
    """
    if not ctx._pending_ops:
        return None, []

    from vten.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(
        kernels=ctx._kernels,
        ops=ctx._pending_ops,
        project_params=ctx._project_params,
        alias_registry=ctx._alias_registry,
    )
    compiled = engine.compile()
    ctx._last_compiled = compiled
    ctx._pending_ops = []
    return compiled.shm_image, compiled.bfm_configs


def run_test(
    project_dir: str = ".",
    kernel_name: str = "",
    test_name: str = "",
    waveform: bool = False,
    gui: bool = False,
    config_overrides: dict | None = None,
) -> None:
    """Discover, execute, and record results for a test scenario."""
    project = Path(project_dir).resolve()
    config = load_project_config(project)
    kernel_dir = project / "kernels" / kernel_name

    # Validate kernel directory
    spec_path = kernel_dir / "kernel_spec.yaml"
    if not spec_path.exists():
        raise VTenError(f"kernel_spec.yaml not found: {spec_path}")

    # Test discovery from kernel tests dir
    tests_dir = kernel_dir / "tests"
    scenario = discover_test(test_name, tests_dir)

    base_params = config.get("parameters", {})
    if config_overrides:
        base_params = merge_configs(base_params, config_overrides)

    if scenario.configs is not None:
        run_cfgs = [merge_configs(base_params, c) for c in scenario.configs]
    else:
        run_cfgs = [base_params]

    # Results under results/<kernel>/<test>/
    results_dir = project / "results" / kernel_name / test_name
    results_dir.mkdir(parents=True, exist_ok=True)

    configs_passed = 0
    total_cycles = 0
    all_cmd_stats: list[dict] = []
    verification_count = 0
    verification_passed = 0
    status = "PASS"

    # Inject kernel-level paths into config
    config["_project_dir"] = str(project)
    config["_kernel_build_dir"] = str(kernel_dir / "build")
    if gui:
        config["_gui"] = True

    backend = XsimBackend(config)
    try:
        for cfg in run_cfgs:
            try:
                # Create ExecutionContext with backend so ctx.run() drives
                # the full lifecycle: compile → submit → wait → verify
                from vten.runtime.context import ExecutionContext

                ctx = ExecutionContext(
                    backend=backend,
                    project_params=cfg,
                )
                scenario.run(ctx, cfg)

                if ctx._pending_ops:
                    # Scenario recorded DSL ops — ctx.run() handles everything
                    # including deferred verifications
                    batch_result = ctx.run()
                    configs_passed += 1

                    if batch_result.per_command_stats:
                        max_cycle = max(
                            (s.commit_cycle for s in batch_result.per_command_stats
                             if s.commit_cycle),
                            default=0,
                        )
                        total_cycles = max(total_cycles, max_cycle)
                        for s in batch_result.per_command_stats:
                            all_cmd_stats.append({
                                "cmd_id": s.cmd_id,
                                "status": s.status,
                                "issue_cycle": s.issue_cycle,
                                "commit_cycle": s.commit_cycle,
                                "active_cycles": s.active_cycles,
                                "stall_cycles": s.stall_cycles,
                                "total_beats": s.total_beats,
                                "latency_cycles": s.latency_cycles,
                            })

                    # Count verifications that passed (no VerificationError raised)
                    verification_count += batch_result.verification_count
                    verification_passed += batch_result.verification_count
                else:
                    # No DSL ops — fall back to pre-built SHM image
                    shm_image = _build_shm_image(kernel_dir)
                    bfm_configs = _build_bfm_configs(kernel_dir)
                    backend.submit(shm_image, bfm_configs)
                    result = backend.wait()
                    configs_passed += 1
                    if result.stats:
                        max_cycle = max(
                            (s.commit_cycle for s in result.stats
                             if s.commit_cycle),
                            default=0,
                        )
                        total_cycles = max(total_cycles, max_cycle)
                        for s in result.stats:
                            all_cmd_stats.append({
                                "cmd_id": s.cmd_id,
                                "status": s.status,
                                "issue_cycle": s.issue_cycle,
                                "commit_cycle": s.commit_cycle,
                                "active_cycles": s.active_cycles,
                                "stall_cycles": s.stall_cycles,
                                "total_beats": s.total_beats,
                                "latency_cycles": s.latency_cycles,
                            })
            except VerificationError as ve:
                status = "FAIL"
                verification_count += 1
                # verification_passed not incremented
            except Exception:
                status = "FAIL"

        if configs_passed < len(run_cfgs):
            status = "FAIL"
    finally:
        try:
            backend.shutdown()
        except Exception:
            pass
        try:
            backend.cleanup()
        except Exception:
            pass

    (results_dir / "summary.json").write_text(json.dumps({
        "test_name": test_name,
        "kernel": kernel_name,
        "status": status,
        "total_cycles": total_cycles,
        "configs_run": len(run_cfgs),
        "configs_passed": configs_passed,
        "verification_count": verification_count,
        "verification_passed": verification_passed,
    }))

    (results_dir / "stats.json").write_text(json.dumps({
        "commands": all_cmd_stats,
    }))
