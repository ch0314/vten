"""vten run: test execution orchestration.

Spec reference: 00_data_models.md §14, 06_codegen_and_cli.md §4.4
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

from vten.backend.registry import get_backend, resolve_backend_name
from vten.cli.config import load_project_config
from vten.cli.discovery import discover_all_tests, discover_test
from vten.cli.probe_report import enrich_stats, report_probe_mismatch
from vten.cli.scenario import TestScenario
from vten.errors import BackendError, ProbeMismatchError, VTenError, VerificationError


logger = logging.getLogger(__name__)


def merge_configs(base: dict, override: dict | None) -> dict:
    """Merge base config with override. Override wins on conflicts."""
    if not override:
        return dict(base)
    return {**base, **override}


def _run_single_test(
    project: Path,
    config: dict,
    kernel_name: str,
    test_name: str,
    scenario: TestScenario,
    kernel_dir: Path,
    backend_name: str,
    backend_inst,
    config_overrides: dict | None = None,
    verify: bool = False,
    gui: bool = False,
) -> None:
    """Execute a single test scenario and write results."""
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

    # Set mismatch dir for probe logging (C bridge reads via env var)
    backend_inst._run_ctx.mismatch_dir = results_dir

    configs_passed = 0
    total_cycles = 0
    all_cmd_stats: list[dict] = []
    verification_count = 0
    verification_passed = 0
    all_verification_results: list[dict] = []
    status = "PASS"

    logger.info("run %s/%s (backend=%s, configs=%d)",
                kernel_name, test_name, backend_name, len(run_cfgs))

    last_error: Exception | None = None
    for cfg_idx, cfg in enumerate(run_cfgs):
        logger.debug("config %d/%d: %s", cfg_idx + 1, len(run_cfgs), cfg)
        try:
            from vten.runtime.context import ExecutionContext

            # Include build_params so resolver Tier 2 can access them
            if "build_params" not in cfg and "build_params" in config:
                cfg["build_params"] = config["build_params"]

            # Propagate project-level paths into per-config params
            for _pk in ("_project_dir", "_kernel_build_dir"):
                if _pk in config and _pk not in cfg:
                    cfg[_pk] = config[_pk]

            ctx = ExecutionContext(
                backend=backend_inst,
                project_params=cfg,
            )
            scenario.run(ctx, cfg)

            if not ctx._pending_ops:
                # Scenario recorded no ops — count as pass, skip execution
                configs_passed += 1
                continue

            batch_result = ctx.run(verify=verify)
            configs_passed += 1

            if batch_result.per_command_stats:
                max_cycle = max(
                    (s.commit_cycle for s in batch_result.per_command_stats
                     if s.commit_cycle),
                    default=0,
                )
                total_cycles = max(total_cycles, max_cycle)
                all_cmd_stats.extend(
                    enrich_stats(
                        batch_result.per_command_stats,
                        ctx._last_compiled,
                    )
                )

            logger.info("  config %d/%d: PASS (%d cycles, %d verifications)",
                       cfg_idx + 1, len(run_cfgs),
                       batch_result.total_cycles,
                       batch_result.verification_count)

            verification_count += batch_result.verification_count
            verification_passed += batch_result.verification_count
            for vr in batch_result.verification_results:
                all_verification_results.append({
                    "tensor": vr.tensor_name,
                    "passed": vr.passed,
                    "max_diff": vr.max_diff,
                })
        except VerificationError as ve:
            status = "FAIL"
            last_error = ve
            logger.warning("verification failed (config %d/%d): %s",
                           cfg_idx + 1, len(run_cfgs), ve)
            vr_list = ve.context.get("verification_results", [])
            verification_count += len(vr_list) if vr_list else 1
            for vr in vr_list:
                all_verification_results.append({
                    "tensor": vr.tensor_name,
                    "passed": vr.passed,
                    "max_diff": vr.max_diff,
                })
                if vr.passed:
                    verification_passed += 1
            if not vr_list:
                all_verification_results.append({
                    "tensor": ve.tensor,
                    "passed": False,
                    "max_diff": ve.max_diff,
                })
        except ProbeMismatchError as pme:
            status = "FAIL"
            last_error = pme
            report_probe_mismatch(pme, results_dir, ctx, cfg_idx, len(run_cfgs))
        except BackendError as be:
            status = "FAIL"
            last_error = be
            logger.error("backend error (config %d/%d): %s",
                         cfg_idx + 1, len(run_cfgs), be)
        except Exception as exc:
            status = "FAIL"
            last_error = exc
            logger.error("test execution failed (config %d/%d): %s",
                         cfg_idx + 1, len(run_cfgs), exc)
            logger.debug("traceback:", exc_info=True)

    if configs_passed < len(run_cfgs):
        status = "FAIL"

    logger.info("result: %s (%d/%d configs passed)", status, configs_passed, len(run_cfgs))

    summary: dict = {
        "test_name": test_name,
        "kernel": kernel_name,
        "status": status,
        "total_cycles": total_cycles,
        "configs_run": len(run_cfgs),
        "configs_passed": configs_passed,
        "verification_count": verification_count,
        "verification_passed": verification_passed,
    }
    if all_verification_results:
        summary["verification_results"] = all_verification_results
    if last_error is not None:
        summary["error_message"] = str(last_error)
        summary["error_traceback"] = traceback.format_exception(last_error)
        if isinstance(last_error, ProbeMismatchError):
            summary["probe_mismatch"] = {
                "cmd_id": last_error.cmd_id,
                "beat_index": last_error.beat_index,
                "mismatches": last_error.mismatches[:10],  # Cap at 10 entries
            }

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    (results_dir / "stats.json").write_text(json.dumps(
        {"commands": all_cmd_stats}, indent=2,
    ))

    # Waveform file management
    rctx = backend_inst._run_ctx
    if rctx.waveform:
        build_dir = rctx.kernel_build_dir or Path(".")
        # Look for .wdb files in common locations
        for wdb_glob in [build_dir / "*.wdb",
                         build_dir / "generated" / "*.wdb",
                         build_dir / "xsim.dir" / "*.wdb"]:
            import glob as glob_mod
            for wdb_path in glob_mod.glob(str(wdb_glob)):
                wdb_src = Path(wdb_path)
                wdb_dst = results_dir / "waveform.wdb"
                import shutil
                shutil.copy2(wdb_src, wdb_dst)
                logger.info("waveform saved: %s", wdb_dst)
                break

        # waveform_on_fail: delete wdb on PASS
        if rctx.waveform_on_fail and status == "PASS":
            wdb_file = results_dir / "waveform.wdb"
            if wdb_file.exists():
                wdb_file.unlink()
                logger.debug("waveform deleted (test passed, --waveform-on-fail)")


def run_test(
    project_dir: str = ".",
    kernel_name: str = "",
    test_name: str = "",
    backend: str | None = None,
    waveform: bool = False,
    waveform_on_fail: bool = False,
    gui: bool = False,
    sim_verbose: bool = False,
    config_overrides: dict | None = None,
    verify: bool = False,
) -> None:
    """Discover, execute, and record results for test scenario(s).

    When test_name is empty, discovers and runs all TestScenario subclasses
    in the kernel's tests/ directory.
    """
    project = Path(project_dir).resolve()
    config = load_project_config(project)
    kernel_dir = project / "kernels" / kernel_name

    # Validate kernel directory
    from vten.build.composite import is_composite_kernel
    spec_path = kernel_dir / "kernel_spec.yaml"
    if not spec_path.exists() and not is_composite_kernel(kernel_dir):
        raise VTenError(f"kernel_spec.yaml not found: {spec_path}")

    # Add kernels base to sys.path so test files can import shared modules
    # (e.g., model_configs.py, _common.py) from the kernels/ directory.
    kernels_base = str(kernel_dir.parent)
    if kernels_base not in sys.path:
        sys.path.insert(0, kernels_base)

    # Test discovery
    tests_dir = kernel_dir / "tests"
    if test_name:
        scenarios = [(test_name, discover_test(test_name, tests_dir))]
    else:
        scenarios = discover_all_tests(tests_dir)
        if not scenarios:
            raise VTenError(f"no test scenarios found in {tests_dir}")
        logger.info("discovered %d test(s): %s",
                     len(scenarios), [n for n, _ in scenarios])

    # Build RunContext for backend (typed runtime state)
    from vten.backend.base import RunContext

    run_ctx = RunContext(
        project_dir=project,
        kernel_build_dir=kernel_dir / "build",
        waveform=waveform or waveform_on_fail,
        waveform_on_fail=waveform_on_fail,
        gui=gui,
        sim_verbose=sim_verbose,
    )

    # Keep legacy _ keys in config for backward compat (runtime pipeline
    # reads _project_dir from project_params, not from RunContext yet).
    config["_project_dir"] = str(project)
    config["_kernel_build_dir"] = str(kernel_dir / "build")

    backend_name = resolve_backend_name(config, cli_backend=backend)
    backend_inst = get_backend(backend_name, config)
    backend_inst.set_run_context(run_ctx)

    import os
    prev_cwd = os.getcwd()
    run_cwd = backend_inst.working_directory(kernel_dir, project)
    os.chdir(str(run_cwd))
    try:
        with backend_inst:
            for test_idx, (scenario_name, scenario) in enumerate(scenarios, 1):
                logger.info("")
                logger.info("════ test %d/%d: %s/%s ════",
                            test_idx, len(scenarios), kernel_name, scenario_name)
                _run_single_test(
                    project=project,
                    config=config,
                    kernel_name=kernel_name,
                    test_name=scenario_name,
                    scenario=scenario,
                    kernel_dir=kernel_dir,
                    backend_name=backend_name,
                    backend_inst=backend_inst,
                    config_overrides=config_overrides,
                    gui=gui,
                    verify=verify,
                )
    finally:
        os.chdir(prev_cwd)
