"""vten run: test execution orchestration.

Spec reference: 00_data_models.md §14, 06_codegen_and_cli.md §4.4
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import traceback
from pathlib import Path

from vten.backend.registry import get_backend, resolve_backend_name
from vten.cli.config import load_project_config
from vten.cli.discovery import discover_all_tests, discover_test
from vten.cli.probe_report import enrich_stats
from vten.cli.scenario import TestScenario
from vten.errors import ProbeMismatchError, VTenError, VerificationError
from vten.execution import execute_batch


logger = logging.getLogger(__name__)


def discover_kernel_class(kernel_name: str, kernel_dir: Path) -> type:
    """Find Kernel subclass from kernel directory.

    Searches ``kernels/{name}/{name}_kernel.py`` for a Kernel subclass.
    """
    from vten.kernel.base import Kernel

    kernel_file = kernel_dir / f"{kernel_name}_kernel.py"
    if not kernel_file.exists():
        raise VTenError(
            f"kernel file not found: {kernel_file}"
        )

    mod_name = f"_vten_kernel_{kernel_name}"
    parent = str(kernel_file.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    kernels_base = str(kernel_file.parent.parent)
    if kernels_base not in sys.path:
        sys.path.insert(0, kernels_base)

    spec = importlib.util.spec_from_file_location(mod_name, kernel_file)
    if spec is None or spec.loader is None:
        raise VTenError(f"cannot load kernel module: {kernel_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    candidates = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, Kernel) and obj is not Kernel:
            candidates.append(obj)

    local = [c for c in candidates if c.__module__ == mod_name]
    if local:
        return local[0]
    if candidates:
        return candidates[0]
    raise VTenError(f"no Kernel subclass found in {kernel_file}")


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

    # Include build_params so resolver Tier 2 can access them
    for cfg in run_cfgs:
        if "build_params" not in cfg and "build_params" in config:
            cfg["build_params"] = config["build_params"]

    # Results under results/<kernel>/<test>/
    results_dir = project / "results" / kernel_name / test_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set mismatch dir for probe logging (C bridge reads via env var)
    backend_inst._run_ctx.mismatch_dir = results_dir

    logger.info("run %s/%s (backend=%s, configs=%d)",
                kernel_name, test_name, backend_name, len(run_cfgs))

    # Discover kernel class from scenario
    kernel_cls = scenario._discover_kernel_class()
    if kernel_cls is None:
        raise VTenError(
            f"cannot discover kernel class for '{scenario.kernel}'"
        )

    # Execute all configs via shared core
    batch = execute_batch(
        backend=backend_inst,
        kernel_class=kernel_cls,
        configs=run_cfgs,
        verify=verify,
        project_dir=project,
        probes=scenario.probes,
        seed=scenario.seed,
        on_error="continue",
    )

    # ── Aggregate results for summary ──
    _write_results(batch, run_cfgs, kernel_name, test_name, results_dir,
                   backend_inst)


def _run_adhoc(
    project: Path,
    config: dict,
    kernel_name: str,
    kernel_dir: Path,
    backend_name: str,
    backend_inst,
    configs: list[dict],
    gui: bool = False,
    verify: bool = False,
) -> None:
    """Execute ad-hoc configs without TestScenario.

    Used when ``--config`` provides a list of config dicts directly.
    Discovers the kernel class from the kernel directory and runs
    all configs via ``execute_batch``.
    """
    # Merge build_params into each config
    for cfg in configs:
        if "build_params" not in cfg and "build_params" in config:
            cfg["build_params"] = config["build_params"]

    test_name = "adhoc"
    results_dir = project / "results" / kernel_name / test_name
    results_dir.mkdir(parents=True, exist_ok=True)

    backend_inst._run_ctx.mismatch_dir = results_dir

    logger.info("run %s (ad-hoc, backend=%s, configs=%d)",
                kernel_name, backend_name, len(configs))

    kernel_cls = discover_kernel_class(kernel_name, kernel_dir)

    import os
    prev_cwd = os.getcwd()
    run_cwd = backend_inst.working_directory(kernel_dir, project)
    os.chdir(str(run_cwd))
    try:
        with backend_inst:
            batch = execute_batch(
                backend=backend_inst,
                kernel_class=kernel_cls,
                configs=configs,
                verify=verify,
                project_dir=project,
                on_error="continue",
            )
    finally:
        os.chdir(prev_cwd)

    _write_results(batch, configs, kernel_name, test_name, results_dir,
                   backend_inst)


def _write_results(
    batch,
    run_cfgs: list[dict],
    kernel_name: str,
    test_name: str,
    results_dir: Path,
    backend_inst,
) -> None:
    """Process BatchResult into summary.json, stats.json, waveforms."""
    from vten.cli.probe_report import report_probe_mismatch

    total_cycles = 0
    all_cmd_stats: list[dict] = []
    verification_count = 0
    verification_passed = 0
    all_verification_results: list[dict] = []
    last_error: Exception | None = None

    for cr in batch.configs:
        cfg_idx = cr.config_index
        if cr.passed and cr.result is not None:
            r = cr.result
            logger.info("  config %d/%d: PASS (%d cycles, %d verifications)",
                        cfg_idx + 1, len(run_cfgs),
                        r.total_cycles, r.verification_count)

            total_cycles = max(total_cycles, r.total_cycles)
            if r.per_command_stats:
                max_cycle = max(
                    (s.commit_cycle for s in r.per_command_stats
                     if s.commit_cycle),
                    default=0,
                )
                total_cycles = max(total_cycles, max_cycle)
                # enrich_stats needs compiled context — skip if unavailable
                # (stats enrichment is secondary to correctness)

            verification_count += r.verification_count
            verification_passed += r.verification_count
            for vr in r.verification_results:
                all_verification_results.append({
                    "tensor": vr.tensor_name,
                    "passed": vr.passed,
                    "max_diff": vr.max_diff,
                })
        elif cr.error is not None:
            last_error = cr.error
            if isinstance(cr.error, VerificationError):
                ve = cr.error
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
            elif isinstance(cr.error, ProbeMismatchError):
                report_probe_mismatch(
                    cr.error, results_dir, None, cfg_idx, len(run_cfgs),
                )
            else:
                logger.error("config %d/%d failed: %s",
                             cfg_idx + 1, len(run_cfgs), cr.error)

    status = "PASS" if batch.all_passed else "FAIL"
    logger.info("result: %s (%d/%d configs passed)",
                status, batch.passed_count, len(run_cfgs))

    summary: dict = {
        "test_name": test_name,
        "kernel": kernel_name,
        "status": status,
        "total_cycles": total_cycles,
        "configs_run": len(run_cfgs),
        "configs_passed": batch.passed_count,
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
                "mismatches": last_error.mismatches[:10],
            }

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    (results_dir / "stats.json").write_text(json.dumps(
        {"commands": all_cmd_stats}, indent=2,
    ))

    # Waveform file management
    rctx = backend_inst._run_ctx
    if rctx.waveform:
        build_dir = rctx.kernel_build_dir or Path(".")
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
    config_overrides: dict | list[dict] | None = None,
    verify: bool = False,
) -> None:
    """Discover, execute, and record results for test scenario(s).

    When config_overrides is a list[dict], runs directly without
    TestScenario discovery (ad-hoc --config mode).

    When test_name is empty and config_overrides is not a list,
    discovers and runs all TestScenario subclasses in the kernel's
    tests/ directory.
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

    backend_name = resolve_backend_name(config, cli_backend=backend)
    backend_inst = get_backend(backend_name, config)
    backend_inst.set_run_context(run_ctx)

    # Ad-hoc mode: --config provides list[dict] → bypass TestScenario
    if isinstance(config_overrides, list):
        _run_adhoc(
            project=project,
            config=config,
            kernel_name=kernel_name,
            kernel_dir=kernel_dir,
            backend_name=backend_name,
            backend_inst=backend_inst,
            configs=config_overrides,
            gui=gui,
            verify=verify,
        )
        return

    # Scenario mode: discover TestScenario(s)
    tests_dir = kernel_dir / "tests"
    if test_name:
        scenarios = [(test_name, discover_test(test_name, tests_dir))]
    else:
        scenarios = discover_all_tests(tests_dir)
        if not scenarios:
            raise VTenError(f"no test scenarios found in {tests_dir}")
        logger.info("discovered %d test(s): %s",
                     len(scenarios), [n for n, _ in scenarios])

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
