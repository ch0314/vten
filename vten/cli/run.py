"""vten run: TestScenario base class and test discovery.

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
from vten.errors import BackendError, ProbeMismatchError, VTenError, VerificationError

logger = logging.getLogger(__name__)


class TestScenario:
    """Base class for user-defined test scenarios.

    For kernels that implement ``run(self, ctx)``, subclasses can omit the
    ``run`` method entirely.  The default implementation:

    1. Discovers the Kernel class from ``self.kernel`` name.
    2. Instantiates with *cfg* as runtime params.
    3. Calls ``generate_inputs(seed=cfg.get("seed", 42))``.
    4. Calls ``kernel.run(ctx)``.
    """

    kernel: str = ""
    configs: list[dict] | None = None
    probes: list[str] | None = None

    def run(self, ctx, cfg) -> None:
        """Default run: auto-discover kernel class, instantiate, run."""
        kernel_cls = self._discover_kernel_class()
        if kernel_cls is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must either override run() "
                f"or set 'kernel' to a valid kernel name."
            )
        k = ctx.instantiate(kernel_cls, **cfg)
        ki = k.kernel_class_instance

        # Register declarative probes before kernel execution
        if self.probes:
            ctx._register_declarative_probes(self.probes)

        ki.generate_inputs(seed=cfg.get("seed", 42))
        ki.run(ctx)

    def _discover_kernel_class(self) -> type | None:
        """Find Kernel subclass from self.kernel name.

        Searches ``kernels/{name}/{name}_kernel.py`` relative to the test
        file location, which is the standard NPU_3D layout.
        """
        if not self.kernel:
            return None

        from vten.kernel.base import Kernel

        # Locate kernel module: tests/ is inside kernels/{name}/tests/
        # so go up two levels to find kernels/{name}/{name}_kernel.py
        test_file = sys.modules.get(self.__class__.__module__)
        if test_file and hasattr(test_file, "__file__") and test_file.__file__:
            tests_dir = Path(test_file.__file__).resolve().parent
            kernel_dir = tests_dir.parent
            kernel_file = kernel_dir / f"{self.kernel}_kernel.py"
            if not kernel_file.exists():
                # Try parent's parent for composites (kernels/{name}/)
                kernels_base = kernel_dir.parent
                kernel_file = (
                    kernels_base / self.kernel / f"{self.kernel}_kernel.py"
                )

            if kernel_file.exists():
                mod_name = f"_vten_kernel_{self.kernel}"
                # Add kernel dir and kernels base to sys.path so that
                # both intra-kernel and sibling-kernel imports resolve
                # (e.g., CompositeKernel importing sub-kernel modules).
                parent = str(kernel_file.parent)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                kernels_base = str(kernel_file.parent.parent)
                if kernels_base not in sys.path:
                    sys.path.insert(0, kernels_base)

                spec = importlib.util.spec_from_file_location(
                    mod_name, kernel_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = module
                    spec.loader.exec_module(module)
                    # Prefer classes defined in this module over imports
                    candidates = []
                    for attr_name in dir(module):
                        obj = getattr(module, attr_name)
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, Kernel)
                            and obj is not Kernel
                        ):
                            candidates.append(obj)
                    # Filter to locally-defined classes first
                    local = [c for c in candidates
                             if c.__module__ == mod_name]
                    if local:
                        return local[0]
                    if candidates:
                        return candidates[0]
        return None


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
        except Exception as exc:
            logger.warning("failed to load %s: %s", test_file.name, exc)
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

    # Priority tiers: exact match wins over fuzzy match
    exact_matches: list[type] = []
    fuzzy_matches: list[type] = []

    for file_stem, cls in candidates:
        cls_name = cls.__name__
        # Tier 1: Exact class name match
        if cls_name == name:
            exact_matches.append(cls)
        # Tier 2: Case-insensitive class name
        elif cls_name.lower() == name_lower:
            fuzzy_matches.append(cls)
        # Tier 2: snake_case / filename stem match
        elif file_stem == name or file_stem == f"test_{name}":
            fuzzy_matches.append(cls)
        elif file_stem.removeprefix("test_") == name:
            fuzzy_matches.append(cls)

    # Use exact matches if available, otherwise fall back to fuzzy
    matches = exact_matches if exact_matches else fuzzy_matches

    if not matches:
        raise VTenError(f"Not found: no test scenario matching '{name}'")

    if len(matches) > 1:
        # Deduplicate by class identity
        unique = list({id(c): c for c in matches}.values())
        if len(unique) > 1:
            names = [c.__name__ for c in unique]
            raise VTenError(f"Ambiguous: multiple matches for '{name}': {names}")
        matches = unique

    return matches[0]()


def discover_all_tests(tests_dir: str | Path) -> list[tuple[str, TestScenario]]:
    """Discover all TestScenario subclasses in tests_dir.

    Returns a list of (class_name, instance) pairs, sorted by class name.
    """
    tests_path = Path(tests_dir)
    test_files = sorted(tests_path.glob("test_*.py"))

    seen: dict[int, tuple[str, type]] = {}

    for test_file in test_files:
        mod_name = f"_vten_discover_{test_file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, test_file)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("failed to load %s: %s", test_file.name, exc)
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, TestScenario)
                and obj is not TestScenario
                and id(obj) not in seen
            ):
                seen[id(obj)] = (obj.__name__, obj)

    return [(name, cls()) for name, cls in sorted(seen.values(), key=lambda x: x[0])]


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
    """Try to derive BFM configs from kernel_spec.yaml or synthesized spec."""
    from vten.runtime.ir import BFMConfig
    from vten.spec.parser import parse_kernel_spec

    spec_path = kernel_dir / "kernel_spec.yaml"
    spec = None

    if spec_path.exists():
        try:
            spec = parse_kernel_spec(spec_path)
        except Exception:
            pass
    else:
        # Composite kernel: synthesize spec
        from vten.build.composite import (
            is_composite_kernel,
            load_composite_class,
            synthesize_spec,
        )
        if is_composite_kernel(kernel_dir):
            try:
                project = kernel_dir.parent.parent
                composite_cls = load_composite_class(kernel_dir)
                spec = synthesize_spec(
                    composite_cls, project, kernel_dir.name
                )
            except Exception:
                pass

    if spec is None:
        return []

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


def _enrich_stats(
    stats: list,
    compiled: object | None,
) -> list[dict]:
    """Build enriched command stats dicts from CmdStats + CompiledResult."""
    from vten.reporting import build_command_metadata, merge_stats_with_metadata

    if compiled is not None and compiled.commands:
        metadata = build_command_metadata(compiled)
        enriched = merge_stats_with_metadata(stats, metadata)
        return [e.to_dict() for e in enriched]

    # Fallback: no CompiledResult available (pre-built SHM path)
    from vten.reporting import _status_name

    return [
        {
            "cmd_id": s.cmd_id,
            "status": s.status,
            "status_name": _status_name(s.status),
            "issue_cycle": s.issue_cycle,
            "commit_cycle": s.commit_cycle,
            "latency_cycles": s.latency_cycles,
            "active_cycles": s.active_cycles,
            "stall_cycles": s.stall_cycles,
            "total_beats": s.total_beats,
        }
        for s in stats
    ]


def _report_probe_mismatch(
    pme: ProbeMismatchError,
    results_dir: Path,
    ctx,
    cfg_idx: int,
    total_cfgs: int,
) -> None:
    """Report ProbeMismatchError with dtype-aware element info."""
    # Resolve tensor name and dtype from compiled context
    tensor_name = "unknown"
    dtype_str = ""
    packing = None
    compiled = getattr(ctx, "_last_compiled", None)
    if compiled and hasattr(compiled, "buffer_ids") and hasattr(compiled, "commands"):
        # Reverse map: cmd_id → buffer_id → tensor_name
        cmd_bid = None
        for cmd in compiled.commands:
            if cmd.cmd_id == pme.cmd_id:
                cmd_bid = cmd.buffer_id
                break
        if cmd_bid is not None:
            bid_to_name = {bid: name for name, bid in compiled.buffer_ids.items()}
            raw_name = bid_to_name.get(cmd_bid, "unknown")
            tensor_name = raw_name.split(":")[0] if ":" in raw_name else raw_name

        # Get dtype and packing from flattened view
        view = compiled.flattened_view
        if view:
            exposed = view.exposed_tensors.get(tensor_name)
            if exposed and exposed.origin_tensor:
                dtype_str = str(exposed.origin_tensor.dtype).replace("torch.", "")
            iface_name = exposed.top_interface if exposed else None
            if iface_name:
                iface = view.top_spec.get_interface(iface_name)
                packing = iface.packing if iface else None

    # Parse mismatches.jsonl for element-level detail
    mismatch_file = results_dir / "mismatches.jsonl"
    mismatches = []
    if mismatch_file.exists():
        try:
            for line in mismatch_file.read_text().strip().splitlines():
                mismatches.append(json.loads(line))
        except Exception:
            pass

    # Build readable message
    lines = [f"probe mismatch (config {cfg_idx + 1}/{total_cfgs})"]
    lines.append(f"  tensor: {tensor_name}" + (f" ({dtype_str})" if dtype_str else ""))
    lines.append(f"  cmd_id: {pme.cmd_id}")

    if mismatches and packing:
        m = mismatches[0]
        beat = m.get("beat", 0)
        # Compute element indices from beat index
        epb = packing.elements_per_beat
        elem_start = beat * epb
        elem_end = elem_start + epb - 1
        lines.append(f"  first mismatch: beat {beat} (elements [{elem_start}..{elem_end}])")

        # Show expected vs actual bytes interpreted as dtype elements
        try:
            exp_hi = int(m.get("expected_hi", "0"), 16)
            exp_lo = int(m.get("expected_lo", "0"), 16)
            act_hi = int(m.get("actual_hi", "0"), 16)
            act_lo = int(m.get("actual_lo", "0"), 16)
            exp_bytes = exp_hi.to_bytes(4, "big") + exp_lo.to_bytes(4, "big")
            act_bytes = act_hi.to_bytes(4, "big") + act_lo.to_bytes(4, "big")

            import struct
            import torch
            ew = packing.element_width
            dtype_torch = None
            if exposed and exposed.origin_tensor:
                dtype_torch = exposed.origin_tensor.dtype

            # Show first few differing elements
            elem_size = ew // 8
            if elem_size > 0:
                n_show = min(epb, len(exp_bytes) // elem_size, 8)
                exp_vals = _unpack_elements(exp_bytes, elem_size, n_show, dtype_torch)
                act_vals = _unpack_elements(act_bytes, elem_size, n_show, dtype_torch)
                diff_indices = [
                    i for i in range(n_show)
                    if exp_vals[i] != act_vals[i]
                ]
                if diff_indices:
                    for i in diff_indices[:4]:
                        lines.append(
                            f"    [{elem_start + i}]: expected={exp_vals[i]}, "
                            f"actual={act_vals[i]}"
                        )
                    if len(diff_indices) > 4:
                        lines.append(f"    ... and {len(diff_indices) - 4} more")
        except Exception:
            # Fall back to raw hex
            lines.append(
                f"    expected: 0x{m.get('expected_hi','')}{m.get('expected_lo','')}"
            )
            lines.append(
                f"    actual:   0x{m.get('actual_hi','')}{m.get('actual_lo','')}"
            )

        if len(mismatches) > 1:
            lines.append(f"  total mismatches logged: {len(mismatches)}")
    elif mismatches:
        m = mismatches[0]
        lines.append(f"  beat {m.get('beat', '?')}, cycle {m.get('cycle', '?')}")
        lines.append(
            f"    expected: 0x{m.get('expected_hi','')}{m.get('expected_lo','')}"
        )
        lines.append(
            f"    actual:   0x{m.get('actual_hi','')}{m.get('actual_lo','')}"
        )

    logger.error("\n".join(lines))


def _unpack_elements(
    raw: bytes, elem_size: int, count: int, dtype=None,
) -> list:
    """Unpack raw bytes into element values based on dtype."""
    import struct
    import torch

    values = []
    for i in range(count):
        chunk = raw[i * elem_size : (i + 1) * elem_size]
        if len(chunk) < elem_size:
            break
        if dtype == torch.float32 and elem_size == 4:
            values.append(round(struct.unpack("<f", chunk)[0], 6))
        elif dtype == torch.float16 and elem_size == 2:
            values.append(round(struct.unpack("<e", chunk)[0], 4))
        elif dtype == torch.int32 and elem_size == 4:
            values.append(struct.unpack("<i", chunk)[0])
        elif dtype == torch.int16 and elem_size == 2:
            values.append(struct.unpack("<h", chunk)[0])
        elif elem_size == 1:
            values.append(chunk[0])
        elif elem_size == 2:
            values.append(int.from_bytes(chunk, "little"))
        elif elem_size == 4:
            values.append(int.from_bytes(chunk, "little"))
        else:
            values.append(f"0x{chunk.hex()}")
    return values


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
    config["_mismatch_dir"] = str(results_dir)

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
    session_open = False  # Track session state across configs
    batch_count = 0      # Track batch number across configs
    try:
        for cfg_idx, cfg in enumerate(run_cfgs):
            logger.debug("config %d/%d: %s", cfg_idx + 1, len(run_cfgs), cfg)
            try:
                # Create ExecutionContext with backend so ctx.run() drives
                # the full lifecycle: compile → execute → verify
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
                # Share session state across configs for multi-batch mode
                ctx._session_open = session_open
                ctx._batch_count = batch_count
                scenario.run(ctx, cfg)

                if ctx._pending_ops:
                    # Scenario recorded DSL ops — ctx.run() handles everything
                    # including deferred verifications
                    batch_result = ctx.run()
                    session_open = ctx._session_open
                    batch_count = ctx._batch_count
                    configs_passed += 1

                    if batch_result.per_command_stats:
                        max_cycle = max(
                            (s.commit_cycle for s in batch_result.per_command_stats
                             if s.commit_cycle),
                            default=0,
                        )
                        total_cycles = max(total_cycles, max_cycle)
                        all_cmd_stats.extend(
                            _enrich_stats(
                                batch_result.per_command_stats,
                                ctx._last_compiled,
                            )
                        )

                    logger.info("  config %d/%d: PASS (%d cycles, %d verifications)",
                               cfg_idx + 1, len(run_cfgs),
                               batch_result.total_cycles,
                               batch_result.verification_count)

                    # Count verifications that passed (no VerificationError raised)
                    verification_count += batch_result.verification_count
                    verification_passed += batch_result.verification_count
                    for vr in batch_result.verification_results:
                        all_verification_results.append({
                            "tensor": vr.tensor_name,
                            "passed": vr.passed,
                            "max_diff": vr.max_diff,
                        })
                else:
                    # No DSL ops — fall back to pre-built SHM image
                    from vten.runtime.engine import CompiledResult
                    shm_image = _build_shm_image(kernel_dir)
                    bfm_configs = _build_bfm_configs(kernel_dir)
                    compiled = CompiledResult(
                        commands=[],
                        shm_image=shm_image or b"",
                        bfm_configs=bfm_configs,
                        buffer_ids={},
                        flattened_view=None,
                    )
                    result = backend_inst.execute(compiled)
                    configs_passed += 1
                    if result.stats:
                        max_cycle = max(
                            (s.commit_cycle for s in result.stats
                             if s.commit_cycle),
                            default=0,
                        )
                        total_cycles = max(total_cycles, max_cycle)
                        all_cmd_stats.extend(
                            _enrich_stats(result.stats, None)
                        )
            except VerificationError as ve:
                status = "FAIL"
                last_error = ve
                logger.warning("verification failed (config %d/%d): %s",
                               cfg_idx + 1, len(run_cfgs), ve)
                # Collect verification results from the error context
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
                _report_probe_mismatch(pme, results_dir, ctx, cfg_idx, len(run_cfgs))
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
    finally:
        # Close session if one was opened (multi-batch mode)
        if session_open:
            try:
                backend_inst.close_session()
            except Exception:
                pass
        else:
            try:
                backend_inst.shutdown()
            except Exception:
                pass
        # Always cleanup (releases XRT resources, removes hw_emu .run/<PID>)
        try:
            backend_inst.cleanup()
        except Exception:
            pass

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
    if config.get("_waveform"):
        build_dir = Path(config.get("_kernel_build_dir", "."))
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
        if config.get("_waveform_on_fail") and status == "PASS":
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

    # Inject kernel-level paths into config
    config["_project_dir"] = str(project)
    config["_kernel_build_dir"] = str(kernel_dir / "build")
    if waveform or waveform_on_fail:
        config["_waveform"] = True
    if waveform_on_fail:
        config["_waveform_on_fail"] = True
    if gui:
        config["_gui"] = True
    if sim_verbose:
        config["_sim_verbose"] = True

    backend_name = resolve_backend_name(config, cli_backend=backend)
    backend_inst = get_backend(backend_name, config)

    # Change CWD so relative paths resolve correctly.
    # For XRT backend, use kernel build/xrt/ to contain runtime artifacts
    # (emconfig.json, device_trace, run_summary, etc.) that XRT dumps to CWD.
    # For sim backends, use project root for kernel_spec relative paths.
    import os
    prev_cwd = os.getcwd()
    if backend_name == "xrt":
        xrt_cwd = kernel_dir / "build" / "xrt"
        xrt_cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(str(xrt_cwd))
    else:
        os.chdir(str(project))
    try:
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
            )
    finally:
        os.chdir(prev_cwd)
