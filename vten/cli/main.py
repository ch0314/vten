"""vten CLI entry point.

Spec reference: 06_codegen_and_cli.md §4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vten", description="vTen verification framework")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output (DEBUG level)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress info messages (WARNING level)")
    parser.add_argument("--log-file", default=None,
                        help="Write debug log to file")
    sub = parser.add_subparsers(dest="command")

    from vten.backend.registry import available_backends
    _backends = available_backends()

    # vten init
    init_parser = sub.add_parser("init", help="Create project skeleton")
    init_parser.add_argument("project_dir", help="Project directory to create")
    init_parser.add_argument("--kernel", help="Add kernel directory to existing project")
    init_parser.add_argument("--backend", default=None,
        choices=_backends,
        help="Target backend for new project (default: xsim)")
    init_parser.add_argument("--add-backend", default=None,
        choices=_backends,
        help="Add backend section to existing project")

    # vten build
    build_parser = sub.add_parser("build", help="Build project")
    build_parser.add_argument("--project", default=".", help="Project directory")
    build_parser.add_argument("--kernel", help="Build specific kernel only")
    build_parser.add_argument("--backend", default=None,
        choices=_backends,
        help="Target backend (default: from vten.toml or xsim)")
    _stages = ["project_setup", "dpi_c", "codegen", "compile_order", "compile"]
    build_parser.add_argument("--stage", choices=_stages,
                              help="Run specific stage only")
    build_parser.add_argument("--upto", choices=_stages,
                              help="Run stages up to (inclusive)")
    build_parser.add_argument("--target", default=None,
                              choices=["hw", "hw_emu"],
                              help="XRT build target (overrides vten.toml)")
    build_parser.add_argument("--force", action="store_true", help="Ignore cache, full rebuild")
    build_parser.add_argument("--clean", action="store_true",
                              help="Remove build artifacts before building")
    build_parser.add_argument("--skip-compile", action="store_true", help="Run codegen only")
    build_parser.add_argument("-v", "--verbose", action="store_true",
                              dest="build_verbose", help="Verbose output (DEBUG level)")
    build_parser.add_argument("--config", nargs="*", metavar="K=V",
                              help="Config overrides (e.g. --config in_ch=64 out_ch=32)")

    # vten run
    run_parser = sub.add_parser("run", help="Run test")
    run_parser.add_argument("--kernel", required=True, help="Kernel name")
    run_parser.add_argument("--test", default=None, help="Test scenario name (omit to run all)")
    run_parser.add_argument("--project", default=".", help="Project directory")
    run_parser.add_argument("--backend", default=None,
        choices=_backends,
        help="Target backend (default: from vten.toml or xsim)")
    run_parser.add_argument("--waveform", action="store_true", help="Enable waveform dump")
    run_parser.add_argument("--waveform-on-fail", action="store_true",
                            help="Dump waveform only on test failure")
    run_parser.add_argument("--gui", action="store_true", help="xsim GUI mode")
    run_parser.add_argument("-v", "--sim-verbose", action="store_true",
                            help="Enable simulator verbose output (+VTEN_VERBOSE)")
    run_parser.add_argument("--verify", action="store_true",
                            help="Auto-verify outputs against behavioral model golden")
    run_parser.add_argument("--config", nargs="*", metavar="SPEC",
                            help="Config overrides: K=V pairs, JSON '{\"k\":v}', "
                                 "or module:VAR[idx] (e.g. model_configs:UNET_MINI[0])")

    # vten list
    list_parser = sub.add_parser("list", help="List tests or params for a kernel")
    list_sub = list_parser.add_subparsers(dest="list_command")

    list_tests_parser = list_sub.add_parser("tests", help="List test scenarios")
    list_tests_parser.add_argument("--kernel", required=True, help="Kernel name")
    list_tests_parser.add_argument("--project", default=".", help="Project directory")

    list_params_parser = list_sub.add_parser("params", help="Show kernel parameters")
    list_params_parser.add_argument("--kernel", required=True, help="Kernel name")
    list_params_parser.add_argument("--project", default=".", help="Project directory")

    # Backward compat: vten list --kernel X (no subcommand) → list tests
    list_parser.add_argument("--kernel", required=False, default=None,
                             help="Kernel name (shorthand for: vten list tests --kernel)")
    list_parser.add_argument("--project", default=".", help="Project directory")

    # vten report
    report_parser = sub.add_parser("report", help="Generate report")
    report_parser.add_argument("--project-dir", default=".", help="Project directory")
    report_parser.add_argument("--format", default="terminal", choices=["terminal", "html", "json"])

    args = parser.parse_args(argv)
    args._parser = parser  # For help in _dispatch

    # Configure logging before any command handler runs
    from vten.log import setup_logging
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    from vten.errors import VTenError

    try:
        _dispatch(args, log_level)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except VTenError as e:
        if args.verbose:
            logger.error("%s", e, exc_info=True)
        else:
            logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            logger.error("internal error", exc_info=True)
        else:
            logger.error("internal error: %s", e)
            logger.error("Re-run with -v for full traceback.")
        sys.exit(2)


def _dispatch(args: argparse.Namespace, log_level: str) -> None:
    """Dispatch to the appropriate subcommand handler."""
    from vten.log import setup_logging

    if args.command == "init":
        from vten.cli.init_cmd import init_project
        init_project(
            args.project_dir,
            kernel_name=args.kernel,
            backend=args.backend,
            add_backend=args.add_backend,
        )

    elif args.command == "build":
        from vten.cli.build import build_project
        if getattr(args, "build_verbose", False) and log_level != "DEBUG":
            setup_logging(level="DEBUG", log_file=args.log_file)
        overrides = {}
        if args.config:
            for item in args.config:
                k, v = item.split("=", 1)
                overrides[k] = int(v) if v.isdigit() else v
        # --target overrides [backend.xrt].target in vten.toml
        if args.target:
            overrides["_xrt_target"] = args.target
        build_project(
            project_dir=args.project,
            kernel_name=args.kernel,
            backend=args.backend,
            stage=args.stage,
            upto=args.upto,
            force=args.force,
            clean=args.clean,
            skip_compile=args.skip_compile,
            config_overrides=overrides or None,
        )

    elif args.command == "run":
        from vten.cli.config_resolver import resolve_config
        from vten.cli.run import run_test

        project = args.project
        kernels_base = Path(project).resolve() / "kernels"
        overrides = resolve_config(
            args.config or [], kernels_base=kernels_base,
        ) if args.config else None

        # -v on run subcommand enables both sim verbose AND Python DEBUG
        effective_sim_verbose = args.sim_verbose or args.verbose
        if effective_sim_verbose and log_level != "DEBUG":
            setup_logging(level="DEBUG", log_file=args.log_file)
        run_test(
            project_dir=project,
            kernel_name=args.kernel,
            test_name=args.test or "",
            backend=args.backend,
            waveform=args.waveform,
            waveform_on_fail=args.waveform_on_fail,
            gui=args.gui,
            sim_verbose=effective_sim_verbose,
            config_overrides=overrides,
            verify=args.verify,
        )

    elif args.command == "list":
        list_cmd = getattr(args, "list_command", None)
        if list_cmd == "params":
            from vten.cli.list_cmd import list_params
            list_params(project_dir=args.project, kernel_name=args.kernel)
        else:
            # Default: list tests (includes backward-compat 'vten list --kernel X')
            from vten.cli.list_cmd import list_tests
            kernel = args.kernel
            if kernel is None:
                print("usage: vten list tests --kernel KERNEL", file=sys.stderr)
                sys.exit(1)
            list_tests(project_dir=args.project, kernel_name=kernel)

    elif args.command == "report":
        from vten.cli.report import generate_report
        print(generate_report(args.project_dir, format=args.format))

    else:
        args._parser.print_help()
        sys.exit(1)



if __name__ == "__main__":
    main()
