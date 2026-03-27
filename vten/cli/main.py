"""vten CLI entry point.

Spec reference: 06_codegen_and_cli.md §4
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vten", description="vTen verification framework")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output (DEBUG level)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress info messages (WARNING level)")
    parser.add_argument("--log-file", default=None,
                        help="Write debug log to file")
    sub = parser.add_subparsers(dest="command")

    # vten init
    init_parser = sub.add_parser("init", help="Create project skeleton")
    init_parser.add_argument("project_dir", help="Project directory to create")
    init_parser.add_argument("--kernel", help="Add kernel directory to existing project")
    init_parser.add_argument("--backend", default=None,
        choices=["xsim", "verilator", "xrt"],
        help="Target backend for new project (default: xsim)")
    init_parser.add_argument("--add-backend", default=None,
        choices=["xsim", "verilator", "xrt"],
        help="Add backend section to existing project")

    # vten build
    build_parser = sub.add_parser("build", help="Build project")
    build_parser.add_argument("--project", default=".", help="Project directory")
    build_parser.add_argument("--kernel", help="Build specific kernel only")
    build_parser.add_argument("--backend", default=None,
        choices=["xsim", "verilator", "xrt"],
        help="Target backend (default: from vten.toml or xsim)")
    build_parser.add_argument("--stage", help="Run specific stage only")
    build_parser.add_argument("--upto", help="Run stages up to (inclusive)")
    build_parser.add_argument("--force", action="store_true", help="Ignore cache, full rebuild")
    build_parser.add_argument("--skip-compile", action="store_true", help="Run codegen only")
    build_parser.add_argument("--config", nargs="*", help="Config overrides (K=V)")

    # vten run
    run_parser = sub.add_parser("run", help="Run test")
    run_parser.add_argument("--kernel", required=True, help="Kernel name")
    run_parser.add_argument("--test", default=None, help="Test scenario name (omit to run all)")
    run_parser.add_argument("--project", default=".", help="Project directory")
    run_parser.add_argument("--backend", default=None,
        choices=["xsim", "verilator", "xrt"],
        help="Target backend (default: from vten.toml or xsim)")
    run_parser.add_argument("--waveform", action="store_true", help="Enable waveform dump")
    run_parser.add_argument("--gui", action="store_true", help="xsim GUI mode")
    run_parser.add_argument("-v", "--sim-verbose", action="store_true",
                            help="Enable simulator verbose output (+VTEN_VERBOSE)")
    run_parser.add_argument("--config", nargs="*", help="Config overrides (K=V)")

    # vten report
    report_parser = sub.add_parser("report", help="Generate report")
    report_parser.add_argument("--project-dir", default=".", help="Project directory")
    report_parser.add_argument("--format", default="terminal", choices=["terminal", "html", "json"])

    args = parser.parse_args(argv)

    # Configure logging before any command handler runs
    from vten.log import setup_logging
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

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
        overrides = {}
        if args.config:
            for item in args.config:
                k, v = item.split("=", 1)
                overrides[k] = int(v) if v.isdigit() else v
        build_project(
            project_dir=args.project,
            kernel_name=args.kernel,
            backend=args.backend,
            stage=args.stage,
            upto=args.upto,
            force=args.force,
            skip_compile=args.skip_compile,
            config_overrides=overrides or None,
        )

    elif args.command == "run":
        from vten.cli.run import run_test
        overrides = {}
        if args.config:
            for item in args.config:
                k, v = item.split("=", 1)
                overrides[k] = int(v) if v.isdigit() else v
        # -v on run subcommand enables both sim verbose AND Python DEBUG
        effective_sim_verbose = args.sim_verbose or args.verbose
        if effective_sim_verbose and log_level != "DEBUG":
            setup_logging(level="DEBUG", log_file=args.log_file)
        run_test(
            project_dir=args.project,
            kernel_name=args.kernel,
            test_name=args.test or "",
            backend=args.backend,
            waveform=args.waveform,
            gui=args.gui,
            sim_verbose=effective_sim_verbose,
            config_overrides=overrides or None,
        )

    elif args.command == "report":
        from vten.cli.report import generate_report
        print(generate_report(args.project_dir, format=args.format))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
