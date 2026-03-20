"""vten CLI entry point.

Spec reference: 06_codegen_and_cli.md §4
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vten", description="vTen verification framework")
    sub = parser.add_subparsers(dest="command")

    # vten init
    init_parser = sub.add_parser("init", help="Create project skeleton")
    init_parser.add_argument("project_dir", help="Project directory to create")

    # vten build
    build_parser = sub.add_parser("build", help="Build project")
    build_parser.add_argument("--project-dir", default=".", help="Project directory")
    build_parser.add_argument("--config", nargs="*", help="Config overrides (K=V)")

    # vten run
    run_parser = sub.add_parser("run", help="Run test")
    run_parser.add_argument("--test", required=True, help="Test scenario name")
    run_parser.add_argument("--project-dir", default=".", help="Project directory")

    # vten report
    report_parser = sub.add_parser("report", help="Generate report")
    report_parser.add_argument("--project-dir", default=".", help="Project directory")
    report_parser.add_argument("--format", default="terminal", choices=["terminal", "html", "json"])

    args = parser.parse_args(argv)

    if args.command == "init":
        from vten.cli.init_cmd import init_project
        init_project(args.project_dir)

    elif args.command == "build":
        from vten.cli.build import build_project
        overrides = {}
        if args.config:
            for item in args.config:
                k, v = item.split("=", 1)
                overrides[k] = int(v) if v.isdigit() else v
        build_project(args.project_dir, config_overrides=overrides or None)

    elif args.command == "run":
        from vten.cli.run import run_test
        run_test(args.project_dir, test_name=args.test)

    elif args.command == "report":
        from vten.cli.report import generate_report
        print(generate_report(args.project_dir, format=args.format))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
