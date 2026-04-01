"""BuildPipeline ABC — backend-specific build orchestration.

Spec reference: 08_backend_abstraction.md §8
"""

from __future__ import annotations

import abc
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class BuildPipeline(abc.ABC):
    """Abstract build pipeline for a specific backend.

    Subclasses define their own stage list and implement each stage.
    The build() method orchestrates stage execution with filtering
    (--stage, --upto, --skip-compile) and per-kernel iteration.
    """

    def __init__(self, project: Path, config: dict) -> None:
        self._project = project
        self._config = config

    # ── Shared IP utilities ──

    @staticmethod
    def _normalize_ip_config(ip_raw: dict | list | None) -> list[dict]:
        """Normalize [ip] table or [[ip]] array to unified list[dict].

        Supports:
          [ip] sources = ["a.xci"]          -> [{"source": "a.xci"}]
          [[ip]] source = "a.xci"           -> [{"source": "a.xci"}]
          [[ip]] vlnv = "x:y:z:1.0" ...    -> [{"vlnv": ..., ...}]
        """
        if ip_raw is None:
            return []
        if isinstance(ip_raw, list):
            return ip_raw
        if isinstance(ip_raw, dict):
            entries: list[dict] = []
            for src in ip_raw.get("sources", []):
                entries.append({"source": src})
            return entries
        return []

    @staticmethod
    def _parse_ip_entries(
        ip_list: list[dict], project: Path,
    ) -> tuple[list[str], list[dict]]:
        """Parse unified [[ip]] entries into (ip_sources, ip_create).

        Entries with 'source' key are existing .xci references (glob supported).
        Entries with 'vlnv' key are declarative IP creation requests.
        """
        ip_sources: list[str] = []
        ip_create: list[dict] = []
        for entry in ip_list:
            if "source" in entry:
                ip_sources.extend(
                    str(p) for p in sorted(project.glob(entry["source"]))
                )
            elif "vlnv" in entry:
                vendor, library, component, version = entry["vlnv"].split(":")
                ip_create.append({
                    "name": entry["name"],
                    "vendor": vendor,
                    "library": library,
                    "component": component,
                    "version": version,
                    "output_dir": f"build/ip/{entry['name']}",
                    "properties": entry.get("properties", {}),
                })
        return ip_sources, ip_create

    @abc.abstractmethod
    def stages(self) -> list[str]:
        """Return ordered list of stage names for this pipeline."""
        ...

    @abc.abstractmethod
    def run_stage(
        self,
        stage: str,
        kernel_name: str | None,
        kernel_dir: Path | None,
        force: bool,
    ) -> None:
        """Execute a single stage.

        Args:
            stage: Stage name (must be in stages()).
            kernel_name: Current kernel being built (None for project-level stages).
            kernel_dir: Kernel directory (None for project-level stages).
            force: Ignore cache, full rebuild.
        """
        ...

    @abc.abstractmethod
    def project_level_stages(self) -> list[str]:
        """Return stage names that run once at project level (not per-kernel)."""
        ...

    def _clean_kernel(self, kernel_dir: Path) -> None:
        """Remove kernel build artifacts."""
        build_dir = kernel_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            logger.info("  cleaned %s", build_dir)

    def _clean_project(self) -> None:
        """Remove project-level build artifacts."""
        build_dir = self._project / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            logger.info("  cleaned %s", build_dir)

    def build(
        self,
        kernel_name: str | None = None,
        stage: str | None = None,
        upto: str | None = None,
        force: bool = False,
        clean: bool = False,
        skip_compile: bool = False,
        config_overrides: dict | None = None,
    ) -> None:
        """Orchestrate build: resolve stages, iterate kernels, run stages."""
        from vten.build.common import discover_kernels, resolve_stages
        from vten.errors import BuildError

        if config_overrides:
            self._config.setdefault("parameters", {}).update(config_overrides)

        all_stages = self.stages()
        target_stages = resolve_stages(all_stages, stage, upto, skip_compile)
        project_stages = set(self.project_level_stages())

        target_kernels = (
            [kernel_name] if kernel_name else discover_kernels(self._project)
        )
        if not target_kernels:
            raise BuildError("No kernels found. Run: vten init --kernel <name>")

        # Clean build artifacts before building
        if clean:
            # Clean project-level build if any project stages are targeted
            if any(s in project_stages for s in target_stages):
                self._clean_project()
            # Clean kernel build dirs
            for kname in target_kernels:
                self._clean_kernel(self._project / "kernels" / kname)

        # Project-level stages (run once)
        for s in target_stages:
            if s in project_stages:
                self.run_stage(s, kernel_name=None, kernel_dir=None, force=force)

        # Per-kernel stages
        for kname in target_kernels:
            kernel_dir = self._project / "kernels" / kname
            logger.info("=== Kernel: %s ===", kname)
            for s in target_stages:
                if s not in project_stages:
                    self.run_stage(s, kernel_name=kname, kernel_dir=kernel_dir, force=force)

        logger.info("Build complete.")
