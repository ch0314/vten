"""BuildPipeline ABC — backend-specific build orchestration.

Spec reference: 08_backend_abstraction.md §8
"""

from __future__ import annotations

import abc
from pathlib import Path


class BuildPipeline(abc.ABC):
    """Abstract build pipeline for a specific backend.

    Subclasses define their own stage list and implement each stage.
    The build() method orchestrates stage execution with filtering
    (--stage, --upto, --skip-compile) and per-kernel iteration.
    """

    def __init__(self, project: Path, config: dict) -> None:
        self._project = project
        self._config = config

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

    def build(
        self,
        kernel_name: str | None = None,
        stage: str | None = None,
        upto: str | None = None,
        force: bool = False,
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

        # Project-level stages (run once)
        for s in target_stages:
            if s in project_stages:
                self.run_stage(s, kernel_name=None, kernel_dir=None, force=force)

        # Per-kernel stages
        for kname in target_kernels:
            kernel_dir = self._project / "kernels" / kname
            print(f"\n=== Kernel: {kname} ===")
            for s in target_stages:
                if s not in project_stages:
                    self.run_stage(s, kernel_name=kname, kernel_dir=kernel_dir, force=force)

        print("\nBuild complete.")
