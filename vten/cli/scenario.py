"""TestScenario — base class for user-defined test scenarios.

Spec reference: 06_codegen_and_cli.md §4.4
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
