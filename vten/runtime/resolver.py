"""Stage 1: Parameter Resolution.

Spec reference: 02_runtime_engine.md §6, 09_user_api.md §9
"""

from __future__ import annotations

import logging
import re

from vten.errors import ParameterResolutionError

logger = logging.getLogger(__name__)


def safe_eval(expr: str) -> int:
    """Evaluate arithmetic expression (integers only, no builtins)."""
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:
        raise ParameterResolutionError(
            f"Failed to evaluate expression '{expr}': {exc}"
        ) from exc
    if isinstance(result, float):
        if result != int(result):
            raise ParameterResolutionError(
                f"Expression '{expr}' evaluated to non-integer float {result}"
            )
        result = int(result)
    return result


class ParameterResolver:
    """Hierarchical parameter resolver with 5-tier merge.

    Priority (high → low):
        Tier 5. runtime_params (test Config / ctx.instantiate kwargs)
        Tier 4. kernel_spec_params (kernel_spec.yaml parameters)
        Tier 3. default_params (Kernel.default_params)
        Tier 2. build_params (vten.toml [build_params] + spec build_params)
        Tier 1. project_params (vten.toml [parameters])
    """

    def __init__(
        self,
        project_params: dict,
        kernel_params: dict,
        runtime_params: dict,
        build_params: dict | None = None,
        default_params: dict | None = None,
    ) -> None:
        self._build_param_keys: set[str] = set()
        self.namespace: dict[str, int | str] = {}

        # Tier 1: project params
        self.namespace.update(project_params)

        # Tier 2: build params
        if build_params:
            self.namespace.update(build_params)
            self._build_param_keys = set(build_params.keys())

        # Tier 3: Kernel.default_params (setdefault — supplements only)
        if default_params:
            for k, v in default_params.items():
                self.namespace.setdefault(k, v)

        # Tier 4: kernel_spec parameters
        self.namespace.update(kernel_params)

        # Tier 5: test override — warn if overriding build_params
        if self._build_param_keys:
            for k in runtime_params:
                if k in self._build_param_keys:
                    logger.warning(
                        "runtime param '%s' overrides build_param "
                        "(synthesis-time constant)", k
                    )
        self.namespace.update(runtime_params)

        # Resolve any string values that are themselves ${} expressions
        self._resolve_namespace()

    def _resolve_namespace(self) -> None:
        """Iteratively resolve namespace values that contain ${} refs.

        Detects circular references: if iterations exhaust without
        convergence (changed=True every round), it's a cycle.
        Unresolved params that reference missing keys are left as-is
        (they'll error at resolve() call time).
        """
        max_iterations = 10
        for iteration in range(max_iterations):
            changed = False
            for key, value in list(self.namespace.items()):
                if isinstance(value, str) and "${" in value:
                    try:
                        resolved = self.resolve(value)
                        if resolved != value:
                            self.namespace[key] = resolved
                            changed = True
                    except ParameterResolutionError:
                        pass  # May resolve in a later iteration
            if not changed:
                break
        else:
            # Exhausted all iterations while still changing — circular reference
            circular = [
                k for k, v in self.namespace.items()
                if isinstance(v, str) and "${" in v
            ]
            if circular:
                raise ParameterResolutionError(
                    f"Circular parameter reference detected after "
                    f"{max_iterations} iterations: {circular}"
                )

    def resolve(self, expr: str | int) -> int | str:
        """Resolve a single expression.

        - int values pass through unchanged
        - str without ${} pass through unchanged
        - str with ${} get substituted and evaluated
        """
        if not isinstance(expr, str):
            return expr
        if "${" not in str(expr):
            # Try to evaluate as numeric literal/expression
            try:
                return safe_eval(expr)
            except (ParameterResolutionError, SyntaxError, ValueError):
                return expr

        def _substitute(m: re.Match) -> str:
            name = m.group(1)
            if name not in self.namespace:
                raise ParameterResolutionError(
                    f"Unresolved parameter '${{{name}}}' in expression "
                    f"'{expr}'. Available: {sorted(self.namespace)}"
                )
            return str(self.namespace[name])

        resolved_str = re.sub(r"\$\{(\w+)\}", _substitute, str(expr))
        return safe_eval(resolved_str)
