"""Inference API — Kernel-Granular Eager Execution.

Provides InferenceSession (eager executor) and InferenceModule (nn.Module wrapper)
for running verified kernels on real FPGA hardware.

Spec reference: 11_inference_api.md
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vten.log import format_elapsed, format_size

from vten.kernel.tensor import Tensor
from vten.runtime.context import ExecutionContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.backend.base import Backend
    from vten.kernel.base import Kernel


class InferenceSession:
    """Kernel-granular eager executor.

    Each run() executes a single kernel and returns Tensor(on_device=True).
    Python controls the data flow between kernels (like PyTorch eager mode).

    Usage::

        # Auto-discover xclbin from project build structure
        session = InferenceSession(
            kernel="npu_pipeline", backend="xrt", target="hw_emu",
        )

        # Or from explicit xclbin path
        session = InferenceSession.from_xclbin("path/to/design.xclbin")

        r1 = session.run(NpuKernel, inputs={"ifm": x}, **L1)
        r2 = session.run(NpuKernel, inputs={"ifm": r1["ofm"]}, **L2)
        y = r2["ofm"].cpu()

        # With per-layer verification:
        r1 = session.run(NpuKernel, inputs={"ifm": x}, verify=True, **L1)
        r2 = session.run(NpuKernel, inputs={"ifm": r1["ofm"]}, verify=True, **L2)
    """

    def __init__(
        self,
        backend: Backend | str = "xrt",
        base_params: dict | None = None,
        *,
        kernel: str | None = None,
        target: str = "hw",
        project_dir: str = ".",
        log_level: str | None = None,
    ) -> None:
        """Create an inference session.

        Args:
            backend: Backend instance, or name ("xrt", "xsim", "verilator").
            base_params: Default parameters for all run() calls.
            kernel: Kernel name for xclbin auto-discovery (e.g. "npu_pipeline").
            target: "hw" or "hw_emu" (used with string backend).
            project_dir: Project root containing vten.toml (default: ".").
            log_level: Log level (e.g. "DEBUG", "INFO"). If None, auto-configures
                to INFO when no vten handlers exist.
        """
        if isinstance(backend, str):
            backend, project_config = self._create_backend(
                backend, kernel=kernel, target=target, project_dir=project_dir,
            )
            # Auto-inject build_params from vten.toml so kernels see Ti, To, etc.
            build_params = project_config.get("build_params")
            if build_params:
                merged = dict(base_params or {})
                merged.setdefault("build_params", {}).update(build_params)
                base_params = merged
        self._backend = backend
        self._base_params = base_params or {}
        self._project_dir = Path(project_dir).resolve()
        self._run_count = 0  # tracks run() calls for logging
        # Auto-configure vten logging if no handlers set (library user mode)
        vten_root = logging.getLogger("vten")
        if log_level or not vten_root.handlers:
            from vten.log import setup_logging
            setup_logging(level=log_level or "INFO")
        # Enable persistent mode for BO pool reuse
        if hasattr(backend, "_persistent"):
            backend._persistent = True

    @staticmethod
    def _create_backend(
        backend_name: str,
        *,
        kernel: str | None = None,
        target: str = "hw",
        project_dir: str = ".",
    ) -> tuple[Backend, dict]:
        """Create backend from name with auto-discovery.

        Uses the central backend registry so that adding a new backend
        only requires updating ``vten.backend.registry``.

        Returns (backend, project_config) so caller can extract build_params.
        """
        from pathlib import Path

        from vten.backend.registry import available_backends, get_backend

        supported = available_backends()
        if backend_name not in supported:
            raise ValueError(
                f"unsupported inference backend: {backend_name!r}"
                f" (supported: {', '.join(supported)})"
            )

        # Load vten.toml if it exists
        project = Path(project_dir).resolve()
        project_config: dict[str, Any] = {}
        toml_path = project / "vten.toml"
        if toml_path.exists():
            from vten.cli.config import load_project_config
            project_config = load_project_config(project)

        # Build RunContext for typed runtime state
        from vten.backend.base import RunContext
        kernel_build_dir = None
        if kernel is not None:
            kernel_dir = project / "kernels" / kernel
            kernel_build_dir = kernel_dir / "build"

        run_ctx = RunContext(
            project_dir=project,
            kernel_build_dir=kernel_build_dir,
        )

        # XRT-specific: inject target and enable persistent BO pool
        kwargs: dict[str, Any] = {}
        if backend_name == "xrt":
            project_config.setdefault("backend", {}).setdefault("xrt", {})["target"] = target
            kwargs["persistent"] = True

        backend = get_backend(backend_name, project_config, **kwargs)
        backend.set_run_context(run_ctx)
        return backend, project_config

    @classmethod
    def from_xclbin(
        cls,
        xclbin_path: str,
        *,
        target: str = "hw",
        base_params: dict | None = None,
    ) -> InferenceSession:
        """Create session from explicit xclbin path.

        Args:
            xclbin_path: Path to .xclbin file.
            target: "hw" for real FPGA, "hw_emu" for hardware emulation.
            base_params: Default parameters for all run() calls.
        """
        from vten.backend.xrt import XrtBackend

        backend = XrtBackend(
            {"backend": {"xrt": {"xclbin_path": xclbin_path, "target": target}}},
            persistent=True,
        )
        return cls(backend, base_params=base_params)

    def run(
        self,
        kernel_class: type[Kernel],
        inputs: dict[str, torch.Tensor | Tensor] | None = None,
        *,
        verify: bool = False,
        **params: Any,
    ) -> dict[str, Tensor]:
        """Execute a single kernel eagerly.

        Args:
            kernel_class: Kernel subclass to execute.
            inputs: Mapping of tensor name to input data.
                torch.Tensor → layout + serialize + LOAD + PUSH
                Tensor(on_device) → skip (BO already on device)
            verify: If True, compare HW output against behavioral model
                    golden (same CompositeKernel.forward() chain as vten run).
                    Golden data is stored on output Tensor.golden for
                    multi-layer chaining.
            **params: Kernel parameters (merged with base_params).

        Returns:
            Dict of output tensor names → Tensor.
            On HW backend: Tensor(on_device=True).
            On SIM backend: Tensor with .data set (host).

        Raises:
            VerificationError: If verify=True and HW output doesn't match golden.
        """
        inputs = inputs or {}
        merged = {**self._base_params, **params}
        spec = merged.pop("_spec", None)

        self._run_count += 1
        run_t0 = _time.monotonic()

        # Layer banner — show run number and key params for context
        label = params.get("name", "") or kernel_class.__name__
        host_names = [
            name for name, d in inputs.items()
            if not (isinstance(d, Tensor) and d.on_device)
        ]
        device_names = [
            name for name, d in inputs.items()
            if isinstance(d, Tensor) and d.on_device
        ]
        parts = []
        if host_names:
            parts.append(f"host: {', '.join(host_names)}")
        if device_names:
            parts.append(f"device: {', '.join(device_names)}")
        input_desc = "; ".join(parts) or "no inputs"
        logger.info(
            "──── run #%d: %s (%s) ────",
            self._run_count, label, input_desc,
        )

        from vten.execution import execute_batch

        batch = execute_batch(
            backend=self._backend,
            kernel_class=kernel_class,
            configs=[merged],
            spec=spec,
            inputs=inputs if inputs else None,
            verify=verify,
            project_dir=self._project_dir,
            quiet=True,
            on_error="raise",
        )

        result = batch.single()

        # Log execution summary (phase-by-phase)
        self._log_execution_summary()

        # Output Tensor objects (with BO binding for HW backends)
        # Golden is set on output tensors by _auto_verify_all when verify=True
        outputs = {name: t for name, t in result.output_tensors.items()}

        run_elapsed = _time.monotonic() - run_t0
        logger.info("  total: %s", format_elapsed(run_elapsed))

        return outputs

    def _log_execution_summary(self) -> None:
        """Log execution phase summary from backend's interpreter."""
        summary = getattr(self._backend, "get_execution_summary", lambda: None)()
        if summary is None:
            return
        for p in summary.phases:
            if p.phase == "configure":
                logger.info("  configure: %d regs (%s)", p.n_cmds, format_elapsed(p.elapsed))
            elif p.phase == "send":
                logger.info("  send: %d tensors (%s)", p.n_tensors, format_size(p.n_bytes))
            elif p.phase == "poll":
                logger.info("  poll: %s, %d polls", format_elapsed(p.elapsed), p.n_polls)
            elif p.phase == "recv":
                logger.info("  recv: %d tensors (%s)", p.n_tensors, format_size(p.n_bytes))
            # trigger: skip (vsync detail unnecessary)

    def upload(
        self,
        data: torch.Tensor,
        tensor_name: str,
        kernel_class: type[Kernel],
        params: dict | None = None,
    ) -> Tensor:
        """Upload a tensor to device memory (1-time, for weights/biases).

        Args:
            data: Host tensor data (logical shape).
            tensor_name: Name of the tensor in the kernel class.
            kernel_class: Kernel class (for layout method lookup).
            params: Parameters for kernel instantiation (shape resolution).

        Returns:
            Tensor(on_device=True) with BO bound.
        """
        merged = {**self._base_params, **(params or {})}
        spec = merged.pop("_spec", None)

        logger.info(
            "upload: %s (%s, %s)",
            tensor_name, list(data.shape), format_size(data.numel() * data.element_size()),
        )

        # For CompositeKernel, find the sub-kernel that owns tensor_name
        # and instantiate only that sub-kernel (avoids serializing unrelated tensors).
        from vten.kernel.composite import CompositeKernel
        upload_cls = kernel_class
        if isinstance(kernel_class, type) and issubclass(kernel_class, CompositeKernel):
            for (sub_name, t_name), exposed_name in kernel_class._auto_exposed.items():
                if t_name == tensor_name or exposed_name == tensor_name:
                    upload_cls = kernel_class._sub_kernel_refs[sub_name]
                    break

        # Create a temporary context to instantiate kernel for layout/packing info
        ctx = ExecutionContext(
            backend=self._backend,
            project_params=merged,
            mode="inference",
            project_dir=self._project_dir,
        )
        ki = ctx.instantiate(upload_cls, spec=spec, **merged)
        tensor = ki.get_tensor(tensor_name)

        # Assign data and let the normal compile pipeline handle layout+serialize
        tensor.data = data

        is_hw = self._backend.compile_target == "hw"

        if is_hw:
            # HW: execute LOAD+PUSH to create BO on device
            h = ctx.push_tensor(tensor)
            result = ctx.run()

            compiled = ctx._last_compiled
            view = compiled.flattened_view
            exposed = view.exposed_tensors.get(tensor_name)

            t = Tensor(
                shape=tensor._resolved_shape or tensor.shape,
                dtype=tensor.dtype,
                interface=tensor.interface,
                direction=tensor.direction,
            )
            t.name = tensor_name
            t._resolved_shape = tensor._resolved_shape
            t._element_count = tensor._element_count
            t.golden = data

            if exposed is not None:
                buffer_id = compiled.buffer_ids.get(tensor_name)
                if buffer_id is not None:
                    bo = self._backend.get_buffer_object(buffer_id)
                    if bo is not None:
                        bo_size = bo.size() if hasattr(bo, "size") else exposed._serialized_size
                        t._bind_bo(bo, bo_size)
        else:
            # SIM: no device memory — just store host data for later run()
            t = Tensor(
                shape=tensor._resolved_shape or tensor.shape,
                dtype=tensor.dtype,
                interface=tensor.interface,
                direction=tensor.direction,
            )
            t.name = tensor_name
            t._resolved_shape = tensor._resolved_shape
            t._element_count = tensor._element_count
            t.golden = data
            t.data = data

        return t

    def run_pipeline(
        self,
        kernel_class: type[Kernel],
        layers: list[dict],
        inputs: dict[str, torch.Tensor | Tensor],
        per_layer_inputs: list[dict[str, torch.Tensor | Tensor]] | None = None,
        chain: dict[str, str] | None = None,
        verify: bool = False,
    ) -> dict[str, Tensor]:
        """Sequential chain convenience. Internally calls run() per layer.

        Args:
            kernel_class: Kernel class for all layers.
            layers: Per-layer parameter dicts.
            inputs: Initial inputs (first layer).
            per_layer_inputs: Per-layer additional inputs (weights, biases).
            chain: Output→input name mapping (default: {"ofm_mem": "ifm_mem"}).
            verify: If True, verify each layer against behavioral model golden.

        Returns:
            Output dict from the last layer.
        """
        chain = chain or {"ofm_mem": "ifm_mem"}
        per_layer_inputs = per_layer_inputs or [{} for _ in layers]

        n = len(layers)
        logger.info("════ pipeline: %d layers (%s) ════", n, kernel_class.__name__)
        pipe_t0 = _time.monotonic()

        current = dict(inputs)
        result: dict[str, Tensor] | None = None
        for i, layer_params in enumerate(layers):
            layer_params = {**layer_params, "name": layer_params.get("name", f"layer {i}/{n}")}
            merged_inputs = {**current, **per_layer_inputs[i]}
            result = self.run(
                kernel_class, inputs=merged_inputs, verify=verify, **layer_params,
            )
            # Chain: map output names to next layer's input names
            current = {
                dst: result[src]
                for src, dst in chain.items()
                if src in result
            }

        if result is None:
            raise ValueError("layers list is empty")

        pipe_elapsed = _time.monotonic() - pipe_t0
        logger.info(
            "════ pipeline done: %d layers, %s ════", n, format_elapsed(pipe_elapsed),
        )
        return result

    def cleanup(self) -> None:
        """Release all device resources. BO references become invalid."""
        self._backend.cleanup()


class InferenceModule(torch.nn.Module):
    """nn.Module wrapper for FPGA kernel inference.

    Subclass and set kernel_cls, input_name, output_name.

    Usage::

        class NPUConv3D(InferenceModule):
            kernel_cls = NpuPipelineKernel
            input_name = "ifm_mem"
            output_name = "ofm_mem"

        conv = NPUConv3D(session, weight=w, bias=b, **params)
        y = conv(x)  # Tensor(on_device)
        y.cpu()       # → torch.Tensor

        # With verification:
        y = conv(x, verify=True)
    """

    kernel_cls: type[Kernel]
    input_name: str = "ifm_mem"
    output_name: str = "ofm_mem"

    def __init__(
        self,
        session: InferenceSession,
        *,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        weight_name: str = "wgt_mem",
        bias_name: str = "bias_mem",
        **params: Any,
    ) -> None:
        super().__init__()
        self._session = session
        self._params = params
        self._extra_inputs: dict[str, Tensor] = {}

        if weight is not None:
            self._extra_inputs[weight_name] = session.upload(
                weight, weight_name, self.kernel_cls, params,
            )
        if bias is not None:
            self._extra_inputs[bias_name] = session.upload(
                bias, bias_name, self.kernel_cls, params,
            )

    def forward(
        self,
        x: torch.Tensor | Tensor,
        *,
        verify: bool = False,
        **extra_inputs: torch.Tensor | Tensor,
    ) -> Tensor:
        """Execute kernel. Returns Tensor(on_device=True).

        Args:
            x: Primary input tensor.
            verify: If True, verify HW output against behavioral model golden.
            **extra_inputs: Additional per-call inputs (e.g. concat_mem=skip).
        """
        inputs: dict[str, torch.Tensor | Tensor] = {self.input_name: x}
        inputs.update(self._extra_inputs)
        inputs.update(extra_inputs)
        result = self._session.run(
            self.kernel_cls, inputs=inputs, verify=verify, **self._params,
        )
        return result[self.output_name]


# ═══════════════════════════════════════════════════════════════════
# InferenceModel — whole-network orchestration with graph capture
# ═══════════════════════════════════════════════════════════════════


@dataclass
class GraphEdge:
    """One input binding of a node invocation.

    ``source`` is the ``node_name`` of the producing node, or the sentinel
    ``"input"`` when the tensor entered the graph from outside (a graph input
    such as the ``x`` passed to ``forward()`` or an uploaded weight/bias).
    ``tensor_name`` is the kernel-side input slot the tensor was bound to
    (e.g. ``"data_in"``, or ``"concat_mem"`` for a skip connection).
    """

    tensor_name: str
    source: str  # producing node_name, or "input"
    source_tensor_id: int


@dataclass
class GraphNode:
    """A single node invocation recorded during eager ``forward()``.

    One record is appended per ``node(...)`` call.  Because capture happens as
    a *side effect of execution*, calling the same node twice (e.g. in a loop)
    produces two records — the graph is an execution trace, not a static DAG.
    """

    node_name: str
    kernel: str  # kernel class __name__
    inputs: list[GraphEdge] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    output_tensor_id: int | None = None
    output_name: str | None = None
    # Hook slots for the later verify/perf agent (Slice C/D). Left empty here;
    # never read by capture/execution so filling them cannot break the trace.
    verification: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


class _Node:
    """A tracked stage in an :class:`InferenceModel`.

    Created via :meth:`InferenceModel.stage`. Calling the node runs its kernel
    eagerly through the shared :class:`InferenceSession` (mirroring
    :class:`InferenceModule`) *and* records the invocation into the owning
    model's dataflow graph. Do not construct directly.
    """

    def __init__(
        self,
        model: InferenceModel,
        kernel_cls: type[Kernel],
        *,
        name: str,
        input_name: str,
        output_name: str,
        extra_inputs: dict[str, Tensor],
        params: dict[str, Any],
    ) -> None:
        self._model = model
        self.kernel_cls = kernel_cls
        self.name = name
        self.input_name = input_name
        self.output_name = output_name
        self._extra_inputs = extra_inputs  # uploaded weights/biases (graph inputs)
        self._params = params

    def __call__(
        self,
        x: torch.Tensor | Tensor,
        *,
        verify: bool = False,
        **extra_inputs: torch.Tensor | Tensor,
    ) -> Tensor:
        """Run this node eagerly and capture the invocation into the graph.

        Args:
            x: Primary input, bound to ``self.input_name``. May be a raw
                ``torch.Tensor`` (a graph input) or an on-device / host
                ``vten.Tensor`` produced by an upstream node (a graph edge).
            verify: Forwarded to ``session.run`` (per-node verify hook).
            **extra_inputs: Additional per-call inputs, e.g. a skip connection
                passed as ``concat_mem=<upstream tensor>``. Each is bound by
                keyword name and captured as an edge — this is how fan-in /
                skip-joins show up in the graph.

        Returns:
            The on-device (HW) or host (SIM/CPU) output ``Tensor``, also
            registered as this node's output in the model's id→node map so a
            downstream node consuming it records the correct edge.
        """
        inputs: dict[str, torch.Tensor | Tensor] = {self.input_name: x}
        inputs.update(self._extra_inputs)
        inputs.update(extra_inputs)

        # ── Graph capture (BEFORE run): resolve each input tensor's producer ──
        edges: list[GraphEdge] = []
        for tensor_name, value in inputs.items():
            edges.append(self._model._resolve_edge(tensor_name, value))

        self._model._on_before_node(self, inputs)  # no-op hook (verify agent)

        result = self._model._session.run(
            self.kernel_cls, inputs=inputs, verify=verify, **self._params,
        )
        out = result[self.output_name]

        # ── Graph capture (AFTER run): register this node as out's producer ──
        # Drop private/plumbing keys (e.g. "_spec", a KernelSpec object) from the
        # captured params so graph() stays plain-data / JSON-serializable.
        captured_params = {
            k: v for k, v in self._params.items() if not k.startswith("_")
        }
        record = GraphNode(
            node_name=self.name,
            kernel=self.kernel_cls.__name__,
            inputs=edges,
            params=captured_params,
            output_tensor_id=id(out),
            output_name=self.output_name,
        )
        self._model._register_output(out, self, record)
        self._model._on_after_node(self, record, out)  # no-op hook (perf agent)
        return out


class InferenceModel:
    """Whole-network orchestration container with internal graph capture.

    A hybrid **imperative + graph-capturing** model. You compose multiple vTen
    kernels into one model; ``forward()`` runs them *eagerly on device* through a
    shared :class:`InferenceSession` (the same zero-copy device chaining as
    :class:`InferenceModule`), while the model records the dataflow graph as a
    *side effect* of that execution. Execution **is** the capture — there is no
    deferred/lazy graph build and no second execution path.

    Subclass and override two methods::

        class MyNet(InferenceModel):
            def build(self):
                # declare stages once (uploads weights/biases here)
                self.scale  = self.stage(ScaleKernel,  scale_factor=2, N=32)
                self.off1   = self.stage(OffsetKernel, offset_value=1, N=32,
                                         name="off1")
                self.off2   = self.stage(OffsetKernel, offset_value=2, N=32,
                                         name="off2")

            def forward(self, x):
                h = self.scale(x)      # h fans out to two consumers below
                a = self.off1(h)
                b = self.off2(h)
                return a               # (b is still captured as a node)

        net = MyNet(session)
        y = net(x)              # runs build() once, resets graph, runs forward()
        graph = net.graph()     # {"nodes": [...], "edges": [...]}

    Graph capture (the id-map mechanism)
    ------------------------------------
    The model keeps an **external identity map** ``id(tensor) -> (node, record)``
    — it never mutates the core :class:`~vten.kernel.tensor.Tensor` to carry
    provenance. On every node call, for each bound input tensor it looks up
    ``id(tensor)`` in that map: a hit records an edge from the producing node; a
    miss records a graph input (sentinel source ``"input"``, e.g. the ``x`` arg
    or an uploaded weight). After the kernel runs, the output tensor's ``id`` is
    registered → this node. Because the same physical output object is passed to
    every consumer, one tensor consumed by two nodes yields **fan-out** in the
    graph, and a tensor threaded through as ``concat_mem=`` yields a **skip**
    edge — both fall out of the id-map automatically.

    Hook points ``_on_before_node`` / ``_on_after_node`` are intentional no-ops
    here; the later verification (Slice C) and perf/memory rollup (Slice D)
    agents attach per-node behavior there and populate ``GraphNode.verification``
    / ``GraphNode.stats`` without restructuring this class.
    """

    def __init__(self, session: InferenceSession) -> None:
        self._session = session
        self._built = False
        self._nodes: dict[str, _Node] = {}
        self._auto_name_counts: dict[str, int] = {}
        # Graph state (reset per forward): trace of node invocations …
        self._graph: list[GraphNode] = []
        # … and the external identity map: id(tensor) -> (producing node, record).
        self._producers: dict[int, tuple[_Node, GraphNode]] = {}

    # ── Declaration ──

    def stage(
        self,
        kernel_cls: type[Kernel],
        *,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        weight_name: str = "wgt_mem",
        bias_name: str = "bias_mem",
        input_name: str = "data_in",
        output_name: str = "data_out",
        name: str | None = None,
        **params: Any,
    ) -> _Node:
        """Declare a tracked stage (a graph node factory).

        Mirrors :class:`InferenceModule`: weights/biases are uploaded to the
        device **once** here (not per call) and become fixed graph inputs on
        every invocation. Call the returned node inside ``forward()`` to run it.

        Args:
            kernel_cls: Kernel subclass to execute for this stage.
            weight, bias: Optional constants uploaded once via
                ``session.upload`` and bound to every call under
                ``weight_name`` / ``bias_name``.
            input_name, output_name: Kernel tensor slot names for the primary
                input/output (default ``"data_in"``/``"data_out"``).
            name: Node name in the captured graph. Auto-derived from the kernel
                class (deduplicated, e.g. ``offset#1``) when omitted.
            **params: Kernel parameters merged into each ``session.run`` (e.g.
                ``scale_factor``, ``offset_value``, ``N``, ``_spec``).

        Returns:
            A callable :class:`_Node`.
        """
        if name is None:
            name = self._auto_name(kernel_cls.__name__)
        elif name in self._nodes:
            raise ValueError(f"duplicate stage name: {name!r}")

        extra_inputs: dict[str, Tensor] = {}
        if weight is not None:
            extra_inputs[weight_name] = self._session.upload(
                weight, weight_name, kernel_cls, params,
            )
        if bias is not None:
            extra_inputs[bias_name] = self._session.upload(
                bias, bias_name, kernel_cls, params,
            )

        node = _Node(
            self,
            kernel_cls,
            name=name,
            input_name=input_name,
            output_name=output_name,
            extra_inputs=extra_inputs,
            params=params,
        )
        self._nodes[name] = node
        return node

    def _auto_name(self, kernel_name: str) -> str:
        """Derive a unique, stable node name from a kernel class name."""
        base = kernel_name.lower()
        if base.endswith("kernel"):
            base = base[: -len("kernel")] or kernel_name.lower()
        count = self._auto_name_counts.get(base, 0)
        self._auto_name_counts[base] = count + 1
        return base if count == 0 else f"{base}#{count}"

    def build(self) -> None:
        """Declare the model's stages. Override in subclasses.

        Called exactly once, lazily, on the first ``net(x)``. Use
        :meth:`stage` here to create nodes and upload any weights/biases.
        """

    def forward(self, x: torch.Tensor | Tensor) -> Tensor:
        """Imperative dataflow. Override in subclasses.

        Call the nodes declared in :meth:`build`, wiring outputs to inputs in
        plain Python. Executes eagerly; the graph is captured as a side effect.
        """
        raise NotImplementedError("InferenceModel subclasses must implement forward()")

    # ── Execution + capture ──

    def __call__(self, x: torch.Tensor | Tensor) -> Tensor:
        """Run the model: lazily ``build()``, reset the graph, then ``forward()``.

        Returns the output ``Tensor`` of ``forward()``. Inspect the captured
        dataflow with :meth:`graph`.
        """
        if not self._built:
            self.build()
            self._built = True
        # Fresh trace per forward — execution IS the capture.
        self._graph = []
        self._producers = {}
        return self.forward(x)

    def _resolve_edge(
        self, tensor_name: str, value: torch.Tensor | Tensor,
    ) -> GraphEdge:
        """Look up a bound input's producer in the id-map → build an edge.

        A hit means the tensor was produced by an upstream node (a real graph
        edge, incl. fan-out and skips); a miss means it entered from outside
        (the ``forward`` argument, or an uploaded weight/bias) → source
        ``"input"``.
        """
        producer = self._producers.get(id(value))
        source = producer[0].name if producer is not None else "input"
        return GraphEdge(
            tensor_name=tensor_name,
            source=source,
            source_tensor_id=id(value),
        )

    def _register_output(
        self, out: Tensor, node: _Node, record: GraphNode,
    ) -> None:
        """Append the node record and mark ``out`` as produced by ``node``."""
        self._graph.append(record)
        self._producers[id(out)] = (node, record)

    # ── Graph accessor ──

    def graph(self) -> dict[str, Any]:
        """Return the captured dataflow graph from the last ``forward()``.

        Structure (plain dicts/lists, safe to serialize)::

            {
              "nodes": [
                {"name", "kernel", "params", "output_name",
                 "inputs": [{"tensor_name", "source"}, ...]},
                ...
              ],
              "edges": [{"src", "dst", "tensor_name"}, ...],  # producer→consumer
            }

        ``src == "input"`` marks a graph input (forward arg / uploaded weight).
        Node order is execution order; fan-out shows up as one ``src`` node
        appearing in multiple edges. Intended for the later verify/perf agent.
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for rec in self._graph:
            nodes.append(
                {
                    "name": rec.node_name,
                    "kernel": rec.kernel,
                    "params": dict(rec.params),
                    "output_name": rec.output_name,
                    "inputs": [
                        {"tensor_name": e.tensor_name, "source": e.source}
                        for e in rec.inputs
                    ],
                }
            )
            for e in rec.inputs:
                edges.append(
                    {
                        "src": e.source,
                        "dst": rec.node_name,
                        "tensor_name": e.tensor_name,
                    }
                )
        return {"nodes": nodes, "edges": edges}

    # ── Hook points (no-ops now; filled by verify/perf agent later) ──

    def _on_before_node(
        self, node: _Node, inputs: dict[str, torch.Tensor | Tensor],
    ) -> None:
        """Hook fired just before a node runs. No-op (verification agent)."""

    def _on_after_node(
        self, node: _Node, record: GraphNode, output: Tensor,
    ) -> None:
        """Hook fired just after a node runs. No-op (perf/memory agent)."""
