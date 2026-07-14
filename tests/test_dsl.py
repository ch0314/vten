"""Tests for vten.dsl — DSL operations, OperationHandle, dependency model.

Spec reference: 01_kernel_and_dsl.md §3 (DSL Operations)
               00_data_models.md §6 (OpKind), §7 (Operation/OperationHandle)
NPU 3D patterns: npu_3d_analysis.md §14 (test_dsl.py)

NPU 3D execution sequence:
  Host workflow: PUSH(ifm) → PUSH(wgt) → PUSH(bias)
                 → CONFIGURE → WRITE_REG(vsync) × 3
                 → POLL_REG(layer_done) → PULL(ofm)
  VSYNC order: bias_loader → weight_loader → fmapIO
  Dependencies: configure dep on 3 pushes, fmapIO vsync dep on bias/wgt vsync
"""

from __future__ import annotations

import pytest
import torch

from vten.kernel.tensor import Tensor
from vten.kernel.base import Kernel, register
from vten.dsl.operations import Operation, OperationHandle
from vten.spec.models import OpKind


# ── NPU 3D sub-kernel stubs for DSL testing ──────────────────────


class FmapIOStub(Kernel):
    """fmapIO — IFM/OFM DDR transfer."""
    spec = "design/fmapIO/rtl/fmapIO_top.yaml"
    ifm = Tensor(
        shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
        dtype=torch.int8, interface="ddr",
    )
    ofm = Tensor(
        shape=("${OUT_CH}", "${OUT_DEPTH}", "${OUT_HEIGHT}", "${OUT_WIDTH}"),
        dtype=torch.int8, interface="ddr",
    )
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.ifm.fill_random(generator=torch.Generator().manual_seed(seed or 0))

    def forward(self, **inputs):
        return self.ifm.to_float()


class BiasLoaderStub(Kernel):
    """bias_loader — int32 bias, DDR."""
    spec = "design/bias_loader/rtl/bias_loader_top.yaml"
    bias = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.bias.fill_random(generator=torch.Generator().manual_seed(seed or 0))

    def forward(self, **inputs):
        return self.bias.to_float()


class WeightLoaderStub(Kernel):
    """weight_loader — int8 weight, HBM 32-bank."""
    spec = "design/weight_loader/rtl/weight_loader_top.yaml"
    weight = Tensor(
        shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
        dtype=torch.int8, interface="hbm",
    )
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.weight.fill_random(generator=torch.Generator().manual_seed(seed or 0))

    def forward(self, **inputs):
        return self.weight.to_float()


# ═══════════════════════════════════════════════════════════════════
# §1  OpKind enum
# ═══════════════════════════════════════════════════════════════════


class TestOpKind:

    def test_all_values(self):
        assert OpKind.PUSH_TENSOR.value == "push_tensor"
        assert OpKind.PULL_TENSOR.value == "pull_tensor"
        assert OpKind.WRITE_REGISTER.value == "write_register"
        assert OpKind.READ_REGISTER.value == "read_register"
        assert OpKind.POLL_REGISTER.value == "poll_register"
        assert OpKind.CONFIGURE.value == "configure"
        assert OpKind.BARRIER.value == "barrier"
        assert len(OpKind) == 7

    def test_from_string(self):
        assert OpKind("push_tensor") == OpKind.PUSH_TENSOR
        assert OpKind("barrier") == OpKind.BARRIER


# ═══════════════════════════════════════════════════════════════════
# §2  Operation dataclass — NPU 3D tensor/register ops
# ═══════════════════════════════════════════════════════════════════


class TestOperation:

    def test_push_ifm_tensor(self):
        """NPU 3D: PUSH IFM tensor (host → DUT)."""
        ifm = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=ifm)
        assert op.kind == OpKind.PUSH_TENSOR
        assert op.tensor is ifm
        assert op.kernel is None
        assert op.dep == []
        assert op.commit_dep == []

    def test_push_bias_int32(self):
        """NPU 3D: PUSH bias int32 tensor."""
        bias = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=bias)
        assert op.tensor.dtype == torch.int32

    def test_defaults(self):
        op = Operation(kind=OpKind.BARRIER)
        assert op.tensor is None
        assert op.kernel is None
        assert op.register_interface is None
        assert op.register_fields is None
        assert op.register_field_name is None
        assert op.probe is False
        assert op.sync is False

    def test_write_register_vsync(self):
        """NPU 3D: WRITE_REG vsync trigger = 1."""
        op = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl",
            register_fields={"vsync": 1},
        )
        assert op.kind == OpKind.WRITE_REGISTER
        assert op.register_interface == "ctrl"
        assert op.register_fields == {"vsync": 1}

    def test_poll_register_layer_done(self):
        """NPU 3D: POLL_REG layer_done (fmapIO 0x054)."""
        op = Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl",
            register_field_name="layer_done",
        )
        assert op.register_field_name == "layer_done"

    def test_configure_op(self):
        """NPU 3D: CONFIGURE auto_bind all 6 IP registers."""
        k = FmapIOStub()
        op = Operation(kind=OpKind.CONFIGURE, kernel=k)
        assert op.kind == OpKind.CONFIGURE
        assert op.kernel is k

    def test_op_with_deps(self):
        """NPU 3D: pull_tensor depends on push completion."""
        push = Operation(kind=OpKind.PUSH_TENSOR)
        h_push = OperationHandle(op=push)
        pull = Operation(kind=OpKind.PULL_TENSOR, dep=[h_push])
        assert len(pull.dep) == 1
        assert pull.dep[0] is h_push


# ═══════════════════════════════════════════════════════════════════
# §3  OperationHandle — commit dependency
# ═══════════════════════════════════════════════════════════════════


class TestOperationHandle:

    def test_wraps_operation(self):
        op = Operation(kind=OpKind.PUSH_TENSOR)
        handle = OperationHandle(op=op)
        assert handle.op is op

    def test_add_commit_dependency(self):
        """NPU 3D: pull commit depends on poll_register."""
        pull = OperationHandle(op=Operation(kind=OpKind.PULL_TENSOR))
        poll = OperationHandle(op=Operation(kind=OpKind.POLL_REGISTER))
        pull.add_commit_dependency(poll)
        assert len(pull.op.commit_dep) == 1
        assert pull.op.commit_dep[0] is poll

    def test_multiple_commit_deps(self):
        """NPU 3D: OFM store commit depends on all 3 IP poll_registers."""
        pull = OperationHandle(op=Operation(kind=OpKind.PULL_TENSOR))
        polls = []
        for _ in range(3):
            poll = OperationHandle(op=Operation(kind=OpKind.POLL_REGISTER))
            pull.add_commit_dependency(poll)
            polls.append(poll)
        assert len(pull.op.commit_dep) == 3
        for i, p in enumerate(polls):
            assert pull.op.commit_dep[i] is p


# ═══════════════════════════════════════════════════════════════════
# §4  NPU 3D host workflow — full operation chain
# ═══════════════════════════════════════════════════════════════════


class TestNPU3DWorkflow:
    """NPU 3D host execution sequence as DSL operations."""

    def test_push_three_tensors(self):
        """Step 1: PUSH ifm, weight, bias (host → DUT)."""
        push_ifm = Operation(kind=OpKind.PUSH_TENSOR)
        push_wgt = Operation(kind=OpKind.PUSH_TENSOR)
        push_bias = Operation(kind=OpKind.PUSH_TENSOR)
        h_ifm = OperationHandle(op=push_ifm)
        h_wgt = OperationHandle(op=push_wgt)
        h_bias = OperationHandle(op=push_bias)
        assert all(
            op.kind == OpKind.PUSH_TENSOR
            for op in [h_ifm.op, h_wgt.op, h_bias.op]
        )

    def test_configure_depends_on_pushes(self):
        """Step 2: CONFIGURE depends on all 3 PUSHes."""
        h_ifm = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))
        h_wgt = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))
        h_bias = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))

        k = FmapIOStub()
        configure = Operation(
            kind=OpKind.CONFIGURE, kernel=k,
            dep=[h_ifm, h_wgt, h_bias],
        )
        assert len(configure.dep) == 3
        assert configure.kernel is k

    def test_vsync_trigger_order(self):
        """Step 3: VSYNC order — bias_loader → weight_loader → fmapIO.

        fmapIO vsync depends on bias/weight vsyncs being issued first.
        """
        h_configure = OperationHandle(op=Operation(kind=OpKind.CONFIGURE))

        # bias_loader vsync
        vsync_bias = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_bias",
            register_fields={"vsync": 1},
            dep=[h_configure],
        )
        h_vsync_bias = OperationHandle(op=vsync_bias)

        # weight_loader vsync
        vsync_wgt = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_wgt",
            register_fields={"vsync": 1},
            dep=[h_configure],
        )
        h_vsync_wgt = OperationHandle(op=vsync_wgt)

        # fmapIO vsync — depends on bias/weight vsyncs
        vsync_fmap = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_fmapio",
            register_fields={"vsync": 1},
            dep=[h_vsync_bias, h_vsync_wgt],
        )
        assert len(vsync_fmap.dep) == 2
        assert vsync_fmap.dep[0].op.register_interface == "ctrl_bias"
        assert vsync_fmap.dep[1].op.register_interface == "ctrl_wgt"

    def test_poll_layer_done(self):
        """Step 4: POLL_REG layer_done on fmapIO ctrl."""
        h_vsync = OperationHandle(op=Operation(kind=OpKind.WRITE_REGISTER))

        poll = Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl_fmapio",
            register_field_name="layer_done",
            dep=[h_vsync],
        )
        assert poll.register_field_name == "layer_done"
        assert poll.register_interface == "ctrl_fmapio"
        assert len(poll.dep) == 1

    def test_pull_ofm(self):
        """Step 5: PULL OFM from DUT."""
        ofm = Tensor(
            shape=("${OUT_CH}", "${OUT_DEPTH}", "${OUT_HEIGHT}", "${OUT_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        h_push = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))

        pull = Operation(
            kind=OpKind.PULL_TENSOR,
            tensor=ofm,
            dep=[h_push],
        )
        assert pull.kind == OpKind.PULL_TENSOR
        assert pull.tensor is ofm

    def test_full_npu3d_chain(self):
        """Full NPU 3D workflow: push×3 → configure → vsync×3 → poll → pull.

        Mirrors a representative multi-IP accelerator host code sequence.
        """
        # Step 1: Push tensors
        h_push_ifm = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))
        h_push_wgt = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))
        h_push_bias = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))

        # Step 2: Configure (auto_bind all registers)
        configure = Operation(
            kind=OpKind.CONFIGURE,
            dep=[h_push_ifm, h_push_wgt, h_push_bias],
        )
        h_configure = OperationHandle(op=configure)

        # Step 3: VSYNC triggers (bias → weight → fmapIO)
        h_vsync_bias = OperationHandle(op=Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_bias",
            register_fields={"vsync": 1},
            dep=[h_configure],
        ))
        h_vsync_wgt = OperationHandle(op=Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_wgt",
            register_fields={"vsync": 1},
            dep=[h_configure],
        ))
        h_vsync_fmap = OperationHandle(op=Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_fmapio",
            register_fields={"vsync": 1},
            dep=[h_vsync_bias, h_vsync_wgt],
        ))

        # Step 3.5: Pull output tensors
        h_pull_ofm = OperationHandle(op=Operation(
            kind=OpKind.PULL_TENSOR, dep=[h_vsync_fmap],
        ))

        # Step 4: Poll layer_done
        h_poll = OperationHandle(op=Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl_fmapio",
            register_field_name="layer_done",
            dep=[h_vsync_fmap],
        ))

        # Commit dependency: pull commit waits for poll
        h_pull_ofm.add_commit_dependency(h_poll)

        # Verify chain structure
        assert len(configure.dep) == 3  # 3 pushes
        assert len(h_vsync_fmap.op.dep) == 2  # bias + wgt vsyncs
        assert len(h_pull_ofm.op.dep) == 1  # vsync_fmap
        assert len(h_pull_ofm.op.commit_dep) == 1  # poll
        assert h_pull_ofm.op.kind == OpKind.PULL_TENSOR


# ═══════════════════════════════════════════════════════════════════
# §5  Memory-mapped push/pull pattern (AXI4 master)
# ═══════════════════════════════════════════════════════════════════


class TestMemoryMappedPattern:
    """NPU 3D: DUT is AXI4 master, BFM is slave responding to DUT reads/writes."""

    def test_push_tensor_slave_response(self):
        """push_tensor: BFM slave responds to DUT read request (IFM from DDR)."""
        ifm = Tensor(
            shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=ifm)
        assert op.kind == OpKind.PUSH_TENSOR
        assert op.tensor is ifm

    def test_pull_tensor_slave_capture(self):
        """pull_tensor: BFM slave captures DUT write (OFM to DDR)."""
        ofm = Tensor(
            shape=("${OUT_CH}", "${OUT_DEPTH}", "${OUT_HEIGHT}", "${OUT_WIDTH}"),
            dtype=torch.int8, interface="ddr",
        )
        op = Operation(kind=OpKind.PULL_TENSOR, tensor=ofm)
        assert op.kind == OpKind.PULL_TENSOR

    def test_weight_push_hbm_32bank(self):
        """weight push: HBM 32-bank split → single push_tensor in DSL."""
        weight = Tensor(
            shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
            dtype=torch.int8, interface="hbm",
        )
        # User writes single push_tensor; runtime splits into 32 PUSH commands
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=weight)
        assert op.kind == OpKind.PUSH_TENSOR
        assert op.tensor.interface == "hbm"


# ═══════════════════════════════════════════════════════════════════
# §6  Dependency patterns — NPU 3D specifics
# ═══════════════════════════════════════════════════════════════════


class TestDependencyPatterns:
    """NPU 3D dependency chain patterns."""

    def test_fan_in_three_pushes(self):
        """3 PUSH ops (ifm, weight, bias) fan into configure."""
        pushes = [
            OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))
            for _ in range(3)
        ]
        configure = Operation(kind=OpKind.CONFIGURE, dep=pushes)
        assert len(configure.dep) == 3

    def test_fan_in_two_vsyncs(self):
        """fmapIO vsync depends on bias_loader + weight_loader vsyncs."""
        h_bias = OperationHandle(op=Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_bias",
        ))
        h_wgt = OperationHandle(op=Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_wgt",
        ))
        fmap_vsync = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl_fmapio",
            dep=[h_bias, h_wgt],
        )
        assert len(fmap_vsync.dep) == 2

    def test_commit_dependency_poll_to_pull(self):
        """pull OFM commit waits for poll layer_done."""
        push = OperationHandle(op=Operation(kind=OpKind.PUSH_TENSOR))
        pull = OperationHandle(op=Operation(kind=OpKind.PULL_TENSOR, dep=[push]))
        poll = OperationHandle(op=Operation(
            kind=OpKind.POLL_REGISTER,
            register_field_name="layer_done",
        ))
        pull.add_commit_dependency(poll)
        assert pull.op.commit_dep[0] is poll
        assert pull.op.commit_dep[0].op.register_field_name == "layer_done"

    def test_barrier_between_layers(self):
        """Barrier separates consecutive NPU 3D layer executions."""
        barrier = Operation(kind=OpKind.BARRIER)
        h_barrier = OperationHandle(op=barrier)
        assert barrier.kind == OpKind.BARRIER
        assert barrier.dep == []
        # Next layer's push depends on barrier
        next_push = Operation(kind=OpKind.PUSH_TENSOR, dep=[h_barrier])
        assert next_push.dep[0].op.kind == OpKind.BARRIER


# ═══════════════════════════════════════════════════════════════════
# §8  OpCode enum (Execution IR level — 00_data_models.md §1.4)
# ═══════════════════════════════════════════════════════════════════


class TestOpCode:

    def test_opcode_values(self):
        from vten.spec.models import OpCode

        assert OpCode.LOAD.value == 1
        assert OpCode.PUSH.value == 2
        assert OpCode.PULL.value == 3
        assert OpCode.STORE.value == 4
        assert OpCode.WRITE_REG.value == 5
        assert OpCode.READ_REG.value == 6
        assert OpCode.POLL_REG.value == 7
        assert OpCode.BARRIER.value == 8
        assert len(OpCode) == 8


# ═══════════════════════════════════════════════════════════════════
# §9  Direction / Role / MappingType / CommandStatus enums
# ═══════════════════════════════════════════════════════════════════


class TestDirectionRole:

    def test_direction_values(self):
        from vten.spec.models import Direction

        assert Direction.HOST_TO_DEV.value == "host_to_dev"
        assert Direction.DEV_TO_HOST.value == "dev_to_host"
        assert Direction.BIDIRECTIONAL.value == "bidirectional"

    def test_role_values(self):
        from vten.spec.models import Role

        assert Role.MASTER.value == "master"
        assert Role.SLAVE.value == "slave"


class TestMappingType:

    def test_mapping_type_values(self):
        """NPU 3D uses EXTERNAL (BFM) and INTERNAL (IP-to-IP wire)."""
        from vten.spec.models import MappingType

        assert MappingType.EXTERNAL.value == "external"
        assert MappingType.EXTERNAL_BANK.value == "external_bank"
        assert MappingType.INTERNAL.value == "internal"
        assert MappingType.INTERNAL_PROBE.value == "internal_probe"


class TestCommandStatus:

    def test_command_status_values(self):
        from vten.spec.models import CommandStatus

        assert CommandStatus.PENDING.value == 0
        assert CommandStatus.ISSUED.value == 1
        assert CommandStatus.ACTIVE.value == 2
        assert CommandStatus.COMMITTED.value == 3
        assert CommandStatus.ERROR.value == 4


# ═══════════════════════════════════════════════════════════════════
# §10  Error hierarchy (00_data_models.md §11)
# ═══════════════════════════════════════════════════════════════════


class TestErrorHierarchy:

    def test_base_error(self):
        from vten.errors import VTenError

        assert issubclass(VTenError, Exception)

    def test_spec_validation_error(self):
        from vten.errors import VTenError, SpecValidationError

        assert issubclass(SpecValidationError, VTenError)

    def test_binding_error(self):
        from vten.errors import VTenError, BindingError

        assert issubclass(BindingError, VTenError)

    def test_validation_error(self):
        from vten.errors import VTenError, ValidationError

        assert issubclass(ValidationError, VTenError)

    def test_raise_and_catch(self):
        from vten.errors import VTenError, SpecValidationError

        with pytest.raises(VTenError):
            raise SpecValidationError("test error")
