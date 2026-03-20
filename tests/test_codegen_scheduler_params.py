"""Phase 4 tests: Scheduler parameter computation and BFM index mapping.

Spec references:
- 06_codegen_and_cli.md §3.3 (_generate_bfm_index_mapping)
- 06_codegen_and_cli.md §3.4 (_compute_scheduler_params)
- 04_backend_xsim.md §10.0 (Parameter auto-determination)
- specs/npu_3d_analysis.md §11.4 (NPU 3D Scheduler params)
"""

from __future__ import annotations

import pytest

from vten.runtime.ir import BFMConfig
from vten.spec.models import (
    InterfaceSpec,
    KernelSpec,
    Protocol,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_bfm_configs(count: int, protocol: Protocol = Protocol.AXI4S) -> list[BFMConfig]:
    """Create N BFMConfigs with sequential interface names."""
    return [
        BFMConfig(
            interface_name=f"iface_{i}",
            protocol=protocol,
            data_width=32 if protocol == Protocol.AXI4L else 256,
            role="master" if protocol == Protocol.AXI4L else "slave",
        )
        for i in range(count)
    ]


def _make_spec_with_interfaces(names: list[str]) -> KernelSpec:
    """Create KernelSpec with named interfaces (all AXI4-Stream)."""
    interfaces = {}
    for name in names:
        interfaces[name] = InterfaceSpec(
            name=name, rtl_port=f"port_{name}", protocol=Protocol.AXI4S,
        )
    return KernelSpec(
        kernel_name="test", rtl_top="test.sv", interfaces=interfaces,
    )


def _npu_40_bfm_configs() -> list[BFMConfig]:
    """NPU 3D: 40 BFMs (6 AXI4-Lite + 2 DDR AXI4 + 32 HBM AXI4)."""
    cfgs: list[BFMConfig] = []
    for name in ["ctrl_fmapio", "ctrl_wgt", "ctrl_mac", "ctrl_psum", "ctrl_bias", "ctrl_act"]:
        cfgs.append(BFMConfig(interface_name=name, protocol=Protocol.AXI4L, data_width=32, role="master"))
    for name in ["ddr_fmap", "ddr_bias"]:
        cfgs.append(BFMConfig(interface_name=name, protocol=Protocol.AXI4, data_width=256, role="slave"))
    for i in range(32):
        cfgs.append(BFMConfig(interface_name=f"hbm_{i:02d}", protocol=Protocol.AXI4, data_width=256, role="slave"))
    return cfgs


def _npu_spec_with_40_interfaces() -> KernelSpec:
    """KernelSpec with 40 interfaces matching NPU 3D BFM topology."""
    names = (
        ["ctrl_fmapio", "ctrl_wgt", "ctrl_mac", "ctrl_psum", "ctrl_bias", "ctrl_act"]
        + ["ddr_fmap", "ddr_bias"]
        + [f"hbm_{i:02d}" for i in range(32)]
    )
    interfaces = {}
    for name in names:
        proto = Protocol.AXI4L if name.startswith("ctrl") else Protocol.AXI4
        interfaces[name] = InterfaceSpec(
            name=name, rtl_port=f"port_{name}", protocol=proto,
            data_width=32 if proto == Protocol.AXI4L else 256,
        )
    return KernelSpec(
        kernel_name="npu_3d", rtl_top="NPU_3D_top.sv", interfaces=interfaces,
    )


def _minimal_config() -> dict:
    return {
        "project": {"name": "test", "version": "0.1.0"},
        "backend": {"xsim": {"vivado_path": "/v"}},
        "rtl": {"sources": [], "top_module": "test", "include_dirs": []},
    }


# ═══════════════════════════════════════════════════════════════════
# §1  _compute_scheduler_params — 06_codegen_and_cli.md §3.4
# ═══════════════════════════════════════════════════════════════════


class TestComputeSchedulerParams:
    """Scheduler parameter auto-calculation from BFMConfig[] and command count."""

    def test_defaults_small_design(self):
        """2 BFMs, 10 commands → minimums: max_bfms=8, max_ifaces=16, max_cmds=256."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _make_spec_with_interfaces(["in", "out"])
        gen = SVGenerator(
            kernel_spec=spec,
            bfm_configs=_make_bfm_configs(2),
            project_config=_minimal_config(),
        )
        params = gen._compute_scheduler_params(num_commands=10)
        assert params["max_bfms"] == 8
        assert params["max_ifaces"] == 16
        assert params["max_cmds"] == 256

    def test_large_bfm_count_npu(self):
        """40 BFMs (NPU 3D) → max_bfms=40."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_npu_spec_with_40_interfaces(),
            bfm_configs=_npu_40_bfm_configs(),
            project_config=_minimal_config(),
        )
        params = gen._compute_scheduler_params(num_commands=100)
        assert params["max_bfms"] == 40

    def test_npu_max_ifaces_at_least_42(self):
        """NPU 3D: max_ifaces >= 42 (40 interfaces + headroom from max(16, max_id+1))."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_npu_spec_with_40_interfaces(),
            bfm_configs=_npu_40_bfm_configs(),
            project_config=_minimal_config(),
        )
        params = gen._compute_scheduler_params(num_commands=100)
        # 40 interfaces → interface_ids 0..39 → max_ifaces = max(16, 40) = 40
        assert params["max_ifaces"] >= 40

    def test_large_command_count(self):
        """500 commands → max_cmds=500 (exceeds default 256)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_make_spec_with_interfaces(["in"]),
            bfm_configs=_make_bfm_configs(1),
            project_config=_minimal_config(),
        )
        params = gen._compute_scheduler_params(num_commands=500)
        assert params["max_cmds"] == 500

    def test_toml_override_larger(self):
        """vten.toml scheduler.max_bfms=48 when auto=40 → uses 48."""
        from vten.codegen.sv_generator import SVGenerator

        config = _minimal_config()
        config["backend"]["scheduler"] = {"max_bfms": 48}

        gen = SVGenerator(
            kernel_spec=_npu_spec_with_40_interfaces(),
            bfm_configs=_npu_40_bfm_configs(),
            project_config=config,
        )
        params = gen._compute_scheduler_params(num_commands=100)
        assert params["max_bfms"] == 48

    def test_toml_override_smaller_ignored(self):
        """vten.toml scheduler.max_bfms=4 when auto=8 → uses 8 (auto wins)."""
        from vten.codegen.sv_generator import SVGenerator

        config = _minimal_config()
        config["backend"]["scheduler"] = {"max_bfms": 4}

        gen = SVGenerator(
            kernel_spec=_make_spec_with_interfaces(["in"]),
            bfm_configs=_make_bfm_configs(1),
            project_config=config,
        )
        params = gen._compute_scheduler_params(num_commands=10)
        assert params["max_bfms"] == 8  # auto minimum, not 4

    def test_toml_override_partial(self):
        """Only max_cmds overridden, others auto-computed."""
        from vten.codegen.sv_generator import SVGenerator

        config = _minimal_config()
        config["backend"]["scheduler"] = {"max_cmds": 512}

        gen = SVGenerator(
            kernel_spec=_make_spec_with_interfaces(["in"]),
            bfm_configs=_make_bfm_configs(1),
            project_config=config,
        )
        params = gen._compute_scheduler_params(num_commands=100)
        assert params["max_cmds"] == 512
        assert params["max_bfms"] == 8  # auto default

    def test_empty_bfm_configs(self):
        """0 BFMs → max_bfms=8 (minimum), max_ifaces=16 (minimum)."""
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_make_spec_with_interfaces([]),
            bfm_configs=[],
            project_config=_minimal_config(),
        )
        params = gen._compute_scheduler_params(num_commands=0)
        assert params["max_bfms"] == 8
        assert params["max_ifaces"] == 16
        assert params["max_cmds"] == 256

    def test_interface_id_gap(self):
        """BFMs with non-contiguous interface_ids: max_ifaces adapts."""
        from vten.codegen.sv_generator import SVGenerator

        # Create spec with interfaces at indices that will produce gaps
        names = ["iface_0", "iface_5", "iface_10"]
        spec = _make_spec_with_interfaces(names)
        cfgs = [
            BFMConfig(interface_name=n, protocol=Protocol.AXI4S, data_width=32, role="master")
            for n in names
        ]

        gen = SVGenerator(
            kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config(),
        )
        params = gen._compute_scheduler_params(num_commands=5)
        # With 3 BFMs → max_bfms = 8 (minimum)
        assert params["max_bfms"] == 8

    def test_returns_dict_with_three_keys(self):
        from vten.codegen.sv_generator import SVGenerator

        gen = SVGenerator(
            kernel_spec=_make_spec_with_interfaces(["in"]),
            bfm_configs=_make_bfm_configs(1),
            project_config=_minimal_config(),
        )
        params = gen._compute_scheduler_params(num_commands=5)
        assert set(params.keys()) == {"max_bfms", "max_ifaces", "max_cmds"}


# ═══════════════════════════════════════════════════════════════════
# §2  _generate_bfm_index_mapping — 06_codegen_and_cli.md §3.3
# ═══════════════════════════════════════════════════════════════════


class TestBFMIndexMapping:
    """interface_id → BFM index mapping for Scheduler's iface_to_bfm[]."""

    def test_simple_1to1_mapping(self):
        """3 interfaces, 3 BFMs, all mapped sequentially."""
        from vten.codegen.sv_generator import SVGenerator

        names = ["a", "b", "c"]
        spec = _make_spec_with_interfaces(names)
        cfgs = [BFMConfig(interface_name=n, protocol=Protocol.AXI4S, data_width=32, role="master") for n in names]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        # interface_names() order defines interface_id
        # bfm_configs list order defines bfm_index
        # Both match: a→0, b→1, c→2
        assert mapping == {0: 0, 1: 1, 2: 2}

    def test_internal_interface_excluded(self):
        """Internal interfaces (no BFM) are not in the mapping."""
        from vten.codegen.sv_generator import SVGenerator

        # spec has 3 interfaces, but only 2 have BFMs
        spec = _make_spec_with_interfaces(["ext_a", "internal_x", "ext_b"])
        cfgs = [
            BFMConfig(interface_name="ext_a", protocol=Protocol.AXI4S, data_width=32, role="master"),
            BFMConfig(interface_name="ext_b", protocol=Protocol.AXI4S, data_width=32, role="slave"),
        ]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        # interface_id: ext_a=0, internal_x=1, ext_b=2
        # bfm_index: ext_a=0, ext_b=1
        assert 0 in mapping  # ext_a → bfm 0
        assert 1 not in mapping  # internal_x → no BFM
        assert 2 in mapping  # ext_b → bfm 1
        assert mapping[0] == 0
        assert mapping[2] == 1

    def test_mapping_order_matches_bfm_configs_order(self):
        """BFMConfig list order determines bfm_index."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _make_spec_with_interfaces(["x", "y", "z"])
        # BFM order: z, x, y (reversed from interface order)
        cfgs = [
            BFMConfig(interface_name="z", protocol=Protocol.AXI4S, data_width=32, role="slave"),
            BFMConfig(interface_name="x", protocol=Protocol.AXI4S, data_width=32, role="master"),
            BFMConfig(interface_name="y", protocol=Protocol.AXI4S, data_width=32, role="master"),
        ]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        # interface_id: x=0, y=1, z=2
        # bfm_index: z=0, x=1, y=2
        assert mapping[0] == 1  # x → bfm 1
        assert mapping[1] == 2  # y → bfm 2
        assert mapping[2] == 0  # z → bfm 0

    def test_mapping_order_matches_interface_names_order(self):
        """interface_names() order determines interface_id."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _make_spec_with_interfaces(["alpha", "beta"])
        cfgs = [
            BFMConfig(interface_name="alpha", protocol=Protocol.AXI4S, data_width=32, role="master"),
            BFMConfig(interface_name="beta", protocol=Protocol.AXI4S, data_width=32, role="slave"),
        ]

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        iface_names = spec.interface_names()
        # alpha is first in interface_names() → interface_id=0
        assert iface_names.index("alpha") == 0
        assert mapping[0] == 0  # alpha → bfm 0

    def test_npu_40bfm_mapping(self):
        """NPU 3D: 40 BFMs all mapped. No missing entries."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_40_interfaces()
        cfgs = _npu_40_bfm_configs()

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        # All 40 interfaces should be in mapping
        assert len(mapping) == 40

        # All bfm_indices should be unique and in range [0, 39]
        bfm_indices = set(mapping.values())
        assert bfm_indices == set(range(40))

    def test_npu_ctrl_fmapio_mapping(self):
        """NPU 3D: ctrl_fmapio interface maps to its BFM index."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_40_interfaces()
        cfgs = _npu_40_bfm_configs()

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        # Find interface_id for ctrl_fmapio
        iface_names = spec.interface_names()
        ctrl_fmapio_iface_id = iface_names.index("ctrl_fmapio")

        # Find bfm_index for ctrl_fmapio
        bfm_names = [c.interface_name for c in cfgs]
        ctrl_fmapio_bfm_idx = bfm_names.index("ctrl_fmapio")

        assert mapping[ctrl_fmapio_iface_id] == ctrl_fmapio_bfm_idx

    def test_npu_hbm_bank_mapping(self):
        """NPU 3D: 32 HBM banks all mapped to correct BFM indices."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_40_interfaces()
        cfgs = _npu_40_bfm_configs()

        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        iface_names = spec.interface_names()
        bfm_names = [c.interface_name for c in cfgs]

        for i in range(32):
            hbm_name = f"hbm_{i:02d}"
            iface_id = iface_names.index(hbm_name)
            bfm_idx = bfm_names.index(hbm_name)
            assert mapping[iface_id] == bfm_idx

    def test_empty_mapping(self):
        """No BFMs → empty mapping."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _make_spec_with_interfaces(["internal_only"])
        gen = SVGenerator(kernel_spec=spec, bfm_configs=[], project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()
        assert mapping == {}

    def test_returns_dict_int_int(self):
        """Mapping is dict[int, int]."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _make_spec_with_interfaces(["a"])
        cfgs = [BFMConfig(interface_name="a", protocol=Protocol.AXI4S, data_width=32, role="master")]
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        mapping = gen._generate_bfm_index_mapping()

        for k, v in mapping.items():
            assert isinstance(k, int)
            assert isinstance(v, int)
