"""Tests for vten.testing — config_table and make_tests utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vten.testing import config_table, make_tests, _to_pascal_case


# ═══════════════════════════════════════════════════════════════════
# config_table
# ═══════════════════════════════════════════════════════════════════


class TestConfigTable:

    def test_basic_merge(self):
        result = config_table(
            defaults={"a": 1, "b": 2},
            configs=[{"name": "c1", "b": 3}],
        )
        assert len(result) == 1
        assert result[0] == {"name": "c1", "a": 1, "b": 3}

    def test_defaults_preserved(self):
        result = config_table(
            defaults={"a": 1, "b": 2},
            configs=[{"name": "c1"}],
        )
        assert result[0] == {"name": "c1", "a": 1, "b": 2}

    def test_override(self):
        result = config_table(
            defaults={"a": 1},
            configs=[{"name": "c1", "a": 99}],
        )
        assert result[0]["a"] == 99

    def test_multiple_configs(self):
        result = config_table(
            defaults={"x": 0},
            configs=[
                {"name": "first"},
                {"name": "second", "x": 1},
                {"name": "third", "x": 2},
            ],
        )
        assert len(result) == 3
        assert result[0]["x"] == 0
        assert result[1]["x"] == 1
        assert result[2]["x"] == 2

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="'name'"):
            config_table(defaults={}, configs=[{"a": 1}])

    def test_empty_configs(self):
        result = config_table(defaults={"a": 1}, configs=[])
        assert result == []

    def test_extra_keys_in_config(self):
        result = config_table(
            defaults={"a": 1},
            configs=[{"name": "c1", "b": 2}],
        )
        assert result[0] == {"name": "c1", "a": 1, "b": 2}


# ═══════════════════════════════════════════════════════════════════
# _to_pascal_case
# ═══════════════════════════════════════════════════════════════════


class TestToPascalCase:

    def test_snake_case(self):
        assert _to_pascal_case("forward_k3") == "ForwardK3"

    def test_single_word(self):
        assert _to_pascal_case("default") == "Default"

    def test_kebab_case(self):
        assert _to_pascal_case("ds-32x64") == "Ds32x64"

    def test_mixed(self):
        assert _to_pascal_case("fwd_k1_32x32") == "FwdK132x32"

    def test_with_numbers(self):
        assert _to_pascal_case("trans_320x320") == "Trans320x320"

    def test_already_capitalized(self):
        assert _to_pascal_case("Forward") == "Forward"


# ═══════════════════════════════════════════════════════════════════
# make_tests
# ═══════════════════════════════════════════════════════════════════


class TestMakeTests:

    def test_creates_classes(self):
        g = {"__name__": "test_module"}
        configs = [{"name": "forward_k3"}, {"name": "ds_32x64"}]
        classes = make_tests(
            configs=configs,
            kernel="my_kernel",
            run_fn=lambda ctx, cfg: None,
            module_globals=g,
        )
        assert len(classes) == 2
        assert "TestForwardK3" in g
        assert "TestDs32x64" in g

    def test_class_naming(self):
        g = {"__name__": "test_module"}
        configs = [{"name": "fwd_128x128"}]
        classes = make_tests(
            configs=configs,
            kernel="k",
            run_fn=lambda ctx, cfg: None,
            module_globals=g,
        )
        assert classes[0].__name__ == "TestFwd128x128"

    def test_custom_prefix(self):
        g = {"__name__": "test_module"}
        configs = [{"name": "basic"}]
        classes = make_tests(
            configs=configs,
            kernel="k",
            run_fn=lambda ctx, cfg: None,
            module_globals=g,
            class_prefix="Verify",
        )
        assert classes[0].__name__ == "VerifyBasic"
        assert "VerifyBasic" in g

    def test_run_calls_fn_with_merged_config(self):
        """run() merges runtime_cfg with frozen sweep config."""
        g = {"__name__": "test_module"}
        call_log = []

        def my_run(ctx, cfg):
            call_log.append(cfg)

        configs = [{"name": "c1", "a": 10}]
        classes = make_tests(
            configs=configs, kernel="k", run_fn=my_run, module_globals=g,
        )

        instance = classes[0]()
        ctx = MagicMock()
        instance.run(ctx, {"b": 20, "a": 1})

        assert len(call_log) == 1
        # Sweep config overrides runtime config
        assert call_log[0]["a"] == 10
        assert call_log[0]["b"] == 20

    def test_kernel_attribute(self):
        g = {"__name__": "test_module"}
        configs = [{"name": "c1"}]
        classes = make_tests(
            configs=configs, kernel="npu_pipeline",
            run_fn=lambda ctx, cfg: None, module_globals=g,
        )
        assert classes[0].kernel == "npu_pipeline"

    def test_closure_isolation(self):
        """Each generated class captures its own config (no late-binding bug)."""
        g = {"__name__": "test_module"}
        call_log = []

        def my_run(ctx, cfg):
            call_log.append(cfg["name"])

        configs = [
            {"name": "first", "val": 1},
            {"name": "second", "val": 2},
            {"name": "third", "val": 3},
        ]
        classes = make_tests(
            configs=configs, kernel="k", run_fn=my_run, module_globals=g,
        )

        ctx = MagicMock()
        for cls in classes:
            cls().run(ctx, {})

        assert call_log == ["first", "second", "third"]

    def test_is_test_scenario_subclass(self):
        from vten.cli.run import TestScenario

        g = {"__name__": "test_module"}
        configs = [{"name": "c1"}]
        classes = make_tests(
            configs=configs, kernel="k",
            run_fn=lambda ctx, cfg: None, module_globals=g,
        )
        assert issubclass(classes[0], TestScenario)
