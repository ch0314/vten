from vten.cli.scenario import TestScenario


class TestInt32X2(TestScenario):
    """64-bit bus: 2 int32 per beat."""

    kernel = "int32_x2"
