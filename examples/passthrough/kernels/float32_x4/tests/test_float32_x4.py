from vten.cli.run import TestScenario


class TestFloat32X4(TestScenario):
    """128-bit bus: 4 float32 per beat."""

    kernel = "float32_x4"
