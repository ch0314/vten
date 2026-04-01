from vten.cli.run import TestScenario


class TestScale(TestScenario):
    """Scale kernel standalone: multiply each byte by scale_factor=2."""

    kernel = "scale"
