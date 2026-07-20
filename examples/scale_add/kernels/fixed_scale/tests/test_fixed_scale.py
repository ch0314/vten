from vten.cli.scenario import TestScenario


class TestFixedScale(TestScenario):
    """Q8.8 fixed-point multiply — four configs in ONE sim session.

    The input (seeded random plus pinned -128/127/-1/1/0) covers the code
    extremes and the exact-tie codes, so each config exercises a distinct
    corner of the declared quant semantics:

    - default:  coeff=384 (1.5) — rounding and saturation mixed.
    - replay:   same coeff, run again after done has latched — exercises the
                S_DONE → S_RUN re-arm path (start is a pulse register).
    - ties:     coeff=128 (0.5) — output is (x + 1) >> 1, an exact half-LSB
                tie at EVERY odd input code; half-up must round toward +inf
                for both signs (x=1 → 1, x=-1 → 0).
    - saturate: coeff=512 (2.0) — half the input range clamps to +-full-scale.
    """

    kernel = "fixed_scale"

    configs = [
        {"name": "default", "coeff": 384},
        {"name": "replay", "coeff": 384},
        {"name": "ties", "coeff": 128},
        {"name": "saturate", "coeff": 512},
    ]
