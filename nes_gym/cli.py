"""argparse types shared by the command-line scripts (bc_train.py, play.py, ...).

Each type rejects a bad value while parsing, so the error is a usage message naming the flag, given before any slow
work starts.
"""

import argparse
import math


def _int_at_least(minimum: int):
    """Returns an argparse type that parses an int and rejects values below `minimum`."""

    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    # argparse names the type in its "invalid <name> value" error for non-numbers.
    parse.__name__ = "int"
    return parse


positive_int = _int_at_least(1)
non_negative_int = _int_at_least(0)


def float_at_least(minimum: float):
    """Returns an argparse type that parses a finite float and rejects values below `minimum`.

    Use a minimum that keeps later arithmetic sane: e.g. --fps needs one well above 0, since 1 / 1e-310 is inf.
    """

    def parse(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"must be a finite number, got {value}")
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    parse.__name__ = "float"
    return parse


def fraction(text: str) -> float:
    """An argparse type: a float strictly between 0 and 1."""
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError(f"must be strictly between 0 and 1, got {value}")
    return value
