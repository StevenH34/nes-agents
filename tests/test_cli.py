"""Tests for the shared argparse types, called directly without a parser."""

import argparse

import pytest

from nes_gym.cli import float_at_least, fraction, non_negative_int, positive_int


@pytest.mark.parametrize(
    "parse, text, expected",
    [
        (positive_int, "1", 1),
        (positive_int, "30", 30),
        (non_negative_int, "0", 0),
        (non_negative_int, "5", 5),
        (float_at_least(0.1), "0.1", 0.1),
        (float_at_least(0.1), "15", 15.0),
        (float_at_least(0.1), "1e6", 1e6),
        (fraction, "0.1", 0.1),
        (fraction, "0.999", 0.999),
    ],
)
def test_accepts_valid_values(parse, text, expected):
    """Valid values, including the exact minimum, parse to the right number.

    Args:
        parse: The argparse type.
        text: The command-line text.
        expected: The parsed value.
    """
    assert parse(text) == expected


@pytest.mark.parametrize(
    "parse, text, message",
    [
        (positive_int, "0", "at least 1"),
        (positive_int, "-3", "at least 1"),
        (non_negative_int, "-1", "at least 0"),
        (float_at_least(0.1), "0.0999", "at least 0.1"),
        (float_at_least(0.1), "1e-310", "at least 0.1"),
        (float_at_least(0.1), "-inf", "finite"),
        (float_at_least(0.1), "inf", "finite"),
        (float_at_least(0.1), "nan", "finite"),
        (fraction, "0", "strictly between 0 and 1"),
        (fraction, "1", "strictly between 0 and 1"),
        (fraction, "inf", "strictly between 0 and 1"),
        (fraction, "nan", "strictly between 0 and 1"),
    ],
)
def test_rejects_out_of_range_values(parse, text, message):
    """Out-of-range, infinite and NaN values raise ArgumentTypeError, which argparse shows as a usage error.

    Args:
        parse: The argparse type.
        text: The command-line text.
        message: Text the error must contain.
    """
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        parse(text)


@pytest.mark.parametrize(
    "parse, text",
    [(positive_int, "abc"), (positive_int, "2.5"), (non_negative_int, ""), (float_at_least(0.1), "abc"),
     (fraction, "half")],
)
def test_rejects_non_numbers(parse, text):
    """Non-numbers raise ValueError, which argparse reports as "invalid <type name> value".

    Args:
        parse: The argparse type.
        text: The command-line text.
    """
    with pytest.raises(ValueError):
        parse(text)


@pytest.mark.parametrize(
    "parse, name",
    [(positive_int, "int"), (non_negative_int, "int"), (float_at_least(0.1), "float"), (fraction, "fraction")],
)
def test_type_names(parse, name):
    """Each type is named for what the user should type, as argparse shows it in errors.

    Args:
        parse: The argparse type.
        name: The expected name.
    """
    assert parse.__name__ == name


def test_parser_error_names_the_type(capsys):
    """Through a real parser, a non-number gives "invalid float value"."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float_at_least(0.1))
    with pytest.raises(SystemExit):
        parser.parse_args(["--fps", "abc"])
    assert "argument --fps: invalid float value: 'abc'" in capsys.readouterr().err
