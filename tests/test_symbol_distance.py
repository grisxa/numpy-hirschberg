"""
Test for the symbol_distance() function.
"""
import numpy as np
import pytest

from tests.distance import symbol_distance
from numpy_hirschberg.types import StringVector


@pytest.mark.parametrize(
    ("source", "target", "result"),
    [
        ("a", "a", [0]),
        ("a", "b", [1]),
        ("c", "a", [2]),
        ("A", "a", [32]),
        ("A", "abc", [32, 33, 34]),
    ],
)
def test_symbol_distance(source: str, target: str, result: int):
    """
    Control string distance evaluation as a difference of the symbol code.

    :param source: one string
    :param target: another string
    :param result: expected result: a number or a list of numbers for longer strings
    """
    # given
    target_vector: StringVector = np.array(list(target))

    # then
    assert symbol_distance(source, target_vector).tolist() == result
