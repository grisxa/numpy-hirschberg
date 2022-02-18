"""
Test for the match_distance() function.
"""
import numpy as np
import pytest

from tests.distance import match_distance
from numpy_hirschberg.types import StringVector


@pytest.mark.parametrize(
    ("source", "target", "result"),
    [
        ("a", "a", [-2]),
        ("a", "b", [1]),
        ("c", "a", [1]),
        ("A", "a", [1]),
        ("a", "abc", [-2, 1, 1]),
        ("bbb", "abc", [1, -2, 1]),
        ([1], [1], [-2]),
        ([1], [1, 2], [-2, 1]),
        ([1, 2, 3], [2, 2, 2], [1, -2, 1]),
    ],
)
def test_match_distance(source: str, target: str, result: int):
    """
    Control vector distance evaluation based on the item match.

    Match gives -2 otherwise 1 point.

    :param source: one vector
    :param target: another vector
    :param result: expected result: a number or a list of numbers
    """
    # given
    source_vector: StringVector = np.array(list(source))
    target_vector: StringVector = np.array(list(target))

    # then
    assert match_distance(source_vector, target_vector).tolist() == result
