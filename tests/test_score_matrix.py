"""
Test for the score_matrix() function.
"""
from typing import List

import numpy as np
import pytest

from numpy_hirschberg.align import score_matrix
from numpy_hirschberg.types import StringVector
from tests.distance import match_distance


def test_line_score_empty():
    """
    Test for empty input: returns empty too.
    """
    # given
    source_vector: StringVector = np.array([])
    target_vector: StringVector = np.array([])

    # when
    line = score_matrix(source_vector, target_vector, cost_function=None)

    # then
    assert np.array_equal(line, [])


def test_line_score_source_empty():
    """
    Test if the source vector is empty, default insertion cost.
    """
    # given
    source_vector: StringVector = np.array([])
    target_vector: StringVector = np.array(list("AB"))

    # when
    line = score_matrix(source_vector, target_vector, cost_function=None)

    # then
    assert np.array_equal(line, [10, 20])


def test_line_score_target_empty():
    """
    Test if the target vector is empty, default deletion cost.
    """
    # given
    source_vector: StringVector = np.array(list("AB"))
    target_vector: StringVector = np.array(list([]))

    # when
    line = score_matrix(source_vector, target_vector, cost_function=None)

    # then
    assert np.array_equal(line, [100, 200])


@pytest.mark.parametrize(
    ("source", "target", "score"),
    [
        ("", "AB", [-2, -4]),
        ("AB", "", [-2, -4]),
        ("A", "A", [-2, 2]),
        ("A", "T", [-2, -1]),
        ("A", "TA", [-2, -1, 0]),
        ("AG", "T", [-4, -3]),
        ("AGTA", "TATGC", [-8, -4, 0, -2, -1, -3]),
        ("ACGC", "CGTAT", [-8, -4, 0, 1, -1, -3]),
    ],
)
def test_line_score(source: str, target: str, score: List[float]):
    """
    Test for obvious transformation, match_distance function.

    :param source: one string
    :param target: another string
    :param score: expected result
    """
    # given
    source_vector: StringVector = np.array(list(source))
    target_vector: StringVector = np.array(list(target))

    # when
    line = score_matrix(
        source_vector,
        target_vector,
        cost_function=match_distance,
        insertion_cost=-2,
        deletion_cost=-2,
    )

    # then
    assert np.array_equal(line, score)
