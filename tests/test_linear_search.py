"""
Test for the linear_search() function.
"""
from typing import List

import numpy as np
import pytest

from numpy_hirschberg.align import linear_search
from numpy_hirschberg.types import StringVector
from tests.distance import symbol_distance


@pytest.mark.parametrize(
    ("subject", "target", "alignment", "cost"),
    [
        ("A", "ABC", ["A", None, None], 0),
        ("B", "ABC", [None, "B", None], 0),
        ("D", "ABC", [None, None, "D"], 1),
        ("F", "ABC", [None, None, "F"], 3),
        ("C", "T", ["C"], 17),
        ("T", "C", ["T"], 17),
    ],
)
def test_linear_search(subject: str, target: str, alignment: List[str], cost: int):
    """
    Test for a proper subject placement in the target vector.

    :param subject: what to place
    :param target: where to place
    :param alignment: expected result
    :param cost: expected cost of the solution
    """
    # given
    target_vector: StringVector = np.array(list(target))

    # when
    line, distance = linear_search(subject, target_vector, symbol_distance)

    # then
    assert np.array_equal(line, alignment)
    assert distance == cost
