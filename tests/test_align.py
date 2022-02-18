"""
Test for the main align() function.
"""
from typing import Tuple

import numpy as np
import pytest

from numpy_hirschberg.align import align
from numpy_hirschberg.types import StringVector
from tests.distance import symbol_distance, match_distance


def test_align_empty():
    """
    Test for alignment of two empty strings. Returns empty at zero price.
    """
    # given
    source_vector: StringVector = np.array([])
    target_vector: StringVector = np.array([])

    # when
    first, second, distance = align(source_vector, target_vector, cost_function=None)

    # then
    assert np.array_equal(first, np.empty(shape=0))
    assert np.array_equal(second, np.empty(shape=0))
    assert distance == 0


def test_align_deletion():
    """
    Test for full source string deletion. Returns a sum of each symbol deletion.
    """
    # given
    source_vector: StringVector = np.array(list("ABC"))
    target_vector: StringVector = np.array([])

    # when
    first, second, distance = align(
        source_vector, target_vector, deletion_cost=-2, cost_function=symbol_distance
    )

    # then
    assert np.array_equal(first, np.array(list("ABC")))
    assert np.array_equal(second, np.array([None, None, None]))
    assert distance == -6


def test_align_insertion():
    """
    Test for full target string insertion. Returns a sum of each symbol insertion.
    """
    # given
    source_vector: StringVector = np.array([])
    target_vector: StringVector = np.array(list("AB"))

    # when
    first, second, distance = align(
        source_vector, target_vector, insertion_cost=-2, cost_function=symbol_distance
    )

    # then
    assert np.array_equal(first, np.array([None, None]))
    assert np.array_equal(second, np.array(list("AB")))
    assert distance == -4


def test_align_single_target():
    """
    Test for one symbol removal. Returns a single deletion cost.
    """
    # given
    source_vector: StringVector = np.array(list("AB"))
    target_vector: StringVector = np.array(["A"])

    # when
    first, second, distance = align(
        source_vector, target_vector, deletion_cost=-2, cost_function=symbol_distance
    )

    # then
    assert np.array_equal(first, np.array(list("AB")))
    assert np.array_equal(second, np.array(["A", None]))
    assert distance == -2


def test_align_single_source():
    """
    Test for one symbol insertion. Returns a single insertion cost.
    """
    # given
    source_vector: StringVector = np.array(["B"])
    target_vector: StringVector = np.array(list("AB"))

    # when
    first, second, distance = align(
        source_vector, target_vector, insertion_cost=-2, cost_function=symbol_distance
    )

    # then
    assert np.array_equal(first, np.array([None, "B"]))
    assert np.array_equal(second, np.array(list("AB")))
    assert distance == -2


@pytest.mark.parametrize(
    ("source", "target", "alignments"),
    [
        ("CG", "TG", (list("CG"), list("TG"), 1)),
        ("C", "CA", (["C", None], list("CA"), 0)),
        ("CA", "C", (list("CA"), ["C", None], 0)),
        ("CGCA", "TGC", (list("CGCA"), ["T", "G", "C", None], 1)),
        ("TA", "TA", (list("TA"), list("TA"), 4)),
        ("AG", "", (list("AG"), [None, None], -4)),
        ("AGTA", "TA", (list("AGTA"), [None, None, "T", "A"], 0)),
        (
            "AGTACGCA",
            "TATGC",
            (list("AGTACGCA"), [None, None, "T", "A", "T", "G", "C", None], 1),
        ),
        (
            "GAAAAAAT",
            "GAAT",
            (list("GAAAAAAT"), ["G", None, None, None, "A", None, "A", "T"], 0),
        ),
    ],
)
def test_align(source: str, target: str, alignments: Tuple[list, list, int]):
    """
    Series of tests with well known results. See detailed explanation in the `Wikipedia article`_.

    :param source: one string
    :param target: another string
    :param alignments: expected results as a tuple of padded source and target
        and a total transformation cost

    .. _Wikipedia article:
        https://en.wikipedia.org/wiki/Hirschberg's_algorithm
    """
    # given
    source_vector: StringVector = np.array(list(source))
    target_vector: StringVector = np.array(list(target))

    # when
    first, second, distance = align(
        source_vector,
        target_vector,
        deletion_cost=-2,
        insertion_cost=-2,
        cost_function=match_distance,
    )

    # then
    assert np.array_equal(first, np.array(alignments[0]))
    assert np.array_equal(second, np.array(alignments[1]))
    assert distance == alignments[2]
