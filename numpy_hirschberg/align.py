"""
Hirschberg's sequence alignment.

A more space-efficient version of the `Needleman-Wunsch algorithm`_
for the sequence alignment problem invented by Dan Hirschberg.

See detailed explanation in the `Wikipedia article`_.
Another good visual analysis is in the `blog post by Piotr Turski`_.

.. _Wikipedia article:
    https://en.wikipedia.org/wiki/Hirschberg's_algorithm

.. _blog post by Piotr Turski:
    http://blog.piotrturski.net/2015/04/hirschbergs-algorithm-explanation.html

.. _Needleman-Wunsch algorithm:
    https://en.wikipedia.org/wiki/Needleman-Wunsch_algorithm
"""

from typing import Tuple

import numpy as np
from numpy import add, full, concatenate, empty, fmax, flipud

from numpy_hirschberg.types import Vector, FloatVector, VectorItem, CostFunction


def align(  # pylint: disable=too-many-locals
    source: Vector,
    target: Vector,
    cost_function: CostFunction,
    deletion_cost: int = 100,
    insertion_cost: int = 0,
) -> Tuple[Vector, Vector, float]:
    """
    Divide and conquer approach to sequence alignment problem invented by Dan Hirschberg.

    See detailed explanation in the `Wikipedia article`_.
    Another good visual analysis is in the `blog post by Piotr Turski`_.

    The function returns the best possible solution as a tuple of the source and target vectors
    padded with None (insertion or deletion) and a total score of transformation.

    :param source: one vector
    :param target: another vector
    :param deletion_cost: fixed price for the source item deletion
    :param insertion_cost: fixed price for the target item insertion
    :param cost_function: dynamic replacement cost algorithm
    :return: a tuple of padded source and target vectors, and a total cost

    .. _Wikipedia article:
        https://en.wikipedia.org/wiki/Hirschberg's_algorithm

    .. _blog post by Piotr Turski:
        http://blog.piotrturski.net/2015/04/hirschbergs-algorithm-explanation.html
    """
    source_length, target_length = len(source), len(target)

    if source_length == 0 and target_length == 0:
        return np.empty(shape=0), np.empty(shape=0), 0

    if target_length == 0:
        return (
            source,
            full(source.shape, None),
            source_length * deletion_cost,
        )

    if source_length == 0:
        return (
            full(target.shape, None),
            target,
            target_length * insertion_cost,
        )

    if target_length == 1:
        indices, cost = linear_search(target[0], source, cost_function)
        return source, indices, deletion_cost * (source_length - 1) - cost

    if source_length == 1:
        indices, cost = linear_search(source[0], target, cost_function)
        return indices, target, insertion_cost * (target_length - 1) - cost

    cut_row: int = int(source_length / 2)
    upper_score: Vector = score_matrix(
        source[:cut_row], target, cost_function, deletion_cost, insertion_cost
    )
    lower_score: Vector = score_matrix(
        flipud(source[cut_row:]),
        flipud(target),
        cost_function,
        deletion_cost,
        insertion_cost,
    )

    max_index: int = int(np.argmax(upper_score + flipud(lower_score)))

    left_source, left_target, left_cost = align(
        source[:cut_row],
        target[:max_index],
        cost_function,
        deletion_cost,
        insertion_cost,
    )
    right_source, right_target, right_cost = align(
        source[cut_row:],
        target[max_index:],
        cost_function,
        deletion_cost,
        insertion_cost,
    )

    return (
        concatenate((left_source, right_source)),
        concatenate((left_target, right_target)),
        left_cost + right_cost,
    )


def score_matrix(  # pylint: disable=too-many-locals
    source: Vector,
    target: Vector,
    cost_function: CostFunction,
    deletion_cost: int = 100,
    insertion_cost: int = 10,
) -> Vector:
    """
    Build a [virtual] matrix of transformation scores for the given source and target vectors,
    and return the last line.

    Rules for the score matrix are described in the `Needleman-Wunsch algorithm`_

    :param source: one vector
    :param target: another vector
    :param deletion_cost: fixed price for the source item deletion
    :param insertion_cost: fixed price for the target item insertion
    :param cost_function: dynamic replacement cost algorithm
    :return: the last line of the score matrix

    .. _Needleman-Wunsch algorithm:
        https://en.wikipedia.org/wiki/Needleman-Wunsch_algorithm
    """
    source_length, target_length = len(source), len(target)

    if source_length == 0:
        return add.accumulate(full(target_length, insertion_cost))

    if target_length == 0:
        return add.accumulate(full(source_length, deletion_cost))

    full_deletion_column: Vector = add.accumulate(
        concatenate(([0], full(source_length, deletion_cost)))
    )
    full_insertion_row: Vector = add.accumulate(
        concatenate(([0], full(target_length, insertion_cost)))
    )

    row1: Vector = full_insertion_row
    row2: Vector = empty([target_length], dtype=np.int32)

    for i in range(source_length):
        replacement_score = row1[:-1] - cost_function(source[i], target)
        deletion_score = row1[1:] + deletion_cost
        replacement_deletion_score_max = fmax(replacement_score, deletion_score)

        for j in range(target_length):
            insertion_score = insertion_cost + (
                full_deletion_column[i + 1] if j == 0 else row2[j - 1]
            )
            row2[j] = max(replacement_deletion_score_max[j], insertion_score)
        row1 = concatenate(([full_deletion_column[i + 1]], row2))
    return row1


def linear_search(
    subject: VectorItem, target: Vector, cost_function: CostFunction
) -> Tuple[Vector, float]:
    """
    Finds the best position in the (target) vector for the subject.
    The cost function gives the minimum on that position.

    :param subject: what to place
    :param target: the vector to search in
    :param cost_function: arbitrary routine returning a vector of cost values
    :return: a vector of None with the only place taken by the subject, and the cost
    """
    line: Vector = full(target.shape, None)
    cost: FloatVector = cost_function(subject, target).astype(float)
    index: int = cost.argmin()
    line[index] = subject
    return line, cost[index]
