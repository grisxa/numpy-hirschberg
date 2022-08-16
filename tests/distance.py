"""
A couple of supplementary functions for distance measurement.
"""
from typing import Tuple

import numpy as np

from numpy_hirschberg.types import (
    StringVector,
    Vector,
    VectorItem,
    FloatVector,
    IntVector,
)

EARTH_RADIUS = 6371e3


def symbol_distance(
    a: str, b: StringVector
) -> IntVector:  # pylint: disable=invalid-name
    """
    Build a vector of distances from a source item list (a) to the target one (b).

    The distance is an absolute difference of each symbol code.

    :param a: the source element array
    :param b: the target element array
    :return: a vector of distances
    """
    return abs(b.view(np.int32) - ord(a))


def match_distance(
    a: VectorItem, b: Vector
) -> IntVector:  # pylint: disable=invalid-name
    """
    Build a vector of distances from a source item list (a) to the target one (b).

    The distance of each position is -2 if equal or 1 if not.

    :param a: the source element array
    :param b: the target element array
    :return: a vector of distances
    """
    return 1 - (a == b) * 3


def geo_distance(
    point: Tuple[float, float],
    track: FloatVector,
) -> FloatVector:
    """
    Build a vector of distances between a track and a reference point.

    :param point: the source point (latitude, longitude)
    :param track: a vector of track points
    :return: a vector of distances
    """
    point_latitude, point_longitude = np.radians(point)
    latitude_radians, longitude_radians = np.radians(track.T)

    return EARTH_RADIUS * np.arccos(
        np.sin(point_latitude) * np.sin(latitude_radians)
        + np.cos(point_latitude)
        * np.cos(latitude_radians)
        * np.cos(longitude_radians - point_longitude)
    )
