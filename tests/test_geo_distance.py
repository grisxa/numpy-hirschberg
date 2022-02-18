"""
Test for the geo_distance() function.
"""
import numpy as np

from tests.distance import geo_distance


def test_geo_distance():
    """
    Test for proper geo distance calculation between a reference point and a track.
    """
    # given
    track = np.array(
        [
            (50, 20),
            (60, 20),
            (60.01, 20.01),
            (60, 30),
            (0, 0),
        ]
    )
    point = (60, 20)
    expected = [1111949.266, 0., 1243.159, 555445.133, 6891381.116]

    # when
    distance = geo_distance(point, track)

    # then
    assert np.around(distance, 3).tolist() == expected
