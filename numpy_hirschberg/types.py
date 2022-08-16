"""
Vector type definitions used in the package.

The Vector is based on the :obj:`numpy.typing.NDArray` with a type hint.
"""
from typing import TypeVar, Tuple, Callable

import numpy
from typing_extensions import TypeAlias

VectorItem = TypeVar("VectorItem")  # pylint: disable=invalid-name

if numpy.__version__ >= "1.20":
    import numpy.typing  # noqa: E0611, E0401

    Vector: TypeAlias = numpy.typing.NDArray[VectorItem]
    StringVector: TypeAlias = numpy.typing.NDArray[str]
    IntVector: TypeAlias = numpy.typing.NDArray[int]
    FloatVector: TypeAlias = numpy.typing.NDArray[float]
    GeoVector: TypeAlias = numpy.typing.NDArray[Tuple[float, float]]
else:
    Vector: TypeAlias = numpy.ndarray
    StringVector: TypeAlias = numpy.ndarray
    IntVector: TypeAlias = numpy.ndarray
    FloatVector: TypeAlias = numpy.ndarray
    GeoVector: TypeAlias = numpy.ndarray


CostFunction: TypeAlias = Callable[[VectorItem, Vector], Vector]
"""
Cost function of replacement each item of a vector with an element of the same type.
May be a predefined constant or arbitrary algorithm. Item's type may be complex - like
a pair of geo coordinates, in which case you likely to calculate a distance.

Returns a vector of costs.
"""
