"""
Vector type definitions used in the package.

The Vector is based on the :obj:`numpy.typing.NDArray` with a type hint.
"""
from typing import TypeVar, Tuple

import numpy
from typing_extensions import TypeAlias

T = TypeVar('T')  # pylint: disable=invalid-name

if numpy.__version__ >= "1.20":
    import numpy.typing

    Vector: TypeAlias = numpy.typing.NDArray[T]
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
