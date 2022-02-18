"""
Vector type definitions used in the package.

The Vector is based on the :obj:`numpy.typing.NDArray` with a type hint.
"""
from typing import TypeVar

import numpy.typing
from typing_extensions import TypeAlias

T = TypeVar('T')  # pylint: disable=invalid-name

Vector: TypeAlias = numpy.typing.NDArray[T]
StringVector: TypeAlias = numpy.typing.NDArray[str]
FloatVector: TypeAlias = numpy.typing.NDArray[float]
