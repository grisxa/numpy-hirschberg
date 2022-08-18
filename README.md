# numpy-hirschberg
Hirschberg algorithm of sequence alignment using NumPy

## The sequence alignment
Sometimes you may want to find out how much one string differs from another.
The criteria can be a simple letter equality or a distance in the alphabet.
Insertions and deletions of the letter may also be considered as a weighted
difference.

Example:
```
fruit-
 |||
druids
```
Letter _f_ has been changed to _d_, _t_ to _d_ and _s_ inserted at the end.

In the biology sequence alignment algorithms help to find similar DNA/RNA parts
in longer chains.

```
GCAT-GCG
| ||  |
G-ATTACA
```

Besides the letters, sequence may consist of numbers of pairs of numbers, in which
case you have to select an appropriate comparison function. As an example, if you
compare two GPS tracks you can use the shortest distance between the coordinates.

## Implementation
There are many approaches with well known pros and cons. This package contains
a more space-efficient version of the
[Needleman-Wunsch algorithm](https://en.wikipedia.org/wiki/Needleman-Wunsch_algorithm)
invented by Dan Hirschberg.

See detailed explanation in the
[Wikipedia article](https://en.wikipedia.org/wiki/Hirschberg's_algorithm).

Another good visual analysis is in the
[blog post by Piotr Turski](http://blog.piotrturski.net/2015/04/hirschbergs-algorithm-explanation.html).

[NumPy](https://numpy.org/) is used as a vector framework.

## Usage
First, compose a cost function of your choice, then convert your sequences
to numpy arrays and call align. It returns modified sequences with insertions/deletions
and the cost of migration (or _the distance_)

```
import numpy as np
from numpy_hirschberg import align


def number_equality(src: int, dst: np.array) -> np.array:
    # results in 1 (equal) or -1 (not equal)
    return 1 - (src == dst) * 2


align(
    np.array([1, 2, 3, 4]),
    np.array([1, 3, 4, 5]),
    # deletion/insertion is cheaper than replacement
    deletion_cost=0,
    insertion_cost=0,
    cost_function=number_equality,
)
# ( [1, 2, 3, 4, None], [1, None, 3, 4, 5], -1.0 )
## 1 2 3 4 -
## |   | |
## 1 - 3 4 5

align(
    np.array([1, 2, 3, 4]),
    np.array([1, 3, 4, 5]),
    # deletion/insertion too pricey => prefer replacement
    deletion_cost=-3,
    insertion_cost=-3,
    cost_function=number_equality,
)
# ( [1, 2, 3, 4], [1, 3, 4, 5], -2.0 )
## 1 2 3 4
## |
## 1 3 4 5
```