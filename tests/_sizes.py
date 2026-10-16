"""Representable cell sizes, for tests that draw random ones.

A size has at most 3 significant digits (see the cell_id module docstring), so `randint(1, max)`
mostly produces sizes `encode` rejects. This enumerates the valid ones directly, from the layout's
own constants rather than through `pack_size`, so it can check that function rather than echo it.
"""

import random

from beahiv import SizeMeasure
from beahiv.cell_id import SIZE_EXPONENT_MAX, SIZE_MANTISSA_MAX, SIZE_MIN

REPRESENTABLE_SIZES = sorted(
    {
        mantissa * 10**exponent
        for exponent in range(SIZE_EXPONENT_MAX + 1)
        for mantissa in range(1, SIZE_MANTISSA_MAX + 1)
    }
)


def sizes(measure: SizeMeasure = SizeMeasure.SIDE_LENGTH, upto: int | None = None) -> list[int]:
    return [s for s in REPRESENTABLE_SIZES if s >= SIZE_MIN[measure] and (upto is None or s <= upto)]


def random_size(rng: random.Random, measure: SizeMeasure = SizeMeasure.SIDE_LENGTH, upto: int | None = None) -> int:
    return rng.choice(sizes(measure, upto))
