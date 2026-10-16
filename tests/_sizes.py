"""Representable cell sizes, for tests that draw random ones.

Above 1023m a size has 10 significant bits (see the cell_id module docstring), so `randint(1, max)`
mostly produces sizes `encode` rejects. This enumerates the valid ones directly, as "a 10-bit number
shifted left by at most 6", rather than through `pack_size`'s exponent/mantissa split -- so it
checks that function rather than echoing it.
"""

import random

from beahiv import SizeMeasure
from beahiv.cell_id import SIZE_MIN

REPRESENTABLE_SIZES = sorted({significand << shift for significand in range(1, 1024) for shift in range(7)})


def sizes(measure: SizeMeasure = SizeMeasure.SIDE_LENGTH, upto: int | None = None) -> list[int]:
    return [s for s in REPRESENTABLE_SIZES if s >= SIZE_MIN[measure] and (upto is None or s <= upto)]


def random_size(rng: random.Random, measure: SizeMeasure = SizeMeasure.SIDE_LENGTH, upto: int | None = None) -> int:
    return rng.choice(sizes(measure, upto))
