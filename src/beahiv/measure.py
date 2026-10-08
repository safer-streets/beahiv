from enum import IntEnum


class SizeMeasure(IntEnum):
    """What a cell id's stored size measures.

    SIDE_LENGTH is the length of one edge (the hexagon's circumradius). SIDE_TO_SIDE is the distance
    between two opposite, parallel edges -- sqrt(3) times the side length. Either way the size is a
    whole number of metres; it is the *geometry* that differs, so a SIDE_TO_SIDE cell's side length
    is generally irrational.

    Like orientation, this is part of what a cell id means: a size-200 SIDE_LENGTH grid and a
    size-200 SIDE_TO_SIDE grid are different lattices, and their q/r are not interchangeable.
    """

    SIDE_LENGTH = 0
    SIDE_TO_SIDE = 1
