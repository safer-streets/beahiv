from enum import IntEnum


class Measure(IntEnum):
    """What a cell id's stored size measures.

    EDGE_LENGTH is the length of one edge (the hexagon's circumradius). EDGE_TO_EDGE is the distance
    between two opposite, parallel edges -- sqrt(3) times the side length. Either way the size is a
    whole number of metres; it is the *geometry* that differs, so an EDGE_TO_EDGE cell's side length
    is generally irrational.

    Like orientation, this is part of what a cell id means: a size-200 EDGE_LENGTH grid and a
    size-200 EDGE_TO_EDGE grid are different lattices, and their q/r are not interchangeable.
    """

    EDGE_LENGTH = 0
    EDGE_TO_EDGE = 1
