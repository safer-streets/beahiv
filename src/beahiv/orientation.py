from enum import IntEnum


class Orientation(IntEnum):
    """Which pair of a hexagon's sides is vertical.

    POINTY cells have a vertex pointing up/down and vertical left/right
    edges. FLAT cells are POINTY rotated 30 degrees: a vertex points
    left/right and the top/bottom edges are horizontal.

    That rotation is about the origin, so a given (q, r) indexes a
    different physical cell under each orientation (they only agree at
    (0, 0)) -- see `coords.py`. q/r and cell ids are not portable across
    orientations.
    """

    POINTY = 0
    FLAT = 1
