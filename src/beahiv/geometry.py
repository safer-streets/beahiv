"""On-demand cell geometry. Nothing here is stored -- vertices are always
regenerated from (q, r, side_length, orientation)."""

import math

from .cell_id import decode
from .coords import axial_to_cartesian
from .orientation import Orientation

_VERTEX_ANGLES_DEG = {
    Orientation.POINTY: (30, 90, 150, 210, 270, 330),
    Orientation.FLAT: (0, 60, 120, 180, 240, 300),
}


def cell_centre(cell_id: int) -> tuple[float, float]:
    """Return the (x, y) centre of a cell in EPSG:27700 metres."""
    idx = decode(cell_id)
    return axial_to_cartesian(idx.q, idx.r, idx.side_length, idx.orientation)


def cell_polygon(cell_id: int) -> list[tuple[float, float]]:
    """Return the six (x, y) vertices of a cell in EPSG:27700 metres."""
    idx = decode(cell_id)
    xc, yc = axial_to_cartesian(idx.q, idx.r, idx.side_length, idx.orientation)
    s = idx.side_length
    return [
        (xc + s * math.cos(math.radians(theta)), yc + s * math.sin(math.radians(theta)))
        for theta in _VERTEX_ANGLES_DEG[idx.orientation]
    ]
