"""On-demand cell geometry. Nothing here is stored -- vertices are always
regenerated from (q, r, side_length, orientation)."""

import math
from collections.abc import Sequence

import numpy as np
from shapely import Polygon

from .batch import cell_centre_batch
from .cell_id import decode
from .coords import axial_to_cartesian
from .orientation import Orientation

_VERTEX_ANGLES_DEG = {
    Orientation.POINTY: (30, 90, 150, 210, 270, 330),
    Orientation.FLAT: (0, 60, 120, 180, 240, 300),
}


def cell_polygon(cell_id: int) -> Polygon:
    """Return a cell's six-vertex outline as a Shapely `Polygon`, in EPSG:27700 metres."""
    idx = decode(cell_id)
    xc, yc = axial_to_cartesian(idx.q, idx.r, idx.side_length, idx.orientation)
    s = idx.side_length
    return Polygon(
        [
            (xc + s * math.cos(math.radians(theta)), yc + s * math.sin(math.radians(theta)))
            for theta in _VERTEX_ANGLES_DEG[idx.orientation]
        ]
    )


def cell_polygons(cell_ids: Sequence[int]) -> list[Polygon]:
    """Vectorised version of the above: one `Polygon` for every cell in `cell_ids`.

    Every cell must share the same side_length and orientation -- the same restriction
    `batch.cell_centre_batch` already applies to the vectorised centre lookup this reuses,
    since a single angle set / radius only applies to one orientation and side_length at a time.
    """
    if len(cell_ids) == 0:
        return []
    xc, yc = cell_centre_batch(cell_ids)
    idx = decode(int(cell_ids[0]))
    theta = np.radians(_VERTEX_ANGLES_DEG[idx.orientation])
    vx = xc[:, None] + idx.side_length * np.cos(theta)
    vy = yc[:, None] + idx.side_length * np.sin(theta)
    return [Polygon(zip(row_x, row_y, strict=True)) for row_x, row_y in zip(vx.tolist(), vy.tolist(), strict=True)]
