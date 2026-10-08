"""On-demand cell geometry. Nothing here is stored -- vertices are always
regenerated from (q, r, side_length, orientation)."""

import math
from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike
from shapely import Point, Polygon

from .batch import cell_centre_batch, cell_to_lonlat_batch
from .cell_id import decode
from .coords import axial_to_cartesian
from .geo import _cell_centre
from .orientation import Orientation

_VERTEX_ANGLES_DEG = {
    Orientation.POINTY: (30, 90, 150, 210, 270, 330),
    Orientation.FLAT: (0, 60, 120, 180, 240, 300),
}


def cell_polygon(cell_id: SupportsIndex) -> Polygon:
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


def cell_polygons(cell_ids: ArrayLike) -> list[Polygon]:
    """Vectorised version of the above: one `Polygon` for every cell in `cell_ids`.

    Takes anything `batch.cell_centre_batch` does -- a list, a numpy array, a pandas Series --
    since that's what the centre lookup here forwards to.

    Every cell must share the same side_length and orientation -- the same restriction
    `batch.cell_centre_batch` already applies to the vectorised centre lookup this reuses,
    since a single angle set / radius only applies to one orientation and side_length at a time.
    """
    ids = np.asarray(cell_ids)
    if ids.size == 0:
        return []
    xc, yc = cell_centre_batch(ids)
    idx = decode(ids[0])
    theta = np.radians(_VERTEX_ANGLES_DEG[idx.orientation])
    vx = xc[:, None] + idx.side_length * np.cos(theta)
    vy = yc[:, None] + idx.side_length * np.sin(theta)
    return [Polygon(zip(row_x, row_y, strict=True)) for row_x, row_y in zip(vx.tolist(), vy.tolist(), strict=True)]


def centroid(cell_id: SupportsIndex, lonlat: bool = False) -> Point:
    """Return a cell's centre as a Shapely `Point`, in EPSG:27700 metres.

    With `lonlat=True` the Point is in WGS84, x/y ordered as `Point(lon, lat)` -- the
    shapely/GeoJSON convention, so it drops straight into a GeoSeries with `crs=4326`. A Point
    carries no CRS of its own, so that ordering is the only thing telling a consumer which axis
    is which; note it is the *opposite* of the `(lat, lon)` tuple this used to return.

    Scalar only -- `centroids` is the vectorised form, mirroring `cell_polygon`/`cell_polygons`.
    """
    # not isinstance(cell_id, SupportsIndex): ndarray defines __index__ (and so satisfies that
    # protocol) but only honours it at size 1, which would let some arrays through and fail the
    # rest deep inside numpy. Rank is the actual question, and it covers Series/list/tuple too.
    if isinstance(cell_id, (list, tuple)) or getattr(cell_id, "ndim", 0) != 0:
        raise TypeError(f"centroid takes a single cell id, not {type(cell_id).__name__} -- use centroids for many")
    return Point(*_cell_centre(cell_id, lonlat))


def centroids(cell_ids: ArrayLike, lonlat: bool = False) -> list[Point]:
    """Vectorised version of the above: one `Point` for every cell in `cell_ids`.

    Takes anything `batch.cell_centre_batch` does, and applies the same restriction -- every cell
    must share one side_length and orientation. Returns `Point`s for symmetry with `cell_polygons`;
    callers wanting plain coordinate columns (`gdf["x"], gdf["y"] = ...`) should use
    `batch.cell_centre_batch` / `batch.cell_to_lonlat_batch` directly, which is what this wraps.
    """
    ids = np.asarray(cell_ids)
    if ids.ndim == 0:
        # the mirror of centroid's array guard -- and the likelier mistake of the two for anyone
        # moving off the old centroid(array). Without this it reaches `.tolist()` on a 0-d result
        # and dies on "'float' object is not iterable", pointing at nothing.
        raise TypeError(
            f"centroids takes many cell ids, not a single {type(cell_ids).__name__} -- use centroid for one"
        )
    if ids.size == 0:
        return []
    xs, ys = cell_to_lonlat_batch(ids) if lonlat else cell_centre_batch(ids)
    return [Point(x, y) for x, y in zip(xs.tolist(), ys.tolist(), strict=True)]
