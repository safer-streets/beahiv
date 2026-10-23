"""Fill a polygon with hex cells.

Deliberately separate from the core indexing modules (`geo.py`,
`geometry.py`, ...), which the project's design goals keep free of
point-in-polygon queries and spatial indexes -- see the README. This is
the one bulk operation that needs a full point-in-polygon query, done via
Shapely rather than anything homegrown.

Vectorised: each Shapely predicate runs over a whole array of candidate cells at once, rather than
once per cell from a Python loop, which was ~20us a cell.
"""

from typing import SupportsIndex

import numpy as np
import shapely
from shapely import box
from shapely.geometry.base import BaseGeometry

from .batch import axial_to_cartesian_batch, encode_batch
from .cell_id import decode, resolve_size
from .coords import cartesian_to_axial, side_length_of
from .geometry import _hexagons, cell_polygon
from .measure import Measure
from .orientation import Orientation

_PREDICATES = ("overlap", "centre", "center", "full")

# Candidate cells per chunk. Caps the memory a large polygon at a small size can take: a
# national-scale fill has hundreds of millions of candidates.
_CHUNK_CELLS = 1 << 18


def _axial_bounds(
    bounds: tuple[float, float, float, float],
    side_length: float,
    orientation: Orientation,
    pad: int = 2,
) -> tuple[int, int, int, int]:
    """Return (q_min, q_max, r_min, r_max) covering `bounds`.

    axial_to_cartesian is affine, and for both orientations one axial
    coordinate is a function of a single Cartesian axis while the other
    is a monotonic combination of both -- so the extremes of q and r
    over a Cartesian rectangle always occur at one of its four corners.
    `pad` absorbs the +/-1 cell rounding cartesian_to_axial applies at
    each corner.
    """
    minx, miny, maxx, maxy = bounds
    corners = ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))
    qs = []
    rs = []
    for x, y in corners:
        q, r = cartesian_to_axial(x, y, side_length, orientation)
        qs.append(q)
        rs.append(r)
    return min(qs) - pad, max(qs) + pad, min(rs) - pad, max(rs) + pad


def polyfill(
    polygon: BaseGeometry,
    *,
    size: int,
    measure: Measure = Measure.EDGE_LENGTH,
    orientation: Orientation = Orientation.FLAT,
    predicate: str = "overlap",
) -> list[int]:
    """Return the ids of every (size, orientation) hex cell covering `polygon`.

    `size` is required and `measure` defaults to EDGE_LENGTH, as for `lonlat_to_cell`.

    `polygon` is a Shapely `Polygon`/`MultiPolygon` in EPSG:27700 metres --
    beahiv's native CRS, so no reprojection happens here.

    `predicate` controls what counts as "covering" (naming matches the h3
    `contain` argument):
      - "overlap" (default): any part of the hex touches the polygon
      - "center": the hex's centre falls inside the polygon
      - "full": the hex lies entirely inside the polygon
    """
    size, measure = resolve_size(size, measure)
    if predicate not in _PREDICATES:
        raise ValueError(f"predicate must be one of {_PREDICATES}, got {predicate!r}")
    if polygon.is_empty:
        return []

    s = side_length_of(size, measure)
    minx, miny, maxx, maxy = polygon.bounds
    q_min, q_max, r_min, r_max = _axial_bounds(polygon.bounds, s, orientation)
    shapely.prepare(polygon)

    rs = np.arange(r_min, r_max + 1, dtype=np.int64)
    q_step = max(1, _CHUNK_CELLS // len(rs))
    chunks = []
    # q-major, r ascending within each q: ascending id order, as the cells come out
    for q_lo in range(q_min, q_max + 1, q_step):
        qs = np.arange(q_lo, min(q_lo + q_step, q_max + 1), dtype=np.int64)
        q, r = (a.ravel() for a in np.meshgrid(qs, rs, indexing="ij"))
        x, y = axial_to_cartesian_batch(q, r, s, orientation)
        # The q/r range is a parallelogram around the bounds, about twice their area. A cell whose
        # centre is more than its circumradius s outside them can't touch the polygon.
        near = (x >= minx - s) & (x <= maxx + s) & (y >= miny - s) & (y <= maxy + s)
        q, r, x, y = q[near], r[near], x[near], y[near]

        # A centre inside the polygon settles "overlap" (it's in both) and is necessary for "full"
        # (a hexagon inside the polygon has its centre in the interior), so hexagons are only built
        # for the cells it leaves open: those with a centre outside for one, inside for the other.
        hit = shapely.contains_xy(polygon, x, y)
        if predicate == "full":
            hit[hit] = shapely.contains(polygon, _hexagons(x[hit], y[hit], s, orientation))
        elif predicate == "overlap":
            hit[~hit] = shapely.intersects(polygon, _hexagons(x[~hit], y[~hit], s, orientation))
        chunks.append(encode_batch(q[hit], r[hit], size=size, measure=measure, orientation=orientation))
    return np.concatenate(chunks).tolist()


def bbox_fill(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    size: int,
    measure: Measure = Measure.EDGE_LENGTH,
    orientation: Orientation = Orientation.FLAT,
    predicate: str = "overlap",
) -> list[int]:
    """Return the ids of every (size, orientation) hex cell covering the axis-aligned
    bounding box (minx, miny, maxx, maxy), in EPSG:27700 metres.

    A thin convenience wrapper around `polyfill` for the common case of an axis-aligned box rather
    than an arbitrary polygon -- same `predicate` semantics, same validation, same edge cases
    (an empty/degenerate box, an invalid predicate), same size arguments.
    """
    return polyfill(
        box(minx, miny, maxx, maxy),
        size=size,
        measure=measure,
        orientation=orientation,
        predicate=predicate,
    )


def resize_cell(
    cell_id: SupportsIndex,
    *,
    size: int,
    measure: Measure | None = None,
    orientation: Orientation | None = None,
    predicate: str = "centre",
) -> list[int]:
    """Return the ids of every `size`-metre cell covering the hexagon `cell_id` spans.

    A thin convenience wrapper around `polyfill`, seeded by `cell_polygon(cell_id)` instead of an
    arbitrary polygon -- same `predicate` semantics. `size` may be smaller (a finer-grained
    covering, the useful replacement for a same-centroid "children" lookup) or larger (a coarser
    covering -- including the trivial case of finding the one coarser cell containing this one)
    than `cell_id`'s own size; both directions are ordinary `polyfill` calls, nothing here assumes
    one is bigger than the other.

    `measure` and `orientation` default to `cell_id`'s own -- the common case of resizing within
    the same grid family, where unlike everywhere else the measure is known rather than implied --
    but either can be overridden to cover the cell with a different family's grid instead. The
    covering is purely geometric, so any combination works.
    """
    idx = decode(cell_id)
    if measure is None:
        measure = idx.measure
    if orientation is None:
        orientation = idx.orientation
    return polyfill(
        cell_polygon(cell_id),
        size=size,
        measure=measure,
        orientation=orientation,
        predicate=predicate,
    )
