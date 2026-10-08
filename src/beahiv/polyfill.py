"""Fill a polygon with hex cells.

Deliberately separate from the core indexing modules (`geo.py`,
`geometry.py`, ...), which the project's design goals keep free of
point-in-polygon queries and spatial indexes -- see the README. This is
the one bulk operation that needs a full point-in-polygon query, done via
Shapely rather than anything homegrown.
"""

from typing import SupportsIndex

from shapely import Point, box, prepared
from shapely.geometry.base import BaseGeometry

from .cell_id import decode, encode_size, resolve_size
from .coords import axial_to_cartesian, cartesian_to_axial, side_length_of
from .geometry import cell_polygon
from .orientation import Orientation

_PREDICATES = ("overlap", "centre", "center", "full")


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
    side_length: int | None = None,
    side_to_side: int | None = None,
    orientation: Orientation = Orientation.FLAT,
    predicate: str = "overlap",
) -> list[int]:
    """Return the ids of every (size, orientation) hex cell covering `polygon`.

    Give exactly one of `side_length` / `side_to_side`, as for `latlon_to_cell`.

    `polygon` is a Shapely `Polygon`/`MultiPolygon` in EPSG:27700 metres --
    beahiv's native CRS, so no reprojection happens here.

    `predicate` controls what counts as "covering" (naming matches the h3
    `contain` argument):
      - "overlap" (default): any part of the hex touches the polygon
      - "center": the hex's centre falls inside the polygon
      - "full": the hex lies entirely inside the polygon
    """
    size, measure = resolve_size(side_length, side_to_side)
    if predicate not in _PREDICATES:
        raise ValueError(f"predicate must be one of {_PREDICATES}, got {predicate!r}")
    if polygon.is_empty:
        return []

    s = side_length_of(size, measure)
    q_min, q_max, r_min, r_max = _axial_bounds(polygon.bounds, s, orientation)
    prepared_polygon = prepared.prep(polygon)

    cells = []
    for q in range(q_min, q_max + 1):
        for r in range(r_min, r_max + 1):
            cell_id = encode_size(q, r, size, measure, orientation)
            if predicate in ("centre", "center"):
                hit = prepared_polygon.contains(Point(axial_to_cartesian(q, r, s, orientation)))
            elif predicate == "full":
                hit = prepared_polygon.contains(cell_polygon(cell_id))
            else:
                hit = prepared_polygon.intersects(cell_polygon(cell_id))
            if hit:
                cells.append(cell_id)
    return cells


def bbox_fill(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    side_length: int | None = None,
    side_to_side: int | None = None,
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
        side_length=side_length,
        orientation=orientation,
        predicate=predicate,
        side_to_side=side_to_side,
    )


def resize_cell(
    cell_id: SupportsIndex,
    *,
    new_side_length: int | None = None,
    new_side_to_side: int | None = None,
    orientation: Orientation | None = None,
    predicate: str = "centre",
) -> list[int]:
    """Return the ids of every new-size cell covering the hexagon `cell_id` spans.

    Give exactly one of `new_side_length` / `new_side_to_side`. Either may be used whatever
    `cell_id`'s own measure is -- the covering is purely geometric.

    A thin convenience wrapper around `polyfill`, seeded by `cell_polygon(cell_id)` instead of an
    arbitrary polygon -- same `predicate` semantics. `new_side_length` may be smaller (a
    finer-grained covering, the useful replacement for a same-centroid "children" lookup) or larger
    (a coarser covering -- including the trivial case of finding the one coarser cell containing
    this one) than `cell_id`'s own `side_length`; both directions are ordinary `polyfill` calls,
    nothing here assumes one is bigger than the other.

    `orientation` defaults to `cell_id`'s own orientation -- the common case of resizing within the
    same grid family -- but can be overridden to cover the cell with a grid of the other
    orientation instead.
    """
    idx = decode(cell_id)
    if orientation is None:
        orientation = idx.orientation
    return polyfill(
        cell_polygon(cell_id),
        side_length=new_side_length,
        orientation=orientation,
        predicate=predicate,
        side_to_side=new_side_to_side,
    )
