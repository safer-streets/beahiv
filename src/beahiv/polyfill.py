"""Fill a polygon with hex cells.

Deliberately separate from the core indexing modules (`geo.py`,
`geometry.py`, ...), which the project's design goals keep free of
point-in-polygon queries and spatial indexes -- see the README. This is
the one bulk operation that needs a full point-in-polygon query, done via
Shapely rather than anything homegrown.
"""

from shapely import Point, Polygon, prepared
from shapely.geometry.base import BaseGeometry

from .cell_id import encode
from .coords import cartesian_to_axial
from .geometry import cell_centre, cell_polygon
from .orientation import Orientation

_PREDICATES = ("overlap", "center", "full")


def _axial_bounds(
    bounds: tuple[float, float, float, float],
    side_length: int,
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
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
    predicate: str = "overlap",
) -> list[int]:
    """Return the ids of every (side_length, orientation) hex cell covering `polygon`.

    `polygon` is a Shapely `Polygon`/`MultiPolygon` in EPSG:27700 metres --
    beahiv's native CRS, so no reprojection happens here.

    `predicate` controls what counts as "covering" (naming matches the h3
    `contain` argument):
      - "overlap" (default): any part of the hex touches the polygon
      - "center": the hex's centre falls inside the polygon
      - "full": the hex lies entirely inside the polygon
    """
    if predicate not in _PREDICATES:
        raise ValueError(f"predicate must be one of {_PREDICATES}, got {predicate!r}")
    if polygon.is_empty:
        return []

    q_min, q_max, r_min, r_max = _axial_bounds(polygon.bounds, side_length, orientation)
    prepared_polygon = prepared.prep(polygon)

    cells = []
    for q in range(q_min, q_max + 1):
        for r in range(r_min, r_max + 1):
            cell_id = encode(q, r, side_length, orientation)
            if predicate == "center":
                hit = prepared_polygon.contains(Point(cell_centre(cell_id)))
            elif predicate == "full":
                hit = prepared_polygon.contains(Polygon(cell_polygon(cell_id)))
            else:
                hit = prepared_polygon.intersects(Polygon(cell_polygon(cell_id)))
            if hit:
                cells.append(cell_id)
    return cells
