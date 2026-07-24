"""Plain-Python geometry helpers used only by tests.

Deliberately independent of Shapely: the spec requires the core indexing
code to have no Shapely dependency, and these helpers double as an
external check on cell_polygon rather than reusing library internals.
"""

import math


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace formula, computed relative to the first vertex.

    Cell centres can sit far from the EPSG:27700 origin (~1e9 in extreme
    cases), and the raw shoelace formula multiplies those large absolute
    coordinates together -- this cancels catastrophically since the
    result (the area) is many orders of magnitude smaller than the
    intermediate products. Subtracting a reference point first keeps
    every value used in the products small (~side_length) without
    changing the area, since area is translation invariant.
    """
    x0, y0 = polygon[0]
    local = [(x - x0, y - y0) for x, y in polygon]
    n = len(local)
    total = 0.0
    for i in range(n):
        x1, y1 = local[i]
        x2, y2 = local[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def point_in_polygon_with_tolerance(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
    centre: tuple[float, float],
    relative_inflation: float = 1e-6,
) -> bool:
    """Point-in-polygon that tolerates floating-point noise on the boundary
    by testing against a slightly outward-scaled copy of the polygon."""
    xc, yc = centre
    inflated = [
        (xc + (x - xc) * (1 + relative_inflation), yc + (y - yc) * (1 + relative_inflation)) for x, y in polygon
    ]
    return point_in_polygon(point, inflated)


def hex_area(side_length: float) -> float:
    return (3 * math.sqrt(3) / 2) * side_length**2
