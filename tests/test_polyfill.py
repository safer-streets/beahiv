import pytest
from shapely.geometry import Point, Polygon

from beahiv import Orientation, centroid, decode, polyfill
from beahiv.geometry import cell_polygon


def _square(minx, miny, maxx, maxy):
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


@pytest.mark.parametrize("orientation", list(Orientation))
def test_overlap_cells_all_intersect_polygon(orientation):
    polygon = _square(0, 0, 2000, 2000)
    side_length = 100

    cells = polyfill(polygon, side_length, orientation, predicate="overlap")

    assert cells
    for cell_id in cells:
        hexagon = Polygon(cell_polygon(cell_id))
        assert hexagon.intersects(polygon)


def test_overlap_covers_every_point_in_polygon():
    # every point on a fine sample grid inside the polygon must fall in some returned cell
    polygon = _square(0, 0, 1000, 1000)
    side_length = 150

    cells = polyfill(polygon, side_length, Orientation.FLAT, predicate="overlap")
    hexagons = [Polygon(cell_polygon(c)) for c in cells]

    for x in range(10, 1000, 37):
        for y in range(10, 1000, 41):
            point = Point(x, y)
            assert any(h.covers(point) for h in hexagons), f"point ({x}, {y}) not covered"


def test_predicate_full_is_subset_of_center_is_subset_of_overlap():
    polygon = _square(0, 0, 3000, 3000)
    side_length = 202

    overlap = set(polyfill(polygon, side_length, Orientation.FLAT, predicate="overlap"))
    center = set(polyfill(polygon, side_length, Orientation.FLAT, predicate="center"))
    full = set(polyfill(polygon, side_length, Orientation.FLAT, predicate="full"))

    assert full <= center <= overlap
    assert full  # the square is large enough relative to the cell to contain some fully


def test_center_predicate_matches_centroid_containment():
    polygon = _square(0, 0, 2000, 2000)
    side_length = 100

    cells = polyfill(polygon, side_length, Orientation.POINTY, predicate="center")

    for cell_id in cells:
        x, y = centroid(cell_id)
        assert polygon.contains(Point(x, y))


def test_empty_polygon_returns_no_cells():
    assert polyfill(Polygon(), 100, Orientation.FLAT) == []


def test_cells_share_requested_side_length_and_orientation():
    polygon = _square(0, 0, 1000, 1000)
    side_length = 202

    cells = polyfill(polygon, side_length, Orientation.FLAT)

    assert cells
    for cell_id in cells:
        idx = decode(cell_id)
        assert idx.side_length == side_length
        assert idx.orientation == Orientation.FLAT


def test_invalid_predicate_raises():
    polygon = _square(0, 0, 100, 100)
    with pytest.raises(ValueError):
        polyfill(polygon, 100, Orientation.FLAT, predicate="bogus")
