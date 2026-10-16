import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from beahiv import Measure, Orientation, bbox_fill, centroid, decode, encode, polyfill, resize_cell
from beahiv.geometry import cell_polygon


def _square(minx, miny, maxx, maxy):
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


@pytest.mark.parametrize("orientation", list(Orientation))
def test_overlap_cells_all_intersect_polygon(orientation):
    polygon = _square(0, 0, 2000, 2000)
    side_length = 100

    cells = polyfill(
        polygon, size=side_length, measure=Measure.EDGE_LENGTH, orientation=orientation, predicate="overlap"
    )

    assert cells
    for cell_id in cells:
        hexagon = cell_polygon(cell_id)
        assert hexagon.intersects(polygon)


def test_overlap_covers_every_point_in_polygon():
    # every point on a fine sample grid inside the polygon must fall in some returned cell
    polygon = _square(0, 0, 1000, 1000)
    side_length = 150

    cells = polyfill(
        polygon, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT, predicate="overlap"
    )
    hexagons = [cell_polygon(c) for c in cells]

    for x in range(10, 1000, 37):
        for y in range(10, 1000, 41):
            point = Point(x, y)
            assert any(h.covers(point) for h in hexagons), f"point ({x}, {y}) not covered"


def test_predicate_full_is_subset_of_center_is_subset_of_overlap():
    polygon = _square(0, 0, 3000, 3000)
    side_length = 202

    overlap = set(
        polyfill(
            polygon, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT, predicate="overlap"
        )
    )
    center = set(
        polyfill(
            polygon, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT, predicate="center"
        )
    )
    full = set(
        polyfill(polygon, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT, predicate="full")
    )

    assert full <= center <= overlap
    assert full  # the square is large enough relative to the cell to contain some fully


def test_center_predicate_matches_centroid_containment():
    polygon = _square(0, 0, 2000, 2000)
    side_length = 100

    cells = polyfill(
        polygon, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY, predicate="center"
    )

    for cell_id in cells:
        assert polygon.contains(centroid(cell_id))


def test_empty_polygon_returns_no_cells():
    assert polyfill(Polygon(), size=100, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT) == []


def test_cells_share_requested_side_length_and_orientation():
    polygon = _square(0, 0, 1000, 1000)
    side_length = 202

    cells = polyfill(polygon, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)

    assert cells
    for cell_id in cells:
        idx = decode(cell_id)
        assert idx.side_length == side_length
        assert idx.orientation == Orientation.FLAT


def test_invalid_predicate_raises():
    polygon = _square(0, 0, 100, 100)
    with pytest.raises(ValueError):
        polyfill(polygon, size=100, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT, predicate="bogus")


@pytest.mark.parametrize("predicate", ["overlap", "center", "full"])
@pytest.mark.parametrize("orientation", list(Orientation))
def test_bbox_fill_matches_polyfill_of_equivalent_square(predicate, orientation):
    minx, miny, maxx, maxy = 0, 0, 2000, 2000
    side_length = 100

    assert bbox_fill(
        minx,
        miny,
        maxx,
        maxy,
        size=side_length,
        measure=Measure.EDGE_LENGTH,
        orientation=orientation,
        predicate=predicate,
    ) == polyfill(
        _square(minx, miny, maxx, maxy),
        size=side_length,
        measure=Measure.EDGE_LENGTH,
        orientation=orientation,
        predicate=predicate,
    )


def test_bbox_fill_cells_share_requested_side_length_and_orientation():
    cells = bbox_fill(0, 0, 1000, 1000, size=202, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)

    assert cells
    for cell_id in cells:
        idx = decode(cell_id)
        assert idx.side_length == 202
        assert idx.orientation == Orientation.FLAT


def test_bbox_fill_degenerate_box_has_no_fully_contained_cell():
    """A zero-area box isn't a Shapely "empty" geometry (shapely.box(0, 0, 0, 0) is a valid,
    zero-area Polygon, not Polygon().is_empty), so "overlap" can still return the one cell
    touching that point -- but no hexagon (positive area) can ever be "full"y inside it."""
    assert (
        bbox_fill(0, 0, 0, 0, size=100, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT, predicate="full")
        == []
    )


def test_bbox_fill_invalid_predicate_raises():
    with pytest.raises(ValueError):
        bbox_fill(
            0, 0, 100, 100, size=100, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT, predicate="bogus"
        )


@pytest.mark.parametrize("predicate", ["overlap", "center", "full"])
@pytest.mark.parametrize("orientation", list(Orientation))
def test_resize_cell_matches_polyfill_of_the_cells_own_polygon(predicate, orientation):
    cell_id = encode(5, -3, size=300, measure=Measure.EDGE_LENGTH, orientation=orientation)
    new_side_length = 100

    assert resize_cell(cell_id, size=new_side_length, measure=Measure.EDGE_LENGTH, predicate=predicate) == polyfill(
        cell_polygon(cell_id),
        size=new_side_length,
        measure=Measure.EDGE_LENGTH,
        orientation=orientation,
        predicate=predicate,
    )


def test_resize_cell_to_a_finer_size_covers_the_original_hexagon():
    cell_id = encode(5, -3, size=300, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)
    hexagon = cell_polygon(cell_id)

    finer_cells = resize_cell(cell_id, size=30, measure=Measure.EDGE_LENGTH, predicate="overlap")
    finer_hexagons = [cell_polygon(c) for c in finer_cells]

    assert finer_cells
    minx, miny, maxx, maxy = hexagon.bounds
    for x in range(int(minx) + 5, int(maxx), 17):
        for y in range(int(miny) + 5, int(maxy), 19):
            point = Point(x, y)
            if hexagon.covers(point):
                assert any(h.covers(point) for h in finer_hexagons), f"point ({x}, {y}) not covered"


def test_resize_cell_to_a_coarser_size_is_permitted():
    """Larger new_side_length than the original cell must work -- this is the replacement for the
    same-centroid get_parent/get_child concept, generalised to any ratio via polyfill."""
    cell_id = encode(5, -3, size=100, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)

    coarser_cells = resize_cell(cell_id, size=300, measure=Measure.EDGE_LENGTH, predicate="overlap")

    assert coarser_cells
    for c in coarser_cells:
        idx = decode(c)
        assert idx.side_length == 300
        assert idx.orientation == Orientation.FLAT
        assert cell_polygon(c).intersects(cell_polygon(cell_id))


def test_resize_cell_cells_share_requested_side_length_and_original_orientation():
    cell_id = encode(0, 0, size=202, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY)

    cells = resize_cell(cell_id, size=50, measure=Measure.EDGE_LENGTH)

    assert cells
    for c in cells:
        idx = decode(c)
        assert idx.side_length == 50
        assert idx.orientation == Orientation.POINTY


def test_resize_cell_invalid_predicate_raises():
    cell_id = encode(0, 0, size=100, measure=Measure.EDGE_LENGTH)
    with pytest.raises(ValueError):
        resize_cell(cell_id, size=50, measure=Measure.EDGE_LENGTH, predicate="bogus")


def test_resize_cell_orientation_override():
    cell_id = encode(0, 0, size=202, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY)

    cells = resize_cell(cell_id, size=50, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)

    assert cells
    assert cells == polyfill(cell_polygon(cell_id), size=50, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)
    for c in cells:
        idx = decode(c)
        assert idx.side_length == 50
        assert idx.orientation == Orientation.FLAT


def test_polyfill_edge_to_edge_centre_cells_have_centres_inside():
    polygon = _square(530_000, 180_000, 532_000, 181_500)
    cells = polyfill(
        polygon, orientation=Orientation.POINTY, predicate="centre", size=200, measure=Measure.EDGE_TO_EDGE
    )
    assert cells
    for cell_id in cells:
        idx = decode(cell_id)
        assert (idx.size, idx.measure) == (200, Measure.EDGE_TO_EDGE)
        assert polygon.contains(centroid(cell_id))
    assert cells == bbox_fill(
        530_000,
        180_000,
        532_000,
        181_500,
        orientation=Orientation.POINTY,
        predicate="centre",
        size=200,
        measure=Measure.EDGE_TO_EDGE,
    )


def test_polyfill_requires_size_and_defaults_the_measure():
    polygon = _square(0, 0, 100, 100)
    assert polyfill(polygon, size=10) == polyfill(polygon, size=10, measure=Measure.EDGE_LENGTH)
    with pytest.raises(TypeError):
        polyfill(polygon, measure=Measure.EDGE_LENGTH)  # ty: ignore[missing-argument] -- the missing size is the point
    with pytest.raises(TypeError, match="measure must be a Measure"):
        polyfill(polygon, size=10, measure=1)  # ty: ignore[invalid-argument-type] -- the bad measure is the point


@pytest.mark.parametrize("measure", list(Measure))
def test_resize_cell_measure_defaults_to_the_cells_own(measure):
    cell_id = encode(10, -4, size=300, measure=measure)
    cells = resize_cell(cell_id, size=100)
    assert cells == resize_cell(cell_id, size=100, measure=measure)
    assert {decode(c).measure for c in cells} == {measure}


def test_resize_cell_to_edge_to_edge_covers_the_cell():
    cell_id = encode(10, -4, size=300, measure=Measure.EDGE_LENGTH)
    cells = resize_cell(cell_id, size=100, measure=Measure.EDGE_TO_EDGE, predicate="overlap")
    assert {(decode(c).size, decode(c).measure) for c in cells} == {(100, Measure.EDGE_TO_EDGE)}
    cell = cell_polygon(cell_id)
    assert cell.difference(unary_union([cell_polygon(c) for c in cells])).area < 1e-9 * cell.area
