"""`point_to_cell` — the shapely/geopandas geometry entry point.

Tests only the plain Shapely surface `point_to_cell` accepts on its own -- a `Point`, or a
list/ndarray of them -- not any geopandas-shaped container (no GeoSeries/GeoDataFrame, real or
faked). `points.py`'s `.crs`/`.geometry` duck-typing for such containers is exercised by callers
that actually have geopandas installed, not by this suite.
"""

import numpy as np
import pytest
import shapely
from shapely import LineString, Point, Polygon

from beahiv import Orientation, bng_to_cell, decode, point_to_cell
from beahiv.cell_id import INVALID_CELL_ID

_BNG_POINTS = [(530000.0, 180000.0), (531000.0, 181000.0), (409000.0, 802000.0)]


def _points(coords):
    return [Point(x, y) for x, y in coords]


def test_matches_bng_to_cell_scalar_by_scalar():
    for orientation in Orientation:
        for side_length in (10, 500):
            cell_ids = point_to_cell(_points(_BNG_POINTS), side_length, orientation)
            expected = [bng_to_cell(x, y, side_length, orientation) for x, y in _BNG_POINTS]
            assert list(cell_ids) == expected


def test_accepts_list_and_ndarray_of_points():
    points = _points(_BNG_POINTS)
    expected = point_to_cell(points, 100)
    assert np.array_equal(point_to_cell(np.asarray(points, dtype=object), 100), expected)


def test_returns_plain_uint64_numpy_array():
    cell_ids = point_to_cell(_points(_BNG_POINTS), 100)
    assert isinstance(cell_ids, np.ndarray)
    assert cell_ids.dtype == np.uint64


def test_missing_and_empty_points_become_invalid_cell_id():
    points = np.array([Point(530000, 180000), None, Point(), Point(531000, 181000)], dtype=object)
    cell_ids = point_to_cell(points, 100)

    assert cell_ids[1] == INVALID_CELL_ID
    assert cell_ids[2] == INVALID_CELL_ID
    assert cell_ids[0] == bng_to_cell(530000.0, 180000.0, 100)
    assert cell_ids[3] == bng_to_cell(531000.0, 181000.0, 100)


def test_masked_path_agrees_with_the_all_present_fast_path():
    # The two branches in _encode_many must produce identical ids for the points they share.
    coords = [(530000.0 + 137 * i, 180000.0 + 211 * i) for i in range(50)]
    dense = point_to_cell(_points(coords), 250)

    with_gaps: list[Point | None] = [Point(x, y) for x, y in coords]
    with_gaps[7] = None
    with_gaps[23] = Point()
    sparse = point_to_cell(np.array(with_gaps, dtype=object), 250)

    kept = [i for i in range(len(coords)) if i not in (7, 23)]
    assert np.array_equal(sparse[kept], dense[kept])
    assert (sparse[[7, 23]] == INVALID_CELL_ID).all()


def test_empty_input_returns_empty_array():
    cell_ids = point_to_cell([], 100)
    assert cell_ids.shape == (0,)
    assert cell_ids.dtype == np.uint64


def test_cells_carry_the_requested_side_length_and_orientation():
    for orientation in Orientation:
        for cell_id in point_to_cell(_points(_BNG_POINTS), 750, orientation):
            idx = decode(int(cell_id))
            assert idx.side_length == 750
            assert idx.orientation == orientation


def test_ignores_z_on_3d_points():
    flat = point_to_cell([Point(530000, 180000)], 100)
    with_z = point_to_cell([Point(530000, 180000, 42)], 100)
    assert np.array_equal(flat, with_z)


@pytest.mark.parametrize(
    ("geometry", "name"),
    [
        (Polygon([(0, 0), (1, 0), (1, 1)]), "POLYGON"),
        (LineString([(0, 0), (1, 1)]), "LINESTRING"),
        (shapely.multipoints([(0, 0), (1, 1)]), "MULTIPOINT"),
    ],
)
def test_rejects_non_point_geometries(geometry, name):
    points = [Point(530000, 180000), geometry]
    with pytest.raises(ValueError, match=f"requires point geometries, got {name}"):
        point_to_cell(points, 100)


# --- scalar form -------------------------------------------------------------------------------


def test_scalar_point_returns_an_int_matching_bng_to_cell():
    for orientation in Orientation:
        cell_id = point_to_cell(Point(530000, 180000), 100, orientation)
        assert isinstance(cell_id, int)
        assert cell_id == bng_to_cell(530000.0, 180000.0, 100, orientation)


def test_scalar_agrees_with_the_array_form():
    points = _points(_BNG_POINTS)
    assert [point_to_cell(p, 250) for p in points] == list(point_to_cell(points, 250))


def test_scalar_missing_and_empty_give_invalid_cell_id():
    assert point_to_cell(None, 100) == INVALID_CELL_ID
    assert point_to_cell(Point(), 100) == INVALID_CELL_ID


def test_undeclared_crs_is_assumed_to_be_bng():
    # Shapely geometry carries no CRS, and a plain list/array has no `.crs` at all, so input is
    # taken at its word as EPSG:27700 metres.
    assert point_to_cell(np.array([Point(530000, 180000)]), 100)[0] == bng_to_cell(530000.0, 180000.0, 100)
    assert point_to_cell(Point(530000, 180000), 100) == bng_to_cell(530000.0, 180000.0, 100)
