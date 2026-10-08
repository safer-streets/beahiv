"""`point_to_cell` — the shapely/geopandas geometry entry point.

Mostly the plain Shapely surface -- a `Point`, or a list/ndarray of them. geopandas is never
imported; the CRS-dispatch tests at the end use `_Container`, a plain class carrying just the
`.geometry`/`.crs` pair `points.py` duck-types on, because the declared CRS now decides how
coordinates are read rather than merely being checked.
"""

import numpy as np
import pytest
import shapely
from pyproj import CRS
from shapely import LineString, Point, Polygon

from beahiv import Orientation, SizeMeasure, bng_to_cell, centroid, decode, lonlat_to_cell, point_to_cell
from beahiv.cell_id import INVALID_CELL_ID

_BNG_POINTS = [(530000.0, 180000.0), (531000.0, 181000.0), (409000.0, 802000.0)]


def _points(coords):
    return [Point(x, y) for x, y in coords]


def test_matches_bng_to_cell_scalar_by_scalar():
    for orientation in Orientation:
        for side_length in (10, 500):
            cell_ids = point_to_cell(_points(_BNG_POINTS), side_length=side_length, orientation=orientation)
            expected = [bng_to_cell(x, y, side_length=side_length, orientation=orientation) for x, y in _BNG_POINTS]
            assert list(cell_ids) == expected


def test_accepts_list_and_ndarray_of_points():
    points = _points(_BNG_POINTS)
    expected = point_to_cell(points, side_length=100)
    assert np.array_equal(point_to_cell(np.asarray(points, dtype=object), side_length=100), expected)


def test_returns_plain_int64_numpy_array():
    cell_ids = point_to_cell(_points(_BNG_POINTS), side_length=100)
    assert isinstance(cell_ids, np.ndarray)
    assert cell_ids.dtype == np.int64


def test_missing_and_empty_points_become_invalid_cell_id():
    points = np.array([Point(530000, 180000), None, Point(), Point(531000, 181000)], dtype=object)
    cell_ids = point_to_cell(points, side_length=100)

    assert cell_ids[1] == INVALID_CELL_ID
    assert cell_ids[2] == INVALID_CELL_ID
    assert cell_ids[0] == bng_to_cell(530000.0, 180000.0, side_length=100)
    assert cell_ids[3] == bng_to_cell(531000.0, 181000.0, side_length=100)


def test_masked_path_agrees_with_the_all_present_fast_path():
    # The two branches in _encode_many must produce identical ids for the points they share.
    coords = [(530000.0 + 137 * i, 180000.0 + 211 * i) for i in range(50)]
    dense = point_to_cell(_points(coords), side_length=250)

    with_gaps: list[Point | None] = [Point(x, y) for x, y in coords]
    with_gaps[7] = None
    with_gaps[23] = Point()
    sparse = point_to_cell(np.array(with_gaps, dtype=object), side_length=250)

    kept = [i for i in range(len(coords)) if i not in (7, 23)]
    assert np.array_equal(sparse[kept], dense[kept])
    assert (sparse[[7, 23]] == INVALID_CELL_ID).all()


def test_empty_input_returns_empty_array():
    cell_ids = point_to_cell([], side_length=100)
    assert cell_ids.shape == (0,)
    assert cell_ids.dtype == np.int64


def test_cells_carry_the_requested_side_length_and_orientation():
    for orientation in Orientation:
        for cell_id in point_to_cell(_points(_BNG_POINTS), side_length=750, orientation=orientation):
            idx = decode(int(cell_id))
            assert idx.side_length == 750
            assert idx.orientation == orientation


def test_ignores_z_on_3d_points():
    flat = point_to_cell([Point(530000, 180000)], side_length=100)
    with_z = point_to_cell([Point(530000, 180000, 42)], side_length=100)
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
        point_to_cell(points, side_length=100)


# --- scalar form -------------------------------------------------------------------------------


def test_scalar_point_returns_an_int_matching_bng_to_cell():
    for orientation in Orientation:
        cell_id = point_to_cell(Point(530000, 180000), side_length=100, orientation=orientation)
        assert isinstance(cell_id, int)
        assert cell_id == bng_to_cell(530000.0, 180000.0, side_length=100, orientation=orientation)


def test_scalar_agrees_with_the_array_form():
    points = _points(_BNG_POINTS)
    assert [point_to_cell(p, side_length=250) for p in points] == list(point_to_cell(points, side_length=250))


def test_scalar_missing_and_empty_give_invalid_cell_id():
    assert point_to_cell(None, side_length=100) == INVALID_CELL_ID
    assert point_to_cell(Point(), side_length=100) == INVALID_CELL_ID


def test_undeclared_crs_is_assumed_to_be_bng():
    # Shapely geometry carries no CRS, and a plain list/array has no `.crs` at all, so input is
    # taken at its word as EPSG:27700 metres.
    assert point_to_cell(np.array([Point(530000, 180000)]), side_length=100)[0] == bng_to_cell(
        530000.0, 180000.0, side_length=100
    )
    assert point_to_cell(Point(530000, 180000), side_length=100) == bng_to_cell(530000.0, 180000.0, side_length=100)


def test_point_to_cell_side_to_side_matches_bng_to_cell():
    points = _points(_BNG_POINTS)
    ids = point_to_cell(points, orientation=Orientation.POINTY, side_to_side=750)
    for (x, y), cell_id in zip(_BNG_POINTS, ids, strict=True):
        assert cell_id == bng_to_cell(x, y, orientation=Orientation.POINTY, side_to_side=750)
        assert decode(cell_id).measure == SizeMeasure.SIDE_TO_SIDE
    assert point_to_cell(Point(*_BNG_POINTS[0]), side_to_side=750) == bng_to_cell(*_BNG_POINTS[0], side_to_side=750)


def test_point_to_cell_requires_exactly_one_size_even_with_no_points():
    with pytest.raises(TypeError):
        point_to_cell(None, side_length=100, side_to_side=100)
    with pytest.raises(TypeError):
        point_to_cell([])


# --- WGS84 input -------------------------------------------------------------------------------

_LATLONS = [(51.5007, -0.1246), (55.9533, -3.1883), (53.4808, -2.2426)]


class _Container:
    """The `.geometry`/`.crs` pair a GeoSeries exposes, without importing geopandas."""

    def __init__(self, points, epsg):
        self.geometry = np.asarray(points, dtype=object)
        self.crs = CRS.from_epsg(epsg)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return self.geometry


def _lonlat_points():
    return [Point(lon, lat) for lat, lon in _LATLONS]


def test_lonlat_flag_matches_lonlat_to_cell():
    for orientation in Orientation:
        expected = [lonlat_to_cell(lon, lat, side_length=100, orientation=orientation) for lat, lon in _LATLONS]
        assert list(point_to_cell(_lonlat_points(), side_length=100, orientation=orientation, lonlat=True)) == expected
        assert [
            point_to_cell(p, side_length=100, orientation=orientation, lonlat=True) for p in _lonlat_points()
        ] == expected


def test_lonlat_side_to_side_matches_lonlat_to_cell():
    expected = [lonlat_to_cell(lon, lat, side_to_side=400) for lat, lon in _LATLONS]
    assert list(point_to_cell(_lonlat_points(), side_to_side=400, lonlat=True)) == expected


def test_lonlat_round_trips_through_centroid():
    for orientation in Orientation:
        cell_id = bng_to_cell(530000.0, 180000.0, side_length=250, orientation=orientation)
        point = centroid(cell_id, lonlat=True)
        assert point_to_cell(point, side_length=250, orientation=orientation, lonlat=True) == cell_id


def test_lonlat_missing_and_empty_points_become_invalid_cell_id():
    points = np.array([Point(-0.1246, 51.5007), None, Point()], dtype=object)
    cell_ids = point_to_cell(points, side_length=100, lonlat=True)
    assert cell_ids[0] == lonlat_to_cell(-0.1246, 51.5007, side_length=100)
    assert (cell_ids[1:] == INVALID_CELL_ID).all()
    assert point_to_cell(None, side_length=100, lonlat=True) == INVALID_CELL_ID
    assert point_to_cell(Point(), side_length=100, lonlat=True) == INVALID_CELL_ID


def test_lonlat_swapped_axes_are_rejected():
    # Point(lat, lon) instead of Point(lon, lat) falls outside EPSG:27700's area of use.
    with pytest.raises(ValueError, match="area of use"):
        point_to_cell(Point(51.5007, -0.1246), side_length=100, lonlat=True)
    with pytest.raises(ValueError, match="area of use"):
        point_to_cell([Point(51.5007, -0.1246)], side_length=100, lonlat=True)


def test_container_crs_selects_the_coordinate_reading():
    wgs84 = point_to_cell(_Container(_lonlat_points(), 4326), side_length=100)
    assert np.array_equal(wgs84, point_to_cell(_lonlat_points(), side_length=100, lonlat=True))

    bng = point_to_cell(_Container(_points(_BNG_POINTS), 27700), side_length=100)
    assert np.array_equal(bng, point_to_cell(_points(_BNG_POINTS), side_length=100))


def test_container_crs_agreeing_with_lonlat_flag_is_accepted():
    assert np.array_equal(
        point_to_cell(_Container(_lonlat_points(), 4326), side_length=100, lonlat=True),
        point_to_cell(_Container(_lonlat_points(), 4326), side_length=100),
    )
    assert np.array_equal(
        point_to_cell(_Container(_points(_BNG_POINTS), 27700), side_length=100, lonlat=False),
        point_to_cell(_Container(_points(_BNG_POINTS), 27700), side_length=100),
    )


def test_container_crs_contradicting_lonlat_flag_raises():
    with pytest.raises(ValueError, match="contradicts"):
        point_to_cell(_Container(_lonlat_points(), 4326), side_length=100, lonlat=False)
    with pytest.raises(ValueError, match="contradicts"):
        point_to_cell(_Container(_points(_BNG_POINTS), 27700), side_length=100, lonlat=True)


def test_container_in_an_unsupported_crs_raises():
    with pytest.raises(ValueError, match="EPSG:3857"):
        point_to_cell(_Container(_points(_BNG_POINTS), 3857), side_length=100)


def test_lonlat_is_keyword_only():
    with pytest.raises(TypeError):
        point_to_cell(Point(-0.1246, 51.5007), True, side_length=100)  # ty: ignore[no-matching-overload] -- the positional call is the point
