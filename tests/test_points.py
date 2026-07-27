"""`point_to_cell` — the shapely/geopandas geometry entry point.

geopandas is a dev dependency only (like pyarrow for `test_arrow.py`): beahiv never imports it,
but `point_to_cell` duck-types on `.geometry`/`.crs`, so those branches need the real thing.
"""

import geopandas as gpd
import numpy as np
import pytest
import shapely
from pyproj import CRS
from shapely import LineString, Point, Polygon

from beahiv import Orientation, bng_to_cell, decode, point_to_cell
from beahiv.cell_id import INVALID_CELL_ID

_BNG_POINTS = [(530000.0, 180000.0), (531000.0, 181000.0), (409000.0, 802000.0)]


def _geoseries(coords, crs="EPSG:27700"):
    return gpd.GeoSeries([Point(x, y) for x, y in coords], crs=crs)


def test_matches_bng_to_cell_scalar_by_scalar():
    for orientation in Orientation:
        for side_length in (10, 500):
            cell_ids = point_to_cell(_geoseries(_BNG_POINTS), side_length, orientation)
            expected = [bng_to_cell(x, y, side_length, orientation) for x, y in _BNG_POINTS]
            assert list(cell_ids) == expected


def test_accepts_geoseries_geodataframe_geometryarray_ndarray_and_list():
    gs = _geoseries(_BNG_POINTS)
    gdf = gpd.GeoDataFrame({"n": range(len(_BNG_POINTS))}, geometry=gs, crs="EPSG:27700")
    inputs = [gs, gdf, gs.values, np.asarray(gs, dtype=object), list(gs)]

    expected = point_to_cell(gs, 100)
    for source in inputs:
        assert np.array_equal(point_to_cell(source, 100), expected)


def test_returns_plain_uint64_numpy_array():
    # Not a Series: assigning back to a frame must be positional, so a non-default index on the
    # input can't silently realign the ids against the wrong rows.
    cell_ids = point_to_cell(_geoseries(_BNG_POINTS), 100)
    assert isinstance(cell_ids, np.ndarray)
    assert cell_ids.dtype == np.uint64


def test_non_default_index_does_not_shift_results():
    gs = _geoseries(_BNG_POINTS)
    reindexed = gpd.GeoSeries(list(gs), crs="EPSG:27700", index=[97, 98, 99])
    assert np.array_equal(point_to_cell(reindexed, 100), point_to_cell(gs, 100))


def test_missing_and_empty_points_become_invalid_cell_id():
    gs = gpd.GeoSeries([Point(530000, 180000), None, Point(), Point(531000, 181000)], crs="EPSG:27700")
    cell_ids = point_to_cell(gs, 100)

    assert cell_ids[1] == INVALID_CELL_ID
    assert cell_ids[2] == INVALID_CELL_ID
    assert cell_ids[0] == bng_to_cell(530000.0, 180000.0, 100)
    assert cell_ids[3] == bng_to_cell(531000.0, 181000.0, 100)


def test_masked_path_agrees_with_the_all_present_fast_path():
    # The two branches in _encode_many must produce identical ids for the points they share.
    coords = [(530000.0 + 137 * i, 180000.0 + 211 * i) for i in range(50)]
    dense = point_to_cell(_geoseries(coords), 250)

    with_gaps: list[Point | None] = [Point(x, y) for x, y in coords]
    with_gaps[7] = None
    with_gaps[23] = Point()
    sparse = point_to_cell(gpd.GeoSeries(with_gaps, crs="EPSG:27700"), 250)

    kept = [i for i in range(len(coords)) if i not in (7, 23)]
    assert np.array_equal(sparse[kept], dense[kept])
    assert (sparse[[7, 23]] == INVALID_CELL_ID).all()


def test_empty_input_returns_empty_array():
    cell_ids = point_to_cell(gpd.GeoSeries([], crs="EPSG:27700"), 100)
    assert cell_ids.shape == (0,)
    assert cell_ids.dtype == np.uint64


def test_cells_carry_the_requested_side_length_and_orientation():
    for orientation in Orientation:
        for cell_id in point_to_cell(_geoseries(_BNG_POINTS), 750, orientation):
            idx = decode(int(cell_id))
            assert idx.side_length == 750
            assert idx.orientation == orientation


def test_ignores_z_on_3d_points():
    flat = point_to_cell(gpd.GeoSeries([Point(530000, 180000)], crs="EPSG:27700"), 100)
    with_z = point_to_cell(gpd.GeoSeries([Point(530000, 180000, 42)], crs="EPSG:27700"), 100)
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
    gs = gpd.GeoSeries([Point(530000, 180000), geometry], crs="EPSG:27700")
    with pytest.raises(ValueError, match=f"requires point geometries, got {name}"):
        point_to_cell(gs, 100)


# --- scalar form -------------------------------------------------------------------------------


def test_scalar_point_returns_an_int_matching_bng_to_cell():
    for orientation in Orientation:
        cell_id = point_to_cell(Point(530000, 180000), 100, orientation)
        assert isinstance(cell_id, int)
        assert cell_id == bng_to_cell(530000.0, 180000.0, 100, orientation)


def test_scalar_agrees_with_the_array_form():
    gs = _geoseries(_BNG_POINTS)
    assert [point_to_cell(p, 250) for p in gs] == list(point_to_cell(gs, 250))


def test_scalar_missing_and_empty_give_invalid_cell_id():
    assert point_to_cell(None, 100) == INVALID_CELL_ID
    assert point_to_cell(Point(), 100) == INVALID_CELL_ID


# --- coordinate reference system ----------------------------------------------------------------


def test_undeclared_crs_is_assumed_to_be_bng():
    # Shapely geometry carries no CRS, so bare input is taken at its word as EPSG:27700 metres.
    assert point_to_cell(np.array([Point(530000, 180000)]), 100)[0] == bng_to_cell(530000.0, 180000.0, 100)
    assert point_to_cell(gpd.GeoSeries([Point(530000, 180000)]), 100)[0] == bng_to_cell(530000.0, 180000.0, 100)
    assert point_to_cell(Point(530000, 180000), 100) == bng_to_cell(530000.0, 180000.0, 100)


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857", "EPSG:4277"])
def test_rejects_a_declared_crs_that_is_not_bng(crs):
    # Nothing is reprojected here, so a column that says it holds something else has to raise --
    # degrees read as metres would encode beside the grid origin, valid-looking and wrong.
    gs = _geoseries(_BNG_POINTS).to_crs(crs)
    with pytest.raises(ValueError, match="not EPSG:27700"):
        point_to_cell(gs, 100)


def test_reprojecting_to_bng_first_is_accepted():
    wgs84 = gpd.GeoSeries([Point(-0.1278, 51.5074)], crs="EPSG:4326")
    reprojected = wgs84.to_crs("EPSG:27700")
    point = reprojected.iloc[0]
    assert point_to_cell(reprojected, 100)[0] == bng_to_cell(point.x, point.y, 100)


def test_crs_guard_accepts_bng_declared_without_an_epsg_code():
    wkt_bng = CRS.from_wkt(CRS.from_epsg(27700).to_wkt())
    gs = gpd.GeoSeries([Point(530000, 180000)], crs=wkt_bng)
    assert point_to_cell(gs, 100)[0] == bng_to_cell(530000.0, 180000.0, 100)
