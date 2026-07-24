import random

import numpy as np
import pytest

from beahiv import Orientation, bng_to_cell, cell_centre, centroid, decode, latlon_to_cell


def test_latlon_round_trip_stays_within_one_cell():
    # London-ish, Edinburgh-ish, Cardiff-ish: spread across GB.
    points = [
        (51.5074, -0.1278),
        (55.9533, -3.1883),
        (51.4816, -3.1791),
        (53.4808, -2.2426),
    ]
    for lat, lon in points:
        for orientation in Orientation:
            cell_id = latlon_to_cell(lat, lon, side_length=100, orientation=orientation)
            recovered_lat, recovered_lon = centroid(cell_id, latlon=True)
            # Cell centre must be close to the original point (within ~ one cell diameter).
            assert abs(recovered_lat - lat) < 0.01
            assert abs(recovered_lon - lon) < 0.01


def test_latlon_to_cell_is_deterministic():
    rng = random.Random(7)
    for _ in range(50):
        lat = rng.uniform(50.0, 58.5)
        lon = rng.uniform(-6.0, 1.5)
        side_length = rng.randint(10, 5000)
        a = latlon_to_cell(lat, lon, side_length)
        b = latlon_to_cell(lat, lon, side_length)
        assert a == b


def test_centroid_matches_decoded_cell():
    cell_id = latlon_to_cell(51.5074, -0.1278, side_length=1000, orientation=Orientation.FLAT)
    idx = decode(cell_id)
    assert idx.side_length == 1000
    assert idx.orientation == Orientation.FLAT

    lat, lon = centroid(cell_id, latlon=True)
    assert 49.0 < lat < 61.0
    assert -8.0 < lon < 2.0


def test_centroid_defaults_to_bng():
    cell_id = latlon_to_cell(51.5074, -0.1278, side_length=1000, orientation=Orientation.FLAT)
    assert centroid(cell_id) == cell_centre(cell_id)


def test_latlon_to_cell_rejects_swapped_lat_lon():
    lat, lon = 51.5074, -0.1278  # London
    with pytest.raises(ValueError, match="area of use"):
        latlon_to_cell(lon, lat, side_length=4)


def test_latlon_to_cell_rejects_point_outside_gb():
    paris_lat, paris_lon = 48.8566, 2.3522
    with pytest.raises(ValueError, match="area of use"):
        latlon_to_cell(paris_lat, paris_lon, side_length=100)


def test_bng_to_cell_matches_latlon_to_cell():
    from pyproj import Transformer

    to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    lat, lon = 51.5074, -0.1278  # London
    x, y = to_bng.transform(lon, lat)
    for side_length in (10, 500):
        for orientation in Orientation:
            assert bng_to_cell(x, y, side_length, orientation) == latlon_to_cell(lat, lon, side_length, orientation)


def test_bng_to_cell_round_trips_through_cell_centre():
    x, y = 530000.0, 180000.0
    for side_length in (50, 500):
        cell_id = bng_to_cell(x, y, side_length)
        cx, cy = cell_centre(cell_id)
        assert abs(cx - x) < side_length
        assert abs(cy - y) < side_length


def test_latlon_to_cell_dispatches_transparently_to_array_input():
    lats = [51.5074, 55.9533, 51.4816]
    lons = [-0.1278, -3.1883, -3.1791]
    side_length = 1000

    array_ids = latlon_to_cell(np.array(lats), np.array(lons), side_length)
    list_ids = latlon_to_cell(lats, lons, side_length)
    scalar_ids = [latlon_to_cell(lat, lon, side_length) for lat, lon in zip(lats, lons, strict=True)]

    assert list(array_ids) == scalar_ids
    assert list(list_ids) == scalar_ids


def test_centroid_dispatches_transparently_to_array_input():
    cell_ids = [
        latlon_to_cell(51.5074, -0.1278, side_length=500),
        latlon_to_cell(55.9533, -3.1883, side_length=500),
    ]

    x_arr, y_arr = centroid(np.array(cell_ids))
    lat_arr, lon_arr = centroid(np.array(cell_ids), latlon=True)
    for i, cell_id in enumerate(cell_ids):
        scalar_x, scalar_y = centroid(cell_id)
        scalar_lat, scalar_lon = centroid(cell_id, latlon=True)
        assert x_arr[i] == pytest.approx(scalar_x)
        assert y_arr[i] == pytest.approx(scalar_y)
        assert lat_arr[i] == pytest.approx(scalar_lat)
        assert lon_arr[i] == pytest.approx(scalar_lon)


def test_bng_to_cell_dispatches_transparently_to_array_input():
    xs = np.array([530000.0, 300000.0])
    ys = np.array([180000.0, 600000.0])
    side_length = 500

    array_ids = bng_to_cell(xs, ys, side_length)
    scalar_ids = [bng_to_cell(float(x), float(y), side_length) for x, y in zip(xs, ys, strict=True)]
    assert list(array_ids) == scalar_ids


def test_latlon_to_cell_array_input_rejects_out_of_domain_point():
    lats = np.array([51.5074, 48.8566])  # London, Paris
    lons = np.array([-0.1278, 2.3522])
    with pytest.raises(ValueError, match="area of use"):
        latlon_to_cell(lats, lons, side_length=500)
