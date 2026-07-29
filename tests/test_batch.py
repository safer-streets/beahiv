import numpy as np
import pytest

from beahiv import Orientation, decode, encode
from beahiv.batch import (
    bng_to_cell_batch,
    cell_to_latlon_batch,
    decode_batch,
    encode_batch,
    latlon_to_cell_batch,
)
from beahiv.cell_id import SIDE_LENGTH_MASK, SIDE_LENGTH_MAX
from beahiv.geo import bng_to_cell, centroid, latlon_to_cell


def test_encode_batch_matches_scalar_encode():
    rng = np.random.default_rng(0)
    q = rng.integers(-1000, 1000, size=500)
    r = rng.integers(-1000, 1000, size=500)
    side_length = 250

    batch_ids = encode_batch(q, r, side_length, Orientation.POINTY)
    scalar_ids = [encode(int(qi), int(ri), side_length, Orientation.POINTY) for qi, ri in zip(q, r, strict=True)]

    assert list(batch_ids.astype(object)) == scalar_ids


@pytest.mark.parametrize("side_length", [0, SIDE_LENGTH_MAX + 1, SIDE_LENGTH_MASK])
def test_encode_batch_rejects_side_lengths_the_scalar_path_rejects(side_length):
    """Scalar/batch parity on validation, not just on the formula.

    side_length doesn't fill its field, so an unchecked oversized value would
    overflow into the orientation bit instead of being masked off.
    """
    q = np.array([0, 1])
    r = np.array([0, 1])
    with pytest.raises(ValueError):
        encode_batch(q, r, side_length, Orientation.FLAT)
    with pytest.raises(ValueError):
        encode(0, 0, side_length, Orientation.FLAT)


def test_decode_batch_matches_scalar_decode():
    rng = np.random.default_rng(1)
    q = rng.integers(-1000, 1000, size=500)
    r = rng.integers(-1000, 1000, size=500)
    side_length = 500
    orientation = Orientation.FLAT

    batch_ids = encode_batch(q, r, side_length, orientation)
    dq, dr, ds, do = decode_batch(batch_ids)

    for i in range(len(q)):
        idx = decode(int(batch_ids[i]))
        assert idx.q == dq[i]
        assert idx.r == dr[i]
        assert idx.side_length == ds[i]
        assert idx.orientation == do[i]


def test_latlon_to_cell_batch_matches_scalar():
    lats = np.array([51.5074, 55.9533, 51.4816])
    lons = np.array([-0.1278, -3.1883, -3.1791])
    side_length = 1000

    batch_ids = latlon_to_cell_batch(lats, lons, side_length, Orientation.POINTY)
    scalar_ids = [
        latlon_to_cell(float(lat), float(lon), side_length, Orientation.POINTY)
        for lat, lon in zip(lats, lons, strict=True)
    ]

    assert list(batch_ids.astype(object)) == scalar_ids


def test_cell_to_latlon_batch_matches_scalar():
    lats = np.array([51.5074, 55.9533, 51.4816])
    lons = np.array([-0.1278, -3.1883, -3.1791])
    side_length = 1000

    batch_ids = latlon_to_cell_batch(lats, lons, side_length, Orientation.POINTY)
    batch_lat, batch_lon = cell_to_latlon_batch(batch_ids)

    for i in range(len(lats)):
        scalar_lat, scalar_lon = centroid(int(batch_ids[i]), latlon=True)
        assert abs(batch_lat[i] - scalar_lat) < 1e-9
        assert abs(batch_lon[i] - scalar_lon) < 1e-9


def test_latlon_to_cell_batch_rejects_point_outside_area_of_use():
    # London, Paris (outside EPSG:27700's area of use).
    lats = np.array([51.5074, 48.8566])
    lons = np.array([-0.1278, 2.3522])
    with pytest.raises(ValueError, match="area of use"):
        latlon_to_cell_batch(lats, lons, side_length=500)


def test_latlon_to_cell_batch_still_maps_nan_to_invalid():
    lats = np.array([51.5074, np.nan])
    lons = np.array([-0.1278, np.nan])
    ids = latlon_to_cell_batch(lats, lons, side_length=500)
    assert ids[1] == 0


def test_bng_to_cell_batch_matches_scalar():
    rng = np.random.default_rng(1)
    x = rng.uniform(100_000, 600_000, size=200)
    y = rng.uniform(50_000, 900_000, size=200)
    side_length = 202

    batch_ids = bng_to_cell_batch(x, y, side_length, Orientation.FLAT)
    scalar_ids = [bng_to_cell(float(xi), float(yi), side_length, Orientation.FLAT) for xi, yi in zip(x, y, strict=True)]

    assert list(batch_ids.astype(object)) == scalar_ids


def test_bng_to_cell_batch_maps_nan_to_invalid():
    """Mirrors latlon_to_cell_batch: a NaN coordinate is an absent point, not an encode error."""
    x = np.array([530034.0, np.nan])
    y = np.array([180381.0, np.nan])

    ids = bng_to_cell_batch(x, y, side_length=202)

    assert ids[0] == bng_to_cell(530034.0, 180381.0, 202)
    assert ids[1] == 0


def test_bng_to_cell_batch_rejects_coordinates_beyond_the_bit_budget():
    """No area-of-use guard on the BNG path, but the q/r range check still catches absurd input."""
    with pytest.raises(ValueError, match="representable range"):
        bng_to_cell_batch(np.array([1e15]), np.array([1e15]), side_length=1)
