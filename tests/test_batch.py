import numpy as np
import pytest

from beahiv import Orientation, decode, encode
from beahiv.batch import (
    bng_to_cell_batch,
    cell_to_latlon_batch,
    cell_to_lonlat_batch,
    decode_batch,
    encode_batch,
    latlon_to_cell_batch,
    lonlat_to_cell_batch,
)
from beahiv.cell_id import (
    INVALID_CELL_ID,
    Q_MASK,
    Q_OFFSET,
    R_MASK,
    R_OFFSET,
    RESERVED_SHIFT,
    SIDE_LENGTH_MASK,
    SIDE_LENGTH_MAX,
    SIDE_LENGTH_SHIFT,
)
from beahiv.geo import bng_to_cell, lonlat_to_cell
from beahiv.geometry import centroid


def test_encode_batch_matches_scalar_encode():
    rng = np.random.default_rng(0)
    q = rng.integers(-1000, 1000, size=500)
    r = rng.integers(-1000, 1000, size=500)
    side_length = 250

    batch_ids = encode_batch(q, r, side_length, Orientation.POINTY)
    scalar_ids = [encode(int(qi), int(ri), side_length, Orientation.POINTY) for qi, ri in zip(q, r, strict=True)]

    assert list(batch_ids.astype(object)) == scalar_ids


def test_batch_encoders_return_signed_int64():
    """Regression: these returned uint64, which is the one integer type consumers can't use.

    numpy promotes uint64 to float64 against any signed int, so arithmetic or a comparison
    against an id from the scalar path silently loses precision at these magnitudes; pandas
    and Arrow/BIGINT columns want the signed type too. The reserved bits keep every id below
    2**61, so nothing is given up by signing it.
    """
    ids = encode_batch(np.array([4, -6]), np.array([-6, 4]), 100)
    assert ids.dtype == np.int64
    assert (ids > 0).all()

    assert bng_to_cell_batch([530000.0], [180000.0], 100).dtype == np.int64
    assert lonlat_to_cell_batch([-0.1278], [51.5074], 100).dtype == np.int64
    assert bng_to_cell(np.array([530000.0]), np.array([180000.0]), 100).dtype == np.int64

    # the promotion the signed type avoids: uint64 - int64 would land in float64
    assert (ids - np.int64(1)).dtype == np.int64


def test_decode_batch_still_accepts_uint64_ids():
    """Ids stored before the switch, or read back from an unsigned column, decode unchanged."""
    cell_ids = [encode(4, -6, 100), encode(-6, 4, 100)]
    unsigned = decode_batch(np.array(cell_ids, dtype=np.uint64))
    signed = decode_batch(np.array(cell_ids, dtype=np.int64))

    assert all(np.array_equal(u, s) for u, s in zip(unsigned, signed, strict=True))


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


@pytest.mark.parametrize(
    ("q", "r"),
    [
        (-Q_OFFSET - 1, 0),  # q_enc just below 0
        (Q_MASK - Q_OFFSET + 1, 0),  # q_enc just above its mask
        (0, -R_OFFSET - 1),
        (0, R_MASK - R_OFFSET + 1),
    ],
)
def test_encode_batch_rejects_q_r_the_scalar_path_rejects(q, r):
    """Both edges of each field: in int64 a too-negative q/r no longer wraps round to a huge
    value that the upper-bound check would catch, so the lower bound needs its own check."""
    with pytest.raises(ValueError):
        encode(q, r, 100)
    with pytest.raises(ValueError, match="representable range"):
        encode_batch(np.array([0, q]), np.array([0, r]), 100)


@pytest.mark.parametrize(
    "encode_fn",
    [
        lambda: encode_batch(np.array([4, -4]), np.array([-6, 6]), 100),
        lambda: encode_batch(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 100),
        lambda: lonlat_to_cell_batch([-0.1278, np.nan], [51.5074, np.nan], 100),
        lambda: lonlat_to_cell_batch([np.nan], [np.nan], 100),
        lambda: bng_to_cell_batch([530034.0, np.nan], [180381.0, np.nan], 100),
        lambda: bng_to_cell_batch([], [], 100),
        lambda: lonlat_to_cell([-0.1278], [51.5074], 100),
        lambda: bng_to_cell([530034.0], [180381.0], 100),
    ],
)
def test_cell_id_arrays_are_int64_never_uint64(encode_fn):
    """Every path that returns an array of ids -- including the empty and all-missing ones, which
    never reach encode_batch -- returns signed int64.

    uint64 ids promote to float64 when combined with a Python int or an int64 array, silently
    dropping the low bits of an id, and have no native equivalent in a BIGINT column."""
    cell_ids = encode_fn()
    assert cell_ids.dtype == np.int64
    assert (cell_ids >= 0).all()


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


def test_decode_batch_rejects_what_scalar_decode_rejects():
    """Scalar/batch parity applies to the rejection contract, not just the arithmetic."""
    valid = encode(4, -6, 100)
    rejected = [
        valid | (0b101 << RESERVED_SHIFT),  # reserved bits set
        valid & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT),  # side_length 0
        valid & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT) | ((SIDE_LENGTH_MAX + 1) << SIDE_LENGTH_SHIFT),
        INVALID_CELL_ID,
    ]

    for cell_id in rejected:
        with pytest.raises(ValueError):
            decode(cell_id)  # scalar rejects it ...
        with pytest.raises(ValueError):
            decode_batch(np.array([valid, cell_id], dtype=np.uint64))  # ... so the batch must too
        # ... including when the ids arrive as plain Python ints rather than a typed array.
        # Regression: ids at or above 2**63 don't fit the int64 the batch path now uses, and numpy
        # raised OverflowError from the conversion instead of the ValueError the contract promises.
        with pytest.raises(ValueError):
            decode_batch([valid, cell_id])


def test_decode_batch_reports_the_sentinel_by_name():
    """The overwhelmingly likely cause of a bad id in a batch is an unfiltered missing-input
    sentinel, so the error says so rather than only quoting an out-of-range side_length."""
    cell_ids = np.array([encode(4, -6, 100), INVALID_CELL_ID], dtype=np.uint64)
    with pytest.raises(ValueError, match="INVALID_CELL_ID"):
        decode_batch(cell_ids)


def test_decode_batch_accepts_an_empty_array():
    dq, dr, ds, do = decode_batch(np.array([], dtype=np.uint64))
    assert len(dq) == len(dr) == len(ds) == len(do) == 0


def test_lonlat_to_cell_batch_matches_scalar():
    lats = np.array([51.5074, 55.9533, 51.4816])
    lons = np.array([-0.1278, -3.1883, -3.1791])
    side_length = 1000

    batch_ids = lonlat_to_cell_batch(lons, lats, side_length, Orientation.POINTY)
    scalar_ids = [
        lonlat_to_cell(float(lon), float(lat), side_length, Orientation.POINTY)
        for lat, lon in zip(lats, lons, strict=True)
    ]

    assert list(batch_ids.astype(object)) == scalar_ids


def test_cell_to_lonlat_batch_matches_scalar():
    lats = np.array([51.5074, 55.9533, 51.4816])
    lons = np.array([-0.1278, -3.1883, -3.1791])
    side_length = 1000

    batch_ids = lonlat_to_cell_batch(lons, lats, side_length, Orientation.POINTY)
    batch_lon, batch_lat = cell_to_lonlat_batch(batch_ids)

    for i in range(len(lats)):
        scalar_lon, scalar_lat = centroid(int(batch_ids[i]), lonlat=True).coords[0]
        assert abs(batch_lat[i] - scalar_lat) < 1e-9
        assert abs(batch_lon[i] - scalar_lon) < 1e-9


def test_lonlat_to_cell_batch_rejects_point_outside_area_of_use():
    # London, Paris (outside EPSG:27700's area of use).
    lats = np.array([51.5074, 48.8566])
    lons = np.array([-0.1278, 2.3522])
    with pytest.raises(ValueError, match="area of use"):
        lonlat_to_cell_batch(lons, lats, side_length=500)


def test_lonlat_to_cell_batch_still_maps_nan_to_invalid():
    lats = np.array([51.5074, np.nan])
    lons = np.array([-0.1278, np.nan])
    ids = lonlat_to_cell_batch(lons, lats, side_length=500)
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
    """Mirrors lonlat_to_cell_batch: a NaN coordinate is an absent point, not an encode error."""
    x = np.array([530034.0, np.nan])
    y = np.array([180381.0, np.nan])

    ids = bng_to_cell_batch(x, y, side_length=202)

    assert ids[0] == bng_to_cell(530034.0, 180381.0, 202)
    assert ids[1] == 0


def test_bng_to_cell_batch_rejects_coordinates_beyond_the_bit_budget():
    """No area-of-use guard on the BNG path, but the q/r range check still catches absurd input."""
    with pytest.raises(ValueError, match="representable range"):
        bng_to_cell_batch(np.array([1e15]), np.array([1e15]), side_length=1)


# --- deprecated lat-first spellings ------------------------------------------------------------


def test_deprecated_latlon_to_cell_batch_warns_and_takes_lats_first():
    lats, lons = [51.5074, 55.9533], [-0.1278, -3.1883]
    with pytest.deprecated_call(match="lonlat_to_cell_batch"):
        old = latlon_to_cell_batch(lats, lons, 500)
    assert np.array_equal(old, lonlat_to_cell_batch(lons, lats, 500))


def test_deprecated_cell_to_latlon_batch_warns_and_returns_lat_first():
    ids = lonlat_to_cell_batch([-0.1278, -3.1883], [51.5074, 55.9533], 500)
    with pytest.deprecated_call(match="cell_to_lonlat_batch"):
        lat, lon = cell_to_latlon_batch(ids)
    new_lon, new_lat = cell_to_lonlat_batch(ids)
    assert np.array_equal(lat, new_lat)
    assert np.array_equal(lon, new_lon)
