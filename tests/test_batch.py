import numpy as np
import pytest

from beahiv import Measure, Orientation, batch, decode, encode
from beahiv.batch import (
    bng_to_cell_batch,
    cell_centre_batch,
    cell_to_lonlat_batch,
    decode_batch,
    encode_batch,
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

    batch_ids = encode_batch(q, r, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY)
    scalar_ids = [
        encode(int(qi), int(ri), size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY)
        for qi, ri in zip(q, r, strict=True)
    ]

    assert list(batch_ids.astype(object)) == scalar_ids


def test_batch_encoders_return_signed_int64():
    """Regression: these returned uint64, which is the one integer type consumers can't use.

    numpy promotes uint64 to float64 against any signed int, so arithmetic or a comparison
    against an id from the scalar path silently loses precision at these magnitudes; pandas
    and Arrow/BIGINT columns want the signed type too. The reserved bits keep every id below
    2**62, so nothing is given up by signing it.
    """
    ids = encode_batch(np.array([4, -6]), np.array([-6, 4]), size=100, measure=Measure.EDGE_LENGTH)
    assert ids.dtype == np.int64
    assert (ids > 0).all()

    assert bng_to_cell_batch([530000.0], [180000.0], size=100, measure=Measure.EDGE_LENGTH).dtype == np.int64
    assert lonlat_to_cell_batch([-0.1278], [51.5074], size=100, measure=Measure.EDGE_LENGTH).dtype == np.int64
    assert (
        bng_to_cell(np.array([530000.0]), np.array([180000.0]), size=100, measure=Measure.EDGE_LENGTH).dtype == np.int64
    )

    # the promotion the signed type avoids: uint64 - int64 would land in float64
    assert (ids - np.int64(1)).dtype == np.int64


def test_decode_batch_still_accepts_uint64_ids():
    """Ids stored before the switch, or read back from an unsigned column, decode unchanged."""
    cell_ids = [
        encode(4, -6, size=100, measure=Measure.EDGE_LENGTH),
        encode(-6, 4, size=100, measure=Measure.EDGE_LENGTH),
    ]
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
        encode_batch(q, r, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)
    with pytest.raises(ValueError):
        encode(0, 0, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)


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
        encode(q, r, size=100, measure=Measure.EDGE_LENGTH)
    with pytest.raises(ValueError, match="representable range"):
        encode_batch(np.array([0, q]), np.array([0, r]), size=100, measure=Measure.EDGE_LENGTH)


@pytest.mark.parametrize(
    "encode_fn",
    [
        lambda: encode_batch(np.array([4, -4]), np.array([-6, 6]), size=100, measure=Measure.EDGE_LENGTH),
        lambda: encode_batch(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64), size=100, measure=Measure.EDGE_LENGTH
        ),
        lambda: lonlat_to_cell_batch([-0.1278, np.nan], [51.5074, np.nan], size=100, measure=Measure.EDGE_LENGTH),
        lambda: lonlat_to_cell_batch([np.nan], [np.nan], size=100, measure=Measure.EDGE_LENGTH),
        lambda: bng_to_cell_batch([530034.0, np.nan], [180381.0, np.nan], size=100, measure=Measure.EDGE_LENGTH),
        lambda: bng_to_cell_batch([], [], size=100, measure=Measure.EDGE_LENGTH),
        lambda: lonlat_to_cell([-0.1278], [51.5074], size=100, measure=Measure.EDGE_LENGTH),
        lambda: bng_to_cell([530034.0], [180381.0], size=100, measure=Measure.EDGE_LENGTH),
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

    batch_ids = encode_batch(q, r, size=side_length, measure=Measure.EDGE_LENGTH, orientation=orientation)
    dq, dr, ds, dm, do = decode_batch(batch_ids)

    for i in range(len(q)):
        idx = decode(int(batch_ids[i]))
        assert idx.q == dq[i]
        assert idx.r == dr[i]
        assert idx.size == ds[i]
        assert idx.orientation == do[i]
        assert idx.measure == dm[i]


_VALID = encode(4, -6, size=100, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)
_REJECTED = [
    _VALID | (0b10 << RESERVED_SHIFT),  # reserved bits set, including bit 63
    _VALID & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT),  # side_length 0
    _VALID & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT) | ((SIDE_LENGTH_MAX + 1) << SIDE_LENGTH_SHIFT),
    INVALID_CELL_ID,
]


@pytest.mark.parametrize("cell_id", _REJECTED)
def test_decode_batch_zeroes_what_scalar_decode_rejects(cell_id):
    """Scalar/batch parity on *which* ids are bad, not on what happens to them: scalar raises,
    batch zeroes the row, so one bad id doesn't cost the rest of the column."""
    with pytest.raises(ValueError):
        decode(cell_id)

    expected = (4, -6, 100, Measure.EDGE_LENGTH, Orientation.FLAT)
    # as uint64, and as plain Python ints -- ids at or above 2**63 fit neither int64 nor the float64
    # numpy would infer for such a list, so this also covers the list path's object fallback
    for cell_ids in (np.array([_VALID, cell_id], dtype=np.uint64), [_VALID, cell_id]):
        decoded = decode_batch(cell_ids)
        assert tuple(field[0] for field in decoded) == expected
        assert tuple(field[1] for field in decoded) == (0, 0, 0, 0, 0)


def test_decode_batch_zeroes_missing_values():
    """None/NaN in an object array, and Python ints too big for int64, are missing rather than errors."""
    cell_ids = np.array([_VALID, None, np.nan, 2**64, "junk"], dtype=object)
    size = decode_batch(cell_ids)[2]
    assert size.tolist() == [100, 0, 0, 0, 0]
    assert decode_batch([None, _VALID])[2].tolist() == [0, 100]


class _NullableColumn:
    """Stands in for a pandas `Int64` Series holding `<NA>`, without importing pandas: numpy only
    sees it as float64 (which has already rounded every id), and only `fillna` gets it to int64
    intact."""

    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None, copy=None):
        return np.array([np.nan if v is None else float(v) for v in self.values])

    def fillna(self, value):
        return np.array([value if v is None else v for v in self.values], dtype=np.int64)


def test_nullable_columns_are_filled_before_numpy_sees_them():
    """A pandas Int64 column holding <NA> is filled via `fillna` and stays integer, rather than
    going through the float64 numpy alone would make of it."""
    q, r, size, _, _ = decode_batch(_NullableColumn([_VALID, None]))

    assert size.tolist() == [100, 0]
    assert (q[0], r[0]) == (4, -6)


@pytest.mark.parametrize(
    "cell_ids",
    [
        np.array([float(_VALID), np.nan]),
        [float(_VALID), None],
        np.array([float(_VALID), None], dtype=object),
    ],
)
def test_float_ids_decode_exactly(cell_ids):
    """Every id is below 2**52, so one that has been through a float64 -- the column a NaN turns an
    int64 one into -- is still exactly itself, and decodes to the right cell."""
    q, r, size, _, _ = decode_batch(cell_ids)
    assert size.tolist() == [100, 0]
    assert (q[0], r[0]) == (4, -6)


def test_floats_that_cant_be_ids_are_missing():
    """Non-integral, infinite, or too large to be exact (which no valid id is): none of them is an
    id, so like any other invalid row they decode to zeros."""
    cell_ids = np.array([float(_VALID) + 0.5, np.inf, -np.inf, 2.0**60, 0.0])
    assert decode_batch(cell_ids)[2].tolist() == [0, 0, 0, 0, 0]
    assert decode_batch([])[2].tolist() == []


def test_cell_centre_batch_gives_nan_for_invalid_ids():
    cell_ids = [INVALID_CELL_ID, _VALID, None, _REJECTED[0]]

    x, y = batch.cell_centre_batch(cell_ids)
    lon, lat = cell_to_lonlat_batch(cell_ids)

    assert (x[1], y[1]) == centroid(_VALID).coords[0]
    assert (lon[1], lat[1]) == pytest.approx(centroid(_VALID, lonlat=True).coords[0])
    # NaN, not the inf pyproj makes of a NaN input
    for column in (x, y, lon, lat):
        assert np.isnan(column[[0, 2, 3]]).all()


def test_cell_centre_batch_on_nothing_valid_is_all_nan():
    x, y = batch.cell_centre_batch([INVALID_CELL_ID, None])
    assert np.isnan(x).all()
    assert np.isnan(y).all()


def test_invalid_ids_dont_count_towards_the_single_grid_rule():
    """A zeroed row reads as side_length 0 / POINTY, which must not look like a second grid."""
    batch.cell_centre_batch(
        [encode(1, 1, size=300, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT), INVALID_CELL_ID]
    )  # doesn't raise
    with pytest.raises(ValueError, match="single size"):
        batch.cell_centre_batch(
            [
                encode(1, 1, size=300, measure=Measure.EDGE_LENGTH),
                INVALID_CELL_ID,
                encode(1, 1, size=200, measure=Measure.EDGE_LENGTH),
            ]
        )


def test_decode_batch_accepts_an_empty_array():
    dq, dr, ds, dm, do = decode_batch(np.array([], dtype=np.uint64))
    assert len(dq) == len(dr) == len(ds) == len(do) == len(dm) == 0


def test_lonlat_to_cell_batch_matches_scalar():
    lats = np.array([51.5074, 55.9533, 51.4816])
    lons = np.array([-0.1278, -3.1883, -3.1791])
    side_length = 1000

    batch_ids = lonlat_to_cell_batch(
        lons, lats, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY
    )
    scalar_ids = [
        lonlat_to_cell(
            float(lon), float(lat), size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY
        )
        for lat, lon in zip(lats, lons, strict=True)
    ]

    assert list(batch_ids.astype(object)) == scalar_ids


def test_cell_to_lonlat_batch_matches_scalar():
    lats = np.array([51.5074, 55.9533, 51.4816])
    lons = np.array([-0.1278, -3.1883, -3.1791])
    side_length = 1000

    batch_ids = lonlat_to_cell_batch(
        lons, lats, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.POINTY
    )
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
        lonlat_to_cell_batch(lons, lats, size=500, measure=Measure.EDGE_LENGTH)


def test_lonlat_to_cell_batch_still_maps_nan_to_invalid():
    lats = np.array([51.5074, np.nan])
    lons = np.array([-0.1278, np.nan])
    ids = lonlat_to_cell_batch(lons, lats, size=500, measure=Measure.EDGE_LENGTH)
    assert ids[1] == 0


def test_bng_to_cell_batch_matches_scalar():
    rng = np.random.default_rng(1)
    x = rng.uniform(100_000, 600_000, size=200)
    y = rng.uniform(50_000, 900_000, size=200)
    side_length = 202

    batch_ids = bng_to_cell_batch(x, y, size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)
    scalar_ids = [
        bng_to_cell(float(xi), float(yi), size=side_length, measure=Measure.EDGE_LENGTH, orientation=Orientation.FLAT)
        for xi, yi in zip(x, y, strict=True)
    ]

    assert list(batch_ids.astype(object)) == scalar_ids


def test_bng_to_cell_batch_maps_nan_to_invalid():
    """Mirrors lonlat_to_cell_batch: a NaN coordinate is an absent point, not an encode error."""
    x = np.array([530034.0, np.nan])
    y = np.array([180381.0, np.nan])

    ids = bng_to_cell_batch(x, y, size=202, measure=Measure.EDGE_LENGTH)

    assert ids[0] == bng_to_cell(530034.0, 180381.0, size=202, measure=Measure.EDGE_LENGTH)
    assert ids[1] == 0


def test_bng_to_cell_batch_rejects_coordinates_beyond_the_bit_budget():
    """No area-of-use guard on the BNG path, but the q/r range check still catches absurd input."""
    with pytest.raises(ValueError, match="representable range"):
        bng_to_cell_batch(np.array([1e15]), np.array([1e15]), size=3, measure=Measure.EDGE_LENGTH)


def test_edge_to_edge_batch_paths_match_scalar():
    rng = np.random.default_rng(4)
    q = rng.integers(-1000, 1000, size=300)
    r = rng.integers(-1000, 1000, size=300)
    x = rng.uniform(100_000, 600_000, size=300)
    y = rng.uniform(50_000, 900_000, size=300)
    for orientation in Orientation:
        ids = encode_batch(q, r, orientation=orientation, size=433, measure=Measure.EDGE_TO_EDGE)
        assert list(ids.astype(object)) == [
            encode(int(qi), int(ri), orientation=orientation, size=433, measure=Measure.EDGE_TO_EDGE)
            for qi, ri in zip(q, r, strict=True)
        ]

        dq, dr, ds, dm, do = decode_batch(ids)
        assert (ds == 433).all() and (dm == Measure.EDGE_TO_EDGE).all()

        xc, yc = cell_centre_batch(ids)
        assert [(float(a), float(b)) for a, b in zip(xc, yc, strict=True)] == [centroid(int(i)).coords[0] for i in ids]

        bng_ids = bng_to_cell_batch(x, y, orientation=orientation, size=433, measure=Measure.EDGE_TO_EDGE)
        assert list(bng_ids.astype(object)) == [
            bng_to_cell(float(xi), float(yi), orientation=orientation, size=433, measure=Measure.EDGE_TO_EDGE)
            for xi, yi in zip(x, y, strict=True)
        ]

        lons, lats = np.array([-0.1278, -3.1883]), np.array([51.5074, 55.9533])
        ll_ids = lonlat_to_cell_batch(lons, lats, orientation=orientation, size=433, measure=Measure.EDGE_TO_EDGE)
        assert list(ll_ids.astype(object)) == [
            lonlat_to_cell(float(a), float(b), orientation=orientation, size=433, measure=Measure.EDGE_TO_EDGE)
            for a, b in zip(lons, lats, strict=True)
        ]


def test_edge_to_edge_is_a_different_grid_from_edge_length():
    """Same number, different measure: a coarser grid (sqrt(3)x smaller side), so different ids --
    not the side_length ids with a flag bit set."""
    x, y = np.array([530034.0]), np.array([180381.0])
    by_side = decode(int(bng_to_cell_batch(x, y, size=200, measure=Measure.EDGE_LENGTH)[0]))
    by_width = decode(int(bng_to_cell_batch(x, y, size=200, measure=Measure.EDGE_TO_EDGE)[0]))
    assert (by_side.q, by_side.r) != (by_width.q, by_width.r)


@pytest.mark.parametrize(
    "call",
    [
        lambda: encode_batch(np.array([0]), np.array([0]), size=100),  # ty: ignore[missing-argument]
        lambda: encode_batch(np.array([0]), np.array([0]), measure=Measure.EDGE_LENGTH),  # ty: ignore[missing-argument]
        lambda: bng_to_cell_batch([1.0], [1.0], size=100),  # ty: ignore[missing-argument]
        lambda: bng_to_cell_batch([1.0], [1.0], measure=Measure.EDGE_LENGTH),  # ty: ignore[missing-argument]
        lambda: lonlat_to_cell_batch([-0.1], [51.5], size=100),  # ty: ignore[missing-argument]
        lambda: lonlat_to_cell_batch([-0.1], [51.5], measure=Measure.EDGE_LENGTH),  # ty: ignore[missing-argument]
    ],
)
def test_batch_paths_require_size_and_measure(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize("measure", [1, Orientation.FLAT, "edge"])
@pytest.mark.parametrize("fn", [encode_batch, bng_to_cell_batch, lonlat_to_cell_batch])
def test_batch_paths_refuse_a_measure_that_is_not_a_measure(fn, measure):
    """1 and Orientation.FLAT both equal Measure.EDGE_TO_EDGE as ints, so converting would pass."""
    with pytest.raises(TypeError, match="measure must be a Measure"):
        fn(np.array([0]), np.array([0]), size=100, measure=measure)


def test_cell_centre_batch_rejects_mixed_measure():
    with pytest.raises(ValueError, match="single size measure"):
        cell_centre_batch(
            [encode(0, 0, size=100, measure=Measure.EDGE_LENGTH), encode(0, 0, size=100, measure=Measure.EDGE_TO_EDGE)]
        )


# --- deprecated lat-first spellings ------------------------------------------------------------


def test_deprecated_latlon_to_cell_batch_warns_and_takes_lats_first():
    lats, lons = [51.5074, 55.9533], [-0.1278, -3.1883]
    with pytest.deprecated_call(match="lonlat_to_cell_batch"):
        old = batch.latlon_to_cell_batch(lats, lons, size=500, measure=Measure.EDGE_LENGTH)  # ty: ignore[deprecated] -- the deprecated call is the point
    assert np.array_equal(old, lonlat_to_cell_batch(lons, lats, size=500, measure=Measure.EDGE_LENGTH))


def test_deprecated_cell_to_latlon_batch_warns_and_returns_lat_first():
    ids = lonlat_to_cell_batch([-0.1278, -3.1883], [51.5074, 55.9533], size=500, measure=Measure.EDGE_LENGTH)
    with pytest.deprecated_call(match="cell_to_lonlat_batch"):
        lat, lon = batch.cell_to_latlon_batch(ids)  # ty: ignore[deprecated] -- the deprecated call is the point
    new_lon, new_lat = cell_to_lonlat_batch(ids)
    assert np.array_equal(lat, new_lat)
    assert np.array_equal(lon, new_lon)
