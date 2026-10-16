import math
import random

import numpy as np
import pytest

from beahiv import CellIndex, Orientation, SizeMeasure, decode, encode
from beahiv.batch import decode_batch
from beahiv.cell_id import (
    FLOAT_EXACT_BITS,
    INVALID_CELL_ID,
    MEASURE_MASK,
    MEASURE_SHIFT,
    Q_MASK,
    Q_OFFSET,
    R_MASK,
    R_OFFSET,
    RESERVED_SHIFT,
    SIDE_LENGTH_MASK,
    SIDE_LENGTH_MAX,
    SIDE_LENGTH_MIN,
    SIDE_LENGTH_SHIFT,
    SIZE_MANTISSA_BITS,
    SIZE_MIN,
    pack_size,
    unpack_size,
)
from beahiv.coords import cartesian_to_axial

from ._sizes import REPRESENTABLE_SIZES, random_size, sizes

Q_MIN, Q_MAX = -Q_OFFSET, Q_MASK - Q_OFFSET
R_MIN, R_MAX = -R_OFFSET, R_MASK - R_OFFSET


def _random_cells(n, seed=0, measure=SizeMeasure.SIDE_LENGTH):
    rng = random.Random(seed)
    for _ in range(n):
        q = rng.randint(Q_MIN, Q_MAX)
        r = rng.randint(R_MIN, R_MAX)
        orientation = rng.choice(list(Orientation))
        yield q, r, random_size(rng, measure), orientation


def _encode(q, r, size, measure, orientation):
    if measure == SizeMeasure.SIDE_LENGTH:
        return encode(q, r, side_length=size, orientation=orientation)
    return encode(q, r, side_to_side=size, orientation=orientation)


@pytest.mark.parametrize("measure", list(SizeMeasure))
def test_round_trip_random_coordinates(measure):
    for q, r, size, orientation in _random_cells(2000, measure=measure):
        cell_id = _encode(q, r, size, measure, orientation)
        assert decode(cell_id) == CellIndex(q, r, size, measure, orientation)


@pytest.mark.parametrize("measure", list(SizeMeasure))
def test_smallest_cell_indexes_the_national_grid(measure):
    """The q/r fields and offsets are sized for this box at SIDE_LENGTH_MIN, under both orientations
    -- the guarantee that every cell in GB is encodable at every allowed size.

    The National Grid runs Easting 0..700km, Northing 0..1300km.
    """
    corners = [(0, 0), (700_000, 0), (0, 1_300_000), (700_000, 1_300_000), (350_000, 650_000)]
    size = SIZE_MIN[measure]
    side_length = size if measure == SizeMeasure.SIDE_LENGTH else size / math.sqrt(3)
    assert side_length >= SIDE_LENGTH_MIN
    for orientation in Orientation:
        for x, y in corners:
            q, r = cartesian_to_axial(x, y, side_length, orientation)
            cell_id = _encode(q, r, size, measure, orientation)
            assert decode(cell_id) == CellIndex(q, r, size, measure, orientation)


def test_round_trip_extremes():
    for measure in SizeMeasure:
        for orientation in Orientation:
            for size in (SIZE_MIN[measure], SIDE_LENGTH_MAX):
                for q, r in ((0, 0), (Q_MAX, R_MAX), (Q_MIN, R_MIN), (Q_MIN, R_MAX), (Q_MAX, R_MIN)):
                    cell_id = _encode(q, r, size, measure, orientation)
                    assert decode(cell_id) == CellIndex(q, r, size, measure, orientation)


def test_every_id_is_exact_as_a_float64():
    """The point of the layout: an id that goes through a float64 -- a pandas int64 column that
    gains a NaN, a JavaScript JSON.parse -- comes back as the same id, not a rounded one that still
    decodes to some other valid cell."""
    assert 2**RESERVED_SHIFT <= 2**FLOAT_EXACT_BITS
    for measure in SizeMeasure:
        for q, r, size, orientation in _random_cells(2000, seed=1, measure=measure):
            cell_id = _encode(q, r, size, measure, orientation)
            assert 0 < cell_id < 2**RESERVED_SHIFT
            assert int(float(cell_id)) == cell_id

    # the largest id the fields allow, not just a sample of them
    largest = _encode(Q_MAX, R_MAX, SIDE_LENGTH_MAX, SizeMeasure.SIDE_TO_SIDE, Orientation.FLAT)
    assert int(float(largest)) == largest


def test_float_ids_decode_exactly_in_a_batch():
    """The batch path accepts the float column a NaN turns an id column into."""
    cell_ids = [_encode(q, r, s, SizeMeasure.SIDE_LENGTH, o) for q, r, s, o in _random_cells(500, seed=2)]
    as_floats = np.array([*map(float, cell_ids), np.nan])

    assert decode_batch(as_floats)[2][-1] == 0
    for field, expected in zip(decode_batch(as_floats), decode_batch(cell_ids), strict=True):
        assert field[:-1].tolist() == expected.tolist()


def test_size_packing_round_trips_every_representable_size():
    """Exhaustive: 3,699 sizes. Each packs to its own field (no two encodings of one size, which would
    give one cell two ids), and unpacks to itself."""
    fields = {pack_size(size, SizeMeasure.SIDE_TO_SIDE): size for size in sizes(SizeMeasure.SIDE_TO_SIDE)}
    assert len(fields) == len(sizes(SizeMeasure.SIDE_TO_SIDE))
    assert all(unpack_size(field) == size for field, size in fields.items())


@pytest.mark.parametrize("size", [3, 999, 1_000, 1_500, 14_500, 200_000, 500_000, SIDE_LENGTH_MAX])
def test_sizes_with_three_significant_digits_encode(size):
    assert decode(encode(0, 0, side_length=size)).size == size


@pytest.mark.parametrize("size", [1_001, 1_234, 14_550, SIDE_LENGTH_MAX + 1, 1_000_000])
def test_sizes_needing_more_than_three_significant_digits_raise(size):
    with pytest.raises(ValueError):
        encode(0, 0, side_length=size)


@pytest.mark.parametrize("measure", list(SizeMeasure))
def test_sizes_below_the_minimum_raise(measure):
    """Smaller cells would overflow q/r over parts of GB -- rejected outright, not just at the edges."""
    _encode(0, 0, SIZE_MIN[measure], measure, Orientation.FLAT)  # doesn't raise
    for size in range(SIZE_MIN[measure]):
        with pytest.raises(ValueError):
            _encode(0, 0, size, measure, Orientation.FLAT)


def test_size_minimum_is_one_geometric_side_length_for_both_measures():
    assert SIZE_MIN[SizeMeasure.SIDE_LENGTH] == SIDE_LENGTH_MIN
    assert SIZE_MIN[SizeMeasure.SIDE_TO_SIDE] / math.sqrt(3) >= SIDE_LENGTH_MIN
    assert (SIZE_MIN[SizeMeasure.SIDE_TO_SIDE] - 1) / math.sqrt(3) < SIDE_LENGTH_MIN


def _with_size_field(cell_id, field):
    return cell_id & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT) | (field << SIDE_LENGTH_SHIFT)


@pytest.mark.parametrize(
    ("exponent", "mantissa"),
    [
        (0, 0),  # size 0
        (1, 50),  # 500m written as 50e1: non-canonical, 500e0 is the encoding
        (3, 99),
        (0, 1000),  # mantissa past 999, inside its 10-bit field
        (2, 1023),
        (0, 1),  # a real size, but below SIDE_LENGTH_MIN
    ],
)
def test_decode_rejects_size_fields_encode_would_not_produce(exponent, mantissa):
    """The size field is the one with bit patterns that aren't valid values, so decode has to look."""
    cell_id = _with_size_field(encode(4, -6, side_length=100), (exponent << SIZE_MANTISSA_BITS) | mantissa)
    with pytest.raises(ValueError, match="size"):
        decode(cell_id)
    assert decode_batch([cell_id])[2].tolist() == [0]


@pytest.mark.parametrize("bit", [RESERVED_SHIFT, RESERVED_SHIFT + 1, 60, 62, 63])
def test_decode_rejects_ids_with_reserved_bits_set(bit):
    """encode never sets them, so decode must never accept them. Bit 52 is what keeps a future flag
    free to mean something; bits above it are what keep an id exact as a float."""
    cell_id = encode(4, -6, side_length=100)
    with pytest.raises(ValueError, match="reserved bits"):
        decode(cell_id | (1 << bit))
    assert decode_batch(np.array([cell_id | (1 << bit)], dtype=np.uint64))[2].tolist() == [0]


def test_decode_rejects_ids_from_the_previous_64_bit_layout():
    """Ids stored under the old layout must fail, not decode to some other cell. Every FLAT one had
    bit 60 set, and every POINTY one with a side of 512m or more bit 52 or above."""
    old_layout = {
        "FLAT, 200m": (1 << 60) | (200 << 43) | ((1 << 20) << 22) | (1 << 21),
        "POINTY, 1000m": (1000 << 43) | ((1 << 20) << 22) | (1 << 21),
    }
    for cell_id in old_layout.values():
        with pytest.raises(ValueError, match="reserved bits"):
            decode(cell_id)


def test_decode_rejects_negative_ids():
    with pytest.raises(ValueError):
        decode(-1)
    with pytest.raises(ValueError):
        decode(2**64)


def test_decode_rejects_the_invalid_cell_id_sentinel():
    """The *_to_cell functions emit INVALID_CELL_ID for missing input, so it reaches decode by
    ordinary use -- it must fail loudly rather than decode to plausible-looking coordinates."""
    with pytest.raises(ValueError, match="INVALID_CELL_ID"):
        decode(INVALID_CELL_ID)


@pytest.mark.parametrize(
    ("q", "r"),
    [(Q_MAX + 1, 0), (Q_MIN - 1, 0), (0, R_MAX + 1), (0, R_MIN - 1)],
)
def test_encode_rejects_out_of_range_q_r(q, r):
    with pytest.raises(ValueError, match="representable range"):
        encode(q, r, side_length=100)


def test_decode_accepts_exactly_what_encode_produces():
    for q, r, size, orientation in _random_cells(500, seed=9):
        cell_id = encode(q, r, side_length=size, orientation=orientation)
        assert decode(cell_id).encode() == cell_id


def test_encode_and_decode_accept_numpy_integers():
    """Ids reach the scalar API as numpy integers by ordinary use (an element of a decode_batch
    result, a pandas column, a DuckDB BIGINT), and numpy's fixed-width arithmetic gets this bit
    layout wrong unless they're coerced -- see the cell_id module docstring."""
    cell_id = encode(-100, -200, side_length=100)

    assert encode(np.int64(-100), np.int64(-200), side_length=np.uint64(100)) == cell_id
    assert type(encode(np.int64(-100), np.int64(-200), side_length=np.uint64(100))) is int

    for dtype in (np.int64, np.uint64):
        idx = decode(dtype(cell_id))
        assert idx == CellIndex(-100, -200, 100, SizeMeasure.SIDE_LENGTH, Orientation.FLAT)
        # equality alone wouldn't catch a numpy scalar leaking through; the arithmetic downstream
        # (neighbours, hierarchy) is what breaks on one, not the comparison here
        assert [type(v) for v in (idx.q, idx.r, idx.side_length)] == [int, int, int]


def test_measure_bit_is_the_only_difference_between_the_measures():
    for q, r, size, orientation in _random_cells(500, seed=5, measure=SizeMeasure.SIDE_TO_SIDE):
        by_side = encode(q, r, side_length=size, orientation=orientation)
        by_width = encode(q, r, orientation=orientation, side_to_side=size)
        assert (by_side >> MEASURE_SHIFT) & MEASURE_MASK == SizeMeasure.SIDE_LENGTH == 0
        assert by_width == by_side | (1 << MEASURE_SHIFT)


def test_encode_requires_exactly_one_size():
    with pytest.raises(TypeError, match="not both"):
        encode(0, 0, side_length=100, side_to_side=100)
    with pytest.raises(TypeError, match="size is required"):
        encode(0, 0)


def test_side_to_side_has_the_same_cap_as_side_length():
    encode(0, 0, side_to_side=SIDE_LENGTH_MAX)
    with pytest.raises(ValueError):
        encode(0, 0, side_to_side=SIDE_LENGTH_MAX + 1)


def test_cell_index_side_length_is_the_geometric_side():
    by_side = decode(encode(3, -2, side_length=200))
    assert by_side.size == 200
    assert by_side.side_length == 200
    assert type(by_side.side_length) is int

    by_width = decode(encode(3, -2, side_to_side=200))
    assert by_width.size == 200
    assert by_width.side_length == pytest.approx(200 / math.sqrt(3), rel=1e-15)


def test_cell_index_methods_keep_the_measure():
    idx = decode(encode(3, -2, side_to_side=200, orientation=Orientation.POINTY))
    assert idx.encode() == encode(3, -2, side_to_side=200, orientation=Orientation.POINTY)
    assert idx.dq(1) == CellIndex(4, -2, 200, SizeMeasure.SIDE_TO_SIDE, Orientation.POINTY)
    assert idx.dr(-1) == CellIndex(3, -3, 200, SizeMeasure.SIDE_TO_SIDE, Orientation.POINTY)


def test_encode_size_cannot_be_passed_positionally():
    with pytest.raises(TypeError, match="positional"):
        encode(0, 0, 100)  # ty: ignore[too-many-positional-arguments]


def test_representable_sizes_helper_matches_the_layout():
    """`tests/_sizes.py` enumerates sizes independently of pack_size; check it agrees."""
    assert len(REPRESENTABLE_SIZES) == 999 + 3 * 900
    assert REPRESENTABLE_SIZES[-1] == SIDE_LENGTH_MAX
