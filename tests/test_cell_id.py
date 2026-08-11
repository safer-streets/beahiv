import random

import pytest

from beahiv import CellIndex, Orientation, decode, encode
from beahiv.cell_id import (
    INVALID_CELL_ID,
    Q_OFFSET,
    R_OFFSET,
    RESERVED_MASK,
    RESERVED_SHIFT,
    SIDE_LENGTH_MASK,
    SIDE_LENGTH_MAX,
    SIDE_LENGTH_SHIFT,
)
from beahiv.coords import cartesian_to_axial


def _random_cells(n, seed=0):
    rng = random.Random(seed)
    q_max = Q_OFFSET - 1
    r_max = R_OFFSET - 1
    for _ in range(n):
        q = rng.randint(-q_max, q_max)
        r = rng.randint(-r_max, r_max)
        side_length = rng.randint(1, SIDE_LENGTH_MAX)
        orientation = rng.choice(list(Orientation))
        yield q, r, side_length, orientation


def test_round_trip_random_coordinates():
    for q, r, side_length, orientation in _random_cells(2000):
        cell_id = encode(q, r, side_length, orientation)
        assert decode(cell_id) == CellIndex(q, r, side_length, orientation)


def test_one_metre_side_length_indexes_full_gb_extent():
    """The finest practical resolution (s=1m) must still index the whole
    British National Grid without hitting the q/r bit ceiling.

    GB's nominal grid extent is Easting 0..700,000m, Northing 0..1,300,000m.
    Q_BITS/R_BITS were sized to cover this at s=1m with headroom, not just
    at the 25m floor the spec anticipated.
    """
    corners = [
        (0, 0),
        (700_000, 0),
        (0, 1_300_000),
        (700_000, 1_300_000),
        (350_000, 650_000),
    ]
    for orientation in Orientation:
        for x, y in corners:
            q, r = cartesian_to_axial(x, y, 1, orientation)
            cell_id = encode(q, r, 1, orientation)
            assert decode(cell_id) == CellIndex(q, r, 1, orientation)


def test_round_trip_extremes():
    for orientation in Orientation:
        for side_length in (1, SIDE_LENGTH_MAX):
            for q, r in ((0, 0), (Q_OFFSET - 1, R_OFFSET - 1), (-Q_OFFSET, -R_OFFSET)):
                cell_id = encode(q, r, side_length, orientation)
                assert decode(cell_id) == CellIndex(q, r, side_length, orientation)


def test_cell_id_fits_uint64():
    for q, r, side_length, orientation in _random_cells(500, seed=1):
        cell_id = encode(q, r, side_length, orientation)
        assert 0 <= cell_id < 2**64


def test_reserved_bits_are_never_set():
    for q, r, side_length, orientation in _random_cells(500, seed=2):
        cell_id = encode(q, r, side_length, orientation)
        assert (cell_id >> RESERVED_SHIFT) & RESERVED_MASK == 0


def test_cell_id_fits_signed_int64():
    """The reserved bits sit at the top, so ids stay positive in an int64 column.

    This is what lets consumers store ids as a plain BIGINT rather than an
    unsigned type or a hex string.
    """
    for q, r, side_length, orientation in _random_cells(500, seed=3):
        cell_id = encode(q, r, side_length, orientation)
        assert 0 <= cell_id < 2**63


def test_side_length_cap_is_below_its_field_capacity():
    """A side_length above the cap but inside the 17-bit field must still raise.

    Nothing masks it off, so were it encoded it would decode back intact and
    look valid -- it is only the cap that keeps 100km the coarsest grid.
    """
    assert SIDE_LENGTH_MAX < SIDE_LENGTH_MASK
    with pytest.raises(ValueError):
        encode(0, 0, SIDE_LENGTH_MASK)


@pytest.mark.parametrize(
    "q, r, side_length",
    [
        (Q_OFFSET, 0, 100),
        (-Q_OFFSET - 1, 0, 100),
        (0, R_OFFSET, 100),
        (0, -R_OFFSET - 1, 100),
        (0, 0, 0),
        (0, 0, SIDE_LENGTH_MAX + 1),
    ],
)
def test_encode_rejects_out_of_range_values(q, r, side_length):
    with pytest.raises(ValueError):
        encode(q, r, side_length)


def test_decode_rejects_non_uint64():
    with pytest.raises(ValueError):
        decode(-1)
    with pytest.raises(ValueError):
        decode(2**64)


def test_decode_rejects_ids_with_reserved_bits_set():
    """encode never sets them (see test_reserved_bits_are_never_set), so decode must never
    accept them -- that is what keeps the field free to mean something later."""
    cell_id = encode(4, -6, 100)
    decode(cell_id)  # doesn't raise
    for pattern in range(1, RESERVED_MASK + 1):
        with pytest.raises(ValueError, match="reserved bits"):
            decode(cell_id | (pattern << RESERVED_SHIFT))


@pytest.mark.parametrize("side_length", [0, SIDE_LENGTH_MAX + 1, SIDE_LENGTH_MASK])
def test_decode_rejects_side_lengths_encode_would_have_rejected(side_length):
    """side_length is the one field whose limit is a range check, not its bit width, so a
    decodable bit pattern is not necessarily an encodable value."""
    cell_id = encode(4, -6, 100) & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT) | (side_length << SIDE_LENGTH_SHIFT)
    with pytest.raises(ValueError, match="side_length"):
        decode(cell_id)


def test_decode_rejects_the_invalid_cell_id_sentinel():
    """The *_to_cell functions emit INVALID_CELL_ID for missing input, so it reaches decode by
    ordinary use -- it must fail loudly rather than decode to plausible-looking coordinates."""
    with pytest.raises(ValueError, match="INVALID_CELL_ID"):
        decode(INVALID_CELL_ID)


def test_decode_accepts_exactly_what_encode_produces():
    for q, r, side_length, orientation in _random_cells(500, seed=9):
        assert decode(encode(q, r, side_length, orientation)).encode() == encode(q, r, side_length, orientation)
