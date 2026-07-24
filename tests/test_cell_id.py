import random

import pytest

from beahiv import CellIndex, Orientation, decode, encode
from beahiv.cell_id import Q_OFFSET, R_OFFSET, SIDE_LENGTH_MAX
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
