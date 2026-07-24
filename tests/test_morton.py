import random

from beahiv import CellIndex, Orientation
from beahiv.cell_id import Q_OFFSET, R_OFFSET, SIDE_LENGTH_MAX
from beahiv.morton import decode_morton, encode_morton


def test_morton_round_trip():
    rng = random.Random(0)
    for _ in range(2000):
        q = rng.randint(-(Q_OFFSET - 1), Q_OFFSET - 1)
        r = rng.randint(-(R_OFFSET - 1), R_OFFSET - 1)
        side_length = rng.randint(1, SIDE_LENGTH_MAX)
        orientation = rng.choice(list(Orientation))

        cell_id = encode_morton(q, r, side_length, orientation)
        assert decode_morton(cell_id) == CellIndex(q, r, side_length, orientation)
        assert 0 <= cell_id < 2**64


def test_morton_ids_differ_from_plain_ids_for_same_cell():
    from beahiv import encode

    plain = encode(3, -2, 100, Orientation.POINTY)
    morton = encode_morton(3, -2, 100, Orientation.POINTY)
    # Same logical cell, different bit arrangement (unless q/r happen to be 0).
    assert plain != morton
