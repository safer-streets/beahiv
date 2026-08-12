import random

import numpy as np
import pytest

from beahiv import CellIndex, Orientation
from beahiv.cell_id import (
    INVALID_CELL_ID,
    Q_OFFSET,
    R_OFFSET,
    RESERVED_SHIFT,
    SIDE_LENGTH_MASK,
    SIDE_LENGTH_MAX,
    SIDE_LENGTH_SHIFT,
)
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


def test_decode_morton_rejects_what_decode_rejects():
    """orientation/side_length/reserved sit in the same places in both layouts, so the two
    decoders reject the same ids -- only the q/r region is arranged differently."""
    valid = encode_morton(3, -2, 100, Orientation.POINTY)
    decode_morton(valid)  # doesn't raise

    for cell_id in [
        valid | (0b011 << RESERVED_SHIFT),
        valid & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT),
        valid & ~(SIDE_LENGTH_MASK << SIDE_LENGTH_SHIFT) | ((SIDE_LENGTH_MAX + 1) << SIDE_LENGTH_SHIFT),
        INVALID_CELL_ID,
    ]:
        with pytest.raises(ValueError):
            decode_morton(cell_id)


def test_encode_morton_and_decode_morton_accept_numpy_integers():
    cell_id = encode_morton(-100, -200, 100)

    assert encode_morton(np.int64(-100), np.int64(-200), np.uint64(100)) == cell_id
    assert decode_morton(np.int64(cell_id)) == CellIndex(-100, -200, 100, Orientation.FLAT)

    idx = decode_morton(np.uint64(cell_id))
    assert [type(v) for v in (idx.q, idx.r, idx.side_length)] == [int, int, int]
