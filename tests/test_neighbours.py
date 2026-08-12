import random

import numpy as np

from beahiv import Orientation, decode, distance, encode, get_neighbours, k_ring
from beahiv.neighbours import NEIGHBOUR_OFFSETS


def _random_origin_cells(n, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        q = rng.randint(-1000, 1000)
        r = rng.randint(-1000, 1000)
        side_length = rng.randint(1, 5000)
        orientation = rng.choice(list(Orientation))
        yield encode(q, r, side_length, orientation)


def test_every_cell_has_six_neighbours():
    for cell_id in _random_origin_cells(200):
        assert len(get_neighbours(cell_id)) == 6


def test_neighbour_symmetry():
    for cell_id_a in _random_origin_cells(200, seed=1):
        for cell_id_b in get_neighbours(cell_id_a):
            assert cell_id_a in get_neighbours(cell_id_b)


def test_neighbour_offsets_are_distance_one():
    origin = encode(0, 0, 100, Orientation.POINTY)
    for cell_id in get_neighbours(origin):
        assert distance(origin, cell_id) == 1


def test_distance_is_symmetric_and_zero_for_self():
    for cell_id in _random_origin_cells(100, seed=2):
        assert distance(cell_id, cell_id) == 0

    rng = random.Random(3)
    cells = list(_random_origin_cells(50, seed=3))
    for cell_id in cells:
        idx = decode(cell_id)
        dq, dr = rng.choice(NEIGHBOUR_OFFSETS)
        other = encode(idx.q + dq, idx.r + dr, idx.side_length, idx.orientation)
        assert distance(cell_id, other) == distance(other, cell_id)


def test_k_ring_counts_match_hex_formula():
    origin = encode(0, 0, 100, Orientation.POINTY)
    for k in range(0, 6):
        ring = k_ring(origin, k)
        assert len(set(ring)) == len(ring)  # no duplicates
        assert len(ring) == 1 + 3 * k * (k + 1)


def test_k_ring_cells_are_within_k_hops():
    origin = encode(5, -3, 250, Orientation.FLAT)
    k = 4
    for cell_id in k_ring(origin, k):
        assert distance(origin, cell_id) <= k


def test_ring_operations_accept_numpy_integer_ids():
    """Regression: k_ring on an np.int64 id raised OverflowError, encode's `& UINT64_MASK` not
    fitting an int64 operand."""
    origin = encode(5, -3, 250)

    assert k_ring(np.int64(origin), 2) == k_ring(origin, 2)
    assert get_neighbours(np.int64(origin)) == get_neighbours(origin)
    assert distance(np.int64(origin), np.uint64(origin)) == 0
