import random

import numpy as np
import pytest
from shapely.ops import unary_union

from beahiv import Orientation, cell_polygon, decode, encode, get_child, get_children, get_parent, get_parents
from beahiv.cell_id import SIDE_LENGTH_MAX
from beahiv.coords import axial_to_cartesian

# Two cells sharing an edge produce a sliver intersection of ~1e-29 of a cell's area out of
# shapely's floating-point arithmetic. "Overlaps" here means a real, positive-area overlap, so
# these tests measure against a fraction of the cell's own area rather than against zero.
OVERLAP_TOL = 1e-12


def _random_cells(n, seed=0, side_lengths=None):
    rng = random.Random(seed)
    for _ in range(n):
        q = rng.randint(-1000, 1000)
        r = rng.randint(-1000, 1000)
        side_length = rng.choice(side_lengths) if side_lengths else rng.randint(1, 5000)
        orientation = rng.choice(list(Orientation))
        yield encode(q, r, side_length, orientation)


def _overlapping(cell_id, side_length, search=3):
    """Brute-force every cell at side_length whose overlap with cell_id has real area."""
    idx = decode(cell_id)
    cell = cell_polygon(cell_id)
    # Search around the target-grid cell nearest this one, which is (q/2, r/2) going coarser
    # and (2q, 2r) going finer -- the same starting point the implementation uses.
    q0, r0 = (idx.q // 2, idx.r // 2) if side_length > idx.side_length else (idx.q * 2, idx.r * 2)
    found = set()
    for dq in range(-search, search + 1):
        for dr in range(-search, search + 1):
            other = encode(q0 + dq, r0 + dr, side_length, idx.orientation)
            if cell.intersection(cell_polygon(other)).area > OVERLAP_TOL * cell.area:
                found.add(other)
    return found


def _uncovered_fraction(cell_id, cell_ids):
    cell = cell_polygon(cell_id)
    union = unary_union([cell_polygon(c) for c in cell_ids])
    return cell.difference(union).area / cell.area


# --- get_parent / get_child: the same-centroid partner --------------------------------------


def test_get_parent_requires_even_q_and_r():
    assert get_parent(encode(4, -6, 100)) is not None
    for q, r in [(3, -6), (4, -5), (3, -5)]:
        assert get_parent(encode(q, r, 100)) is None


def test_get_child_requires_even_side_length():
    get_child(encode(0, 0, 100))  # doesn't raise
    with pytest.raises(ValueError):
        get_child(encode(0, 0, 101))


def test_get_parent_doubles_side_length_and_shares_centroid():
    for cell_id in _random_cells(200, side_lengths=[2, 4, 50, 202]):
        idx = decode(cell_id)
        parent = get_parent(cell_id)
        if idx.q % 2 != 0 or idx.r % 2 != 0:
            assert parent is None
            continue
        assert parent is not None
        parent_idx = decode(parent)

        assert parent_idx.side_length == idx.side_length * 2
        assert parent_idx.orientation == idx.orientation
        assert (parent_idx.q, parent_idx.r) == (idx.q // 2, idx.r // 2)
        assert axial_to_cartesian(idx.q, idx.r, idx.side_length, idx.orientation) == axial_to_cartesian(
            parent_idx.q, parent_idx.r, parent_idx.side_length, parent_idx.orientation
        )


def test_get_child_halves_side_length_and_shares_centroid():
    for cell_id in _random_cells(200, seed=1, side_lengths=[2, 4, 50, 202]):
        idx = decode(cell_id)
        child = get_child(cell_id)
        child_idx = decode(child)

        assert child_idx.side_length == idx.side_length // 2
        assert child_idx.orientation == idx.orientation
        assert (child_idx.q, child_idx.r) == (idx.q * 2, idx.r * 2)
        assert axial_to_cartesian(idx.q, idx.r, idx.side_length, idx.orientation) == axial_to_cartesian(
            child_idx.q, child_idx.r, child_idx.side_length, child_idx.orientation
        )


def test_get_parent_and_get_child_are_inverses_when_both_are_defined():
    for cell_id in _random_cells(200, seed=2, side_lengths=[2, 4, 50, 202]):
        parent = get_parent(cell_id)
        if parent is None:
            continue
        assert get_child(parent) == cell_id


def test_get_parent_rejects_side_length_that_would_exceed_the_cap():
    with pytest.raises(ValueError):
        get_parent(encode(0, 0, SIDE_LENGTH_MAX))


# --- get_parents / get_children: every overlapping cell -------------------------------------


def test_get_parents_returns_the_containing_cell_alone_when_q_and_r_are_even():
    cell_id = encode(4, -6, 100)
    assert get_parents(cell_id) == (get_parent(cell_id),)


def test_get_parents_returns_two_distinct_straddled_cells_otherwise():
    for q, r in [(3, -6), (4, -5), (3, -5)]:
        parents = get_parents(encode(q, r, 100))
        assert len(set(parents)) == 2


def test_get_parents_is_exactly_the_set_of_overlapping_2x_cells():
    for cell_id in _random_cells(60, seed=3, side_lengths=[2, 50, 200]):
        idx = decode(cell_id)
        assert set(get_parents(cell_id)) == _overlapping(cell_id, idx.side_length * 2)


def test_get_parents_cover_the_cell():
    for cell_id in _random_cells(60, seed=4, side_lengths=[2, 50, 200]):
        assert _uncovered_fraction(cell_id, get_parents(cell_id)) <= OVERLAP_TOL


def test_get_parents_are_at_2x_side_length_and_the_same_orientation():
    for cell_id in _random_cells(100, seed=5, side_lengths=[2, 50, 200]):
        idx = decode(cell_id)
        for parent in get_parents(cell_id):
            parent_idx = decode(parent)
            assert parent_idx.side_length == idx.side_length * 2
            assert parent_idx.orientation == idx.orientation


def test_get_parents_rejects_side_length_that_would_exceed_the_cap():
    with pytest.raises(ValueError):
        get_parents(encode(3, -6, SIDE_LENGTH_MAX))


def test_get_children_returns_seven_including_the_same_centroid_child():
    cell_id = encode(4, -6, 100)
    children = get_children(cell_id)
    assert len(set(children)) == 7
    assert get_child(cell_id) in children


def test_get_children_is_exactly_the_set_of_overlapping_half_size_cells():
    for cell_id in _random_cells(60, seed=6, side_lengths=[2, 50, 200]):
        idx = decode(cell_id)
        assert set(get_children(cell_id)) == _overlapping(cell_id, idx.side_length // 2)


def test_get_children_cover_the_cell_and_overspill_it():
    for cell_id in _random_cells(30, seed=7, side_lengths=[2, 50, 200]):
        children = get_children(cell_id)
        assert _uncovered_fraction(cell_id, children) <= OVERLAP_TOL

        cell = cell_polygon(cell_id)
        union = unary_union([cell_polygon(c) for c in children])
        # 7 disjoint quarter-area cells span 1.75x the cell, so 75% of its area lies outside it.
        assert union.difference(cell).area == pytest.approx(0.75 * cell.area, rel=1e-9)


def test_get_children_requires_even_side_length():
    get_children(encode(0, 0, 100))  # doesn't raise
    with pytest.raises(ValueError):
        get_children(encode(0, 0, 101))


def test_lookups_accept_numpy_integer_ids():
    cell_id = encode(4, -6, 100)

    assert get_parent(np.int64(cell_id)) == get_parent(cell_id)
    assert get_parents(np.int64(cell_id)) == get_parents(cell_id)
    assert get_child(np.int64(cell_id)) == get_child(cell_id)
    assert get_children(np.int64(cell_id)) == get_children(cell_id)


@pytest.mark.parametrize("lookup", [get_parent, get_parents, get_child, get_children])
def test_hierarchy_lookups_accept_array_like_ids(lookup):
    # (4, -6) has a same-centroid parent, (3, -6) doesn't, so get_parent gives an id and a None.
    cell_ids = np.array([encode(4, -6, 100), encode(3, -6, 100), encode(0, 0, 100)], dtype=np.uint64)

    result = lookup(cell_ids)

    assert isinstance(result, np.ndarray)
    assert result.dtype == object
    assert result.shape == cell_ids.shape
    # Compared as a list, not with np.array_equal against a np.array of the expected entries: the
    # equal-length tuples from get_children would collapse into a 2-D array of ids, which is not
    # what these functions return (and never compares equal to the 1-D array of tuples they do).
    assert result.tolist() == [lookup(cell_id) for cell_id in cell_ids]

    # A plain (nested) list is array-like too, and the shape is whatever was handed in.
    assert lookup(cell_ids.tolist()).tolist() == result.tolist()
    assert lookup(cell_ids[:2].reshape(2, 1)).tolist() == [[result[0]], [result[1]]]
