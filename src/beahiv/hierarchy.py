"""Parent/child lookups between cells at 2x/0.5x side_length.

`axial_to_cartesian(q, r, side_length, ...)` is linear in (q, r), so a cell at `side_length` shares
its exact centroid with a cell at `2 * side_length` iff `(Q, R) = (q / 2, r / 2)` is itself integer
(i.e. `q` and `r` are both even), and with a cell at `side_length / 2` iff `side_length` is itself
even (the child is always `(2 * q, 2 * r)`, exactly -- doubling q/r is always an integer, unlike
halving them). Neither direction needs cube-coordinate rounding or a Cartesian round trip: the
same-centroid partner either exists as an exact integer solution, or it doesn't exist at all.

Singular and plural answer two different questions, and the name is the whole distinction:

- `get_parent`/`get_child` -- the same-centroid partner, or nothing. Pure parity arithmetic, no
  geometry: a cheap way to ask whether this cell happens to line up exactly with the 2x/0.5x grid.
  Most cells have no same-centroid parent, so `get_parent` returns None for them.
- `get_parents`/`get_children` -- every cell at the target side_length that overlaps this one, which
  always exists: 1 or 2 parents, always 7 children. This is the covering question, and the one most
  callers resizing by a factor of two actually mean.

Neither pair is a hierarchical tiling, and this module doesn't attempt one. The same-centroid
relation is partial (most cells have no parent), and the overlapping relation shares cells rather
than partitioning them -- a 1x cell straddling two 2x cells belongs to both, and the 7 children of
a cell cover it but overspill it by 75% of its area, hexes at half the side_length not being a
partition of it. See README's "Cell identifiers" section for why hex grids don't nest for arbitrary
integer ratios; for a covering at an arbitrary new side_length, see `polyfill.resize_cell`.

An odd side_length raises rather than returning nothing, unlike odd q/r: side lengths are integer
metres, so an odd one means the 0.5x grid doesn't exist at all and there is nothing to be inside --
a different failure from a cell that merely doesn't line up with a grid that does exist. Doubling
past `SIDE_LENGTH_MAX` likewise raises, from `encode`.

All four functions accept a scalar cell id or an array-like of ids, returning one scalar result or a
same-shape array of results, with `None`/tuple entries where the scalar branch would have had them.
"""

from typing import SupportsIndex, TypeGuard, cast, overload

import numpy as np
from numpy.typing import ArrayLike

from .cell_id import decode, encode
from .neighbours import k_ring


def _is_scalar_id(cell_id: SupportsIndex | ArrayLike) -> TypeGuard[SupportsIndex]:
    """`np.isscalar` as a type guard: a single cell id, rather than an array-like of them.

    Only the narrowing is added -- a type checker can't see through numpy's plain `bool` return, so
    the four public functions would otherwise have to cast on both branches instead of just the
    array one.
    """
    return np.isscalar(cell_id)


def _get_parent_scalar(cell_id: SupportsIndex) -> int | None:
    idx = decode(cell_id)
    if idx.q % 2 != 0 or idx.r % 2 != 0:
        return None
    return encode(idx.q // 2, idx.r // 2, idx.side_length * 2, idx.orientation)


def _get_parent_array(cell_ids: ArrayLike) -> np.ndarray:
    ids = np.asarray(cell_ids)
    out = np.empty(ids.shape, dtype=object)
    for index in np.ndindex(ids.shape):
        out[index] = _get_parent_scalar(ids[index])
    return out


def _get_parents_array(cell_ids: ArrayLike) -> np.ndarray:
    ids = np.asarray(cell_ids)
    out = np.empty(ids.shape, dtype=object)
    for index in np.ndindex(ids.shape):
        out[index] = _get_parents_scalar(ids[index])
    return out


def _get_parents_scalar(cell_id: SupportsIndex) -> tuple[int, ...]:
    idx = decode(cell_id)
    double = idx.side_length * 2
    match idx.q % 2, idx.r % 2:
        case 0, 0:
            return (encode(idx.q // 2, idx.r // 2, double, idx.orientation),)
        case 0, 1:
            return (
                encode(idx.q // 2, (idx.r - 1) // 2, double, idx.orientation),
                encode(idx.q // 2, (idx.r + 1) // 2, double, idx.orientation),
            )
        case 1, 0:
            return (
                encode(idx.q // 2, idx.r // 2, double, idx.orientation),
                encode(idx.q // 2 + 1, idx.r // 2, double, idx.orientation),
            )
        case _:  # 1, 1
            return (
                encode(idx.q // 2 + 1, idx.r // 2, double, idx.orientation),
                encode(idx.q // 2, idx.r // 2 + 1, double, idx.orientation),
            )


def _get_child_scalar(cell_id: SupportsIndex) -> int:
    idx = decode(cell_id)
    if idx.side_length % 2 != 0:
        raise ValueError(f"no 0.5x side_length cell shares this cell's centroid: side_length={idx.side_length} is odd")
    return encode(idx.q * 2, idx.r * 2, idx.side_length // 2, idx.orientation)


def _get_child_array(cell_ids: ArrayLike) -> np.ndarray:
    ids = np.asarray(cell_ids)
    out = np.empty(ids.shape, dtype=object)
    for index in np.ndindex(ids.shape):
        out[index] = _get_child_scalar(ids[index])
    return out


def _get_children_array(cell_ids: ArrayLike) -> np.ndarray:
    ids = np.asarray(cell_ids)
    out = np.empty(ids.shape, dtype=object)
    for index in np.ndindex(ids.shape):
        out[index] = _get_children_scalar(ids[index])
    return out


def _get_children_scalar(cell_id: SupportsIndex) -> tuple[int, ...]:
    return k_ring(_get_child_scalar(cell_id), 1)


@overload
def get_parent(cell_id: SupportsIndex) -> int | None: ...
@overload
def get_parent(cell_id: ArrayLike) -> np.ndarray: ...
def get_parent(cell_id: SupportsIndex | ArrayLike) -> int | None | np.ndarray:
    """Return the id of the cell at 2x side_length sharing this cell's exact centroid.

    Returns None if there is no such cell -- q and r must both be even. For the cells at 2x
    side_length that merely overlap this one, which always exist, see `get_parents`.

    Accepts either a single cell id or an array-like of ids; array input returns a same-shape
    object array with one result per id.

    Raises ValueError if doubling side_length would exceed SIDE_LENGTH_MAX.
    """
    if _is_scalar_id(cell_id):
        return _get_parent_scalar(cell_id)
    return _get_parent_array(cast("ArrayLike", cell_id))


@overload
def get_parents(cell_id: SupportsIndex) -> tuple[int, ...]: ...
@overload
def get_parents(cell_id: ArrayLike) -> np.ndarray: ...
def get_parents(cell_id: SupportsIndex | ArrayLike) -> tuple[int, ...] | np.ndarray:
    """Return the ids of every cell at 2x side_length overlapping this cell.

    Returns the 1 cell containing this one when q and r are both even -- a cell sharing a 2x cell's
    centroid lies wholly inside it -- and otherwise the 2 cells this one straddles. Never empty.

    Accepts either a single cell id or an array-like of ids; array input returns a same-shape
    object array with one tuple per id.

    Raises ValueError if doubling side_length would exceed SIDE_LENGTH_MAX.
    """
    if _is_scalar_id(cell_id):
        return _get_parents_scalar(cell_id)
    return _get_parents_array(cast("ArrayLike", cell_id))


@overload
def get_child(cell_id: SupportsIndex) -> int: ...
@overload
def get_child(cell_id: ArrayLike) -> np.ndarray: ...
def get_child(cell_id: SupportsIndex | ArrayLike) -> int | np.ndarray:
    """Return the id of the cell at side_length / 2 sharing this cell's exact centroid.

    Always exists when side_length is even, and is always `(2q, 2r)`.

    Accepts either a single cell id or an array-like of ids; array input returns a same-shape
    object array with one result per id.

    Raises ValueError if side_length is odd -- there is no 0.5x grid at all.
    """
    if _is_scalar_id(cell_id):
        return _get_child_scalar(cell_id)
    return _get_child_array(cast("ArrayLike", cell_id))


@overload
def get_children(cell_id: SupportsIndex) -> tuple[int, ...]: ...
@overload
def get_children(cell_id: ArrayLike) -> np.ndarray: ...
def get_children(cell_id: SupportsIndex | ArrayLike) -> tuple[int, ...] | np.ndarray:
    """Return the ids of every cell at side_length / 2 overlapping this cell.

    Always 7: the same-centroid `get_child` plus the 6 partially contained cells ringing it. Their
    union covers this cell, and overspills it by 75% of its area.

    Accepts either a single cell id or an array-like of ids; array input returns a same-shape
    object array with one tuple per id.

    Raises ValueError if side_length is odd -- there is no 0.5x grid at all.
    """
    if _is_scalar_id(cell_id):
        return _get_children_scalar(cell_id)
    return _get_children_array(cast("ArrayLike", cell_id))
