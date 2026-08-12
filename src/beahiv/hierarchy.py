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

All four are scalar. There are no vectorised forms: bulk same-centroid lookups aren't a useful
operation (see `resize_cell` for the useful, covering-based replacement), and these stay as cheap
single-cell queries.
"""

from typing import SupportsIndex

from .cell_id import decode, encode
from .neighbours import k_ring


def get_parent(cell_id: SupportsIndex) -> int | None:
    """Return the id of the cell at 2x side_length sharing this cell's exact centroid.

    Returns None if there is no such cell -- q and r must both be even. For the cells at 2x
    side_length that merely overlap this one, which always exist, see `get_parents`.

    Raises ValueError if doubling side_length would exceed SIDE_LENGTH_MAX.
    """
    idx = decode(cell_id)
    if idx.q % 2 != 0 or idx.r % 2 != 0:
        return None
    return encode(idx.q // 2, idx.r // 2, idx.side_length * 2, idx.orientation)


def get_parents(cell_id: SupportsIndex) -> tuple[int, ...]:
    """Return the ids of every cell at 2x side_length overlapping this cell.

    Returns the 1 cell containing this one when q and r are both even -- a cell sharing a 2x cell's
    centroid lies wholly inside it -- and otherwise the 2 cells this one straddles. Never empty.

    Raises ValueError if doubling side_length would exceed SIDE_LENGTH_MAX.
    """
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


def get_child(cell_id: SupportsIndex) -> int:
    """Return the id of the cell at side_length / 2 sharing this cell's exact centroid.

    Always exists when side_length is even, and is always `(2q, 2r)`.

    Raises ValueError if side_length is odd -- there is no 0.5x grid at all.
    """
    idx = decode(cell_id)
    if idx.side_length % 2 != 0:
        raise ValueError(f"no 0.5x side_length cell shares this cell's centroid: side_length={idx.side_length} is odd")
    return encode(idx.q * 2, idx.r * 2, idx.side_length // 2, idx.orientation)


def get_children(cell_id: SupportsIndex) -> tuple[int, ...]:
    """Return the ids of every cell at side_length / 2 overlapping this cell.

    Always 7: the same-centroid `get_child` plus the 6 partially contained cells ringing it. Their
    union covers this cell, and overspills it by 75% of its area.

    Raises ValueError if side_length is odd -- there is no 0.5x grid at all.
    """
    return k_ring(get_child(cell_id), 1)
