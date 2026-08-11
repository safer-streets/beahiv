"""Neighbour/distance/ring operations. Pure coordinate arithmetic -- no
geometry is constructed."""

from .cell_id import decode, encode

NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
)


def get_neighbours(cell_id: int) -> tuple[int, ...]:
    """Return the six neighbouring cell ids, same side_length/orientation."""
    idx = decode(cell_id)
    return tuple(encode(idx.q + dq, idx.r + dr, idx.side_length, idx.orientation) for dq, dr in NEIGHBOUR_OFFSETS)


def distance(cell_a: int, cell_b: int) -> int:
    """Hex grid distance (number of hops) between two cells."""
    a = decode(cell_a)
    b = decode(cell_b)
    if a.side_length != b.side_length or a.orientation != b.orientation:
        raise ValueError("distance requires both cells to share side_length and orientation")

    ax, az, ay = a.q, a.r, -a.q - a.r
    bx, bz, by = b.q, b.r, -b.q - b.r

    return (abs(ax - bx) + abs(ay - by) + abs(az - bz)) // 2


def k_ring(cell_id: int, k: int) -> tuple[int, ...]:
    """Return all cells within hex distance k of cell_id (including itself)."""
    if k < 0:
        raise ValueError("k must be >= 0")

    idx = decode(cell_id)
    cells = []
    for dq in range(-k, k + 1):
        r_lo = max(-k, -dq - k)
        r_hi = min(k, -dq + k)
        for dr in range(r_lo, r_hi + 1):
            cells.append(encode(idx.q + dq, idx.r + dr, idx.side_length, idx.orientation))
    return tuple(cells)
