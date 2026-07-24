"""Optional Morton (Z-order) cell id variant.

Same bit budget and same (orientation, side_length) prefix as the plain
id in cell_id.py, but q/r occupy an interleaved 43-bit region instead of
being simply concatenated. This groups spatially-close cells into nearby
integer ranges, which helps DuckDB clustering, Parquet sort-order pruning,
and range scans. Fully reversible.
"""

from .cell_id import (
    ORIENTATION_MASK,
    ORIENTATION_SHIFT,
    Q_BITS,
    Q_MASK,
    Q_OFFSET,
    R_BITS,
    R_MASK,
    R_OFFSET,
    SIDE_LENGTH_MASK,
    SIDE_LENGTH_SHIFT,
    UINT64_MASK,
    CellIndex,
)
from .orientation import Orientation

_MORTON_BITS = Q_BITS + R_BITS
_MORTON_MASK = (1 << _MORTON_BITS) - 1


def _interleave(a: int, a_bits: int, b: int, b_bits: int) -> int:
    """Interleave bits of a and b, b in the low bit of each pair."""
    result = 0
    for i in range(max(a_bits, b_bits)):
        if i < b_bits:
            result |= ((b >> i) & 1) << (2 * i)
        if i < a_bits:
            result |= ((a >> i) & 1) << (2 * i + 1)
    return result


def _deinterleave(value: int, a_bits: int, b_bits: int) -> tuple[int, int]:
    a = 0
    b = 0
    for i in range(max(a_bits, b_bits)):
        if i < b_bits:
            b |= ((value >> (2 * i)) & 1) << i
        if i < a_bits:
            a |= ((value >> (2 * i + 1)) & 1) << i
    return a, b


def encode_morton(
    q: int,
    r: int,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> int:
    if not (1 <= side_length <= SIDE_LENGTH_MASK):
        raise ValueError(f"side_length must be in [1, {SIDE_LENGTH_MASK}], got {side_length}")

    q_enc = q + Q_OFFSET
    r_enc = r + R_OFFSET
    if not (0 <= q_enc <= Q_MASK):
        raise ValueError(f"q is out of representable range: {q}")
    if not (0 <= r_enc <= R_MASK):
        raise ValueError(f"r is out of representable range: {r}")

    morton = _interleave(q_enc, Q_BITS, r_enc, R_BITS)

    cell_id = (int(orientation) << ORIENTATION_SHIFT) | (side_length << SIDE_LENGTH_SHIFT) | morton
    return cell_id & UINT64_MASK


def decode_morton(cell_id: int) -> CellIndex:
    if not (0 <= cell_id <= UINT64_MASK):
        raise ValueError("cell_id must fit within uint64")

    orientation = Orientation((cell_id >> ORIENTATION_SHIFT) & ORIENTATION_MASK)
    side_length = (cell_id >> SIDE_LENGTH_SHIFT) & SIDE_LENGTH_MASK
    morton = cell_id & _MORTON_MASK

    q_enc, r_enc = _deinterleave(morton, Q_BITS, R_BITS)

    return CellIndex(
        q=q_enc - Q_OFFSET,
        r=r_enc - R_OFFSET,
        side_length=side_length,
        orientation=orientation,
    )
