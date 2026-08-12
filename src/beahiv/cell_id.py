"""64-bit cell identifier encoding.

Bit layout (MSB to LSB)::

    [63..61]  reserved      3 bits  (always zero)
    [60]      orientation   1 bit   (see Orientation)
    [59..43]  side_length  17 bits  (unsigned metres, 1..SIDE_LENGTH_MAX)
    [42..22]  q_enc        21 bits  (q + Q_OFFSET)
    [21..0]   r_enc        22 bits  (r + R_OFFSET)

Side length is stored directly (in whole metres) rather than as an index
into a resolution table, so any grid spacing that fits the bit budget is
representable without a lookup table. It is capped at 100km: a hexagon that
size has an area larger than Wales, so nothing coarser is a meaningful unit
inside GB, and the cap is what frees the top three bits.

Those three are left reserved at the most significant end rather than spent,
which puts every cell id below 2**61 -- comfortably inside a *signed* 64-bit
integer, so consumers can store ids in a plain int64/BIGINT column instead of
needing an unsigned type or a hex string. They are held for future use, such
as flagging optional Morton encoding: a Morton id and a plain id are currently
indistinguishable, so decoding one with the wrong function silently yields
wrong q/r rather than an error. `encode` never sets them and `decode` refuses
an id that has them set, which is what keeps them free to be given a meaning
later -- a decoder that ignored unknown bits could never start honouring one.

q/r get 21/22 bits (not stored per-orientation) which comfortably covers all
of Great Britain down to ~1m side length -- far finer than the 25m floor
anticipated by the spec.

Cell ids and coordinates are taken as `SupportsIndex` rather than `int`, and
coerced with `operator.index` on the way in. Ids reach the scalar API as numpy
integers routinely -- an element of a `decode_batch` result, a pandas column, a
DuckDB/Parquet BIGINT -- and numpy's fixed-width arithmetic is wrong for this
bit layout in both directions: `& UINT64_MASK` raises OverflowError against an
`np.int64` operand because the mask doesn't fit one, and a `q_enc - Q_OFFSET`
that should go negative wraps around instead under `np.uint64`. Coercing at
these entry points is what keeps every value downstream a plain Python int, so
`neighbours`, `hierarchy` and `polyfill` need no numpy awareness of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import SupportsIndex

from .orientation import Orientation

R_BITS = 22
Q_BITS = 21
SIDE_LENGTH_BITS = 17
ORIENTATION_BITS = 1
RESERVED_BITS = 3

assert R_BITS + Q_BITS + SIDE_LENGTH_BITS + ORIENTATION_BITS + RESERVED_BITS == 64

R_SHIFT = 0
Q_SHIFT = R_BITS
SIDE_LENGTH_SHIFT = R_BITS + Q_BITS
ORIENTATION_SHIFT = R_BITS + Q_BITS + SIDE_LENGTH_BITS
RESERVED_SHIFT = ORIENTATION_SHIFT + ORIENTATION_BITS

R_MASK = (1 << R_BITS) - 1
Q_MASK = (1 << Q_BITS) - 1
SIDE_LENGTH_MASK = (1 << SIDE_LENGTH_BITS) - 1
ORIENTATION_MASK = (1 << ORIENTATION_BITS) - 1
RESERVED_MASK = (1 << RESERVED_BITS) - 1

R_OFFSET = 1 << (R_BITS - 1)
Q_OFFSET = 1 << (Q_BITS - 1)

SIDE_LENGTH_MAX = 100_000

# unlike q/r, side_length doesn't use its whole field -- the cap is the limit,
# not the mask, and encode() must check against the cap or a larger value would
# silently overflow into the orientation bit
assert SIDE_LENGTH_MAX <= SIDE_LENGTH_MASK

UINT64_MASK = (1 << 64) - 1

# side_length must be >= 1 (see encode()), so an all-zero id can never be
# produced by a valid encode call -- safe to use as an "invalid" sentinel.
INVALID_CELL_ID = 0


@dataclass(frozen=True, slots=True)
class CellIndex:
    q: int
    r: int
    side_length: int
    orientation: Orientation

    def dq(self, i: int) -> CellIndex:
        return CellIndex(self.q + i, self.r, self.side_length, self.orientation)

    def dr(self, i: int) -> CellIndex:
        return CellIndex(self.q, self.r + i, self.side_length, self.orientation)

    def encode(self) -> int:
        return encode(self.q, self.r, self.side_length, self.orientation)


def encode(
    q: SupportsIndex,
    r: SupportsIndex,
    side_length: SupportsIndex,
    orientation: Orientation = Orientation.FLAT,
) -> int:
    """Encode an axial cell at a given side length/orientation into a uint64 id.

    Accepts numpy integers as well as `int` -- see the module docstring.
    """
    q = index(q)
    r = index(r)
    side_length = index(side_length)

    if not (1 <= side_length <= SIDE_LENGTH_MAX):
        raise ValueError(f"side_length must be in [1, {SIDE_LENGTH_MAX}], got {side_length}")

    q_enc = q + Q_OFFSET
    r_enc = r + R_OFFSET
    if not (0 <= q_enc <= Q_MASK):
        raise ValueError(f"q is out of representable range: {q}")
    if not (0 <= r_enc <= R_MASK):
        raise ValueError(f"r is out of representable range: {r}")

    cell_id = (int(orientation) << ORIENTATION_SHIFT) | (side_length << SIDE_LENGTH_SHIFT) | (q_enc << Q_SHIFT) | r_enc
    return cell_id & UINT64_MASK


def validate_cell_id(cell_id: SupportsIndex) -> tuple[int, int]:
    """Reject any id `encode` could not have produced, and return it with its side_length.

    The id comes back because coercing it (see the module docstring) happens here, so the caller
    would otherwise redo it before reading q/r out of the same bits.

    q/r fill their fields exactly, so every bit pattern is a valid coordinate and there is nothing
    to check there. The other two fields are not self-validating: side_length's limit is a range
    check rather than its bit width, and the reserved field is only meaningful while it stays zero
    (see the module docstring -- a future flag there is what would finally make a Morton id
    distinguishable from a plain one, which requires decode to be looking at it).

    Shared by `decode` and `morton.decode_morton`, which differ only in how they read q/r.
    `batch.decode_batch` re-implements this vectorised rather than calling it per row.
    """
    cell_id = index(cell_id)
    if not (0 <= cell_id <= UINT64_MASK):
        raise ValueError("cell_id must fit within uint64")
    if cell_id == INVALID_CELL_ID:
        raise ValueError("cell_id is INVALID_CELL_ID, the missing-input sentinel -- filter these out before decoding")
    if (cell_id >> RESERVED_SHIFT) & RESERVED_MASK:
        raise ValueError(f"cell_id has reserved bits set, so is not a valid cell id: {cell_id:#018x}")

    side_length = (cell_id >> SIDE_LENGTH_SHIFT) & SIDE_LENGTH_MASK
    if not (1 <= side_length <= SIDE_LENGTH_MAX):
        raise ValueError(f"cell_id has side_length {side_length}, outside the encodable [1, {SIDE_LENGTH_MAX}]")
    return cell_id, side_length


def decode(cell_id: SupportsIndex) -> CellIndex:
    """Recover (q, r, side_length, orientation) from a uint64 cell id.

    Accepts numpy integers as well as `int` -- see the module docstring.

    Raises ValueError for any id `encode` could not have produced -- see `validate_cell_id`.
    """
    cell_id, side_length = validate_cell_id(cell_id)

    orientation = Orientation((cell_id >> ORIENTATION_SHIFT) & ORIENTATION_MASK)
    q_enc = (cell_id >> Q_SHIFT) & Q_MASK
    r_enc = (cell_id >> R_SHIFT) & R_MASK

    return CellIndex(
        q=q_enc - Q_OFFSET,
        r=r_enc - R_OFFSET,
        side_length=side_length,
        orientation=orientation,
    )
