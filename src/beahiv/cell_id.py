"""64-bit cell identifier encoding.

Bit layout (MSB to LSB)::

    [63..62]  reserved      2 bits  (always zero)
    [61]      measure       1 bit   (see SizeMeasure)
    [60]      orientation   1 bit   (see Orientation)
    [59..43]  size         17 bits  (unsigned metres, 1..SIDE_LENGTH_MAX)
    [42..22]  q_enc        21 bits  (q + Q_OFFSET)
    [21..0]   r_enc        22 bits  (r + R_OFFSET)

Size is stored directly (in whole metres) rather than as an index into a
resolution table, so any grid spacing that fits the bit budget is
representable without a lookup table. It is capped at 100km: a hexagon that
size has an area larger than Wales, so nothing coarser is a meaningful unit
inside GB, and the cap is what frees the top three bits.

The first of those three is the measure bit, saying what the size field
measures: the length of a side (0 -- the layout before the bit existed, so
every older id still decodes unchanged) or the distance between two parallel
sides (1). Callers choose with the mutually exclusive `side_length` /
`side_to_side` arguments (see `resolve_size`), and the cap applies to the
stored size whichever it is.

The other two are left reserved at the most significant end rather than spent,
which puts every cell id below 2**62 -- comfortably inside a *signed* 64-bit
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

from .coords import side_length_of
from .measure import SizeMeasure
from .orientation import Orientation

R_BITS = 22
Q_BITS = 21
SIDE_LENGTH_BITS = 17
ORIENTATION_BITS = 1
MEASURE_BITS = 1
RESERVED_BITS = 2

assert R_BITS + Q_BITS + SIDE_LENGTH_BITS + ORIENTATION_BITS + MEASURE_BITS + RESERVED_BITS == 64

R_SHIFT = 0
Q_SHIFT = R_BITS
SIDE_LENGTH_SHIFT = R_BITS + Q_BITS
ORIENTATION_SHIFT = R_BITS + Q_BITS + SIDE_LENGTH_BITS
MEASURE_SHIFT = ORIENTATION_SHIFT + ORIENTATION_BITS
RESERVED_SHIFT = MEASURE_SHIFT + MEASURE_BITS

R_MASK = (1 << R_BITS) - 1
Q_MASK = (1 << Q_BITS) - 1
SIDE_LENGTH_MASK = (1 << SIDE_LENGTH_BITS) - 1
ORIENTATION_MASK = (1 << ORIENTATION_BITS) - 1
MEASURE_MASK = (1 << MEASURE_BITS) - 1
RESERVED_MASK = (1 << RESERVED_BITS) - 1

R_OFFSET = 1 << (R_BITS - 1)
Q_OFFSET = 1 << (Q_BITS - 1)

SIDE_LENGTH_MAX = 100_000

# unlike q/r, the size doesn't use its whole field -- the cap is the limit,
# not the mask, and encode() must check against the cap or a larger value would
# silently overflow into the orientation bit. (The SIDE_LENGTH_* names predate
# the measure bit; the field holds the size under either measure.)
assert SIDE_LENGTH_MAX <= SIDE_LENGTH_MASK

UINT64_MASK = (1 << 64) - 1

# size must be >= 1 (see encode()), so an all-zero id can never be
# produced by a valid encode call -- safe to use as an "invalid" sentinel.
INVALID_CELL_ID = 0


@dataclass(frozen=True, slots=True)
class CellIndex:
    """A decoded cell id. `size` is the stored whole-metre size, measured as `measure` says.

    `side_length` is the *geometric* side length that size implies -- the stored size itself for a
    SIDE_LENGTH cell (so ids from before the measure bit read exactly as they always did), and
    `size / sqrt(3)` for a SIDE_TO_SIDE one. It is what every geometric formula wants; encoding
    and hierarchy arithmetic want `size`.
    """

    q: int
    r: int
    size: int
    measure: SizeMeasure
    orientation: Orientation

    @property
    def side_length(self) -> float:
        return side_length_of(self.size, self.measure)

    def dq(self, i: int) -> CellIndex:
        return CellIndex(self.q + i, self.r, self.size, self.measure, self.orientation)

    def dr(self, i: int) -> CellIndex:
        return CellIndex(self.q, self.r + i, self.size, self.measure, self.orientation)

    def encode(self) -> int:
        return encode_size(self.q, self.r, self.size, self.measure, self.orientation)


def resolve_size(side_length: SupportsIndex | None, side_to_side: SupportsIndex | None) -> tuple[int, SizeMeasure]:
    """Turn the mutually exclusive `side_length` / `side_to_side` arguments into (size, measure).

    Every public function that takes a cell size takes it this way, and funnels it through here.
    """
    # (None, None) must come first: it is what lets the type checker narrow `size` below to non-None
    match side_length, side_to_side:
        case None, None:
            raise TypeError("a cell size is required: pass side_length or side_to_side")
        case size, None:
            return index(size), SizeMeasure.SIDE_LENGTH
        case None, size:
            return index(size), SizeMeasure.SIDE_TO_SIDE
        case _:
            raise TypeError("pass side_length or side_to_side, not both")


def encode(
    q: SupportsIndex,
    r: SupportsIndex,
    *,
    side_length: SupportsIndex | None = None,
    side_to_side: SupportsIndex | None = None,
    orientation: Orientation = Orientation.FLAT,
) -> int:
    """Encode an axial cell at a given size/orientation into a cell id.

    Give exactly one of `side_length` (the length of an edge) or `side_to_side` (the distance
    between two parallel edges), in whole metres. Accepts numpy integers as well as `int` -- see
    the module docstring.
    """
    size, measure = resolve_size(side_length, side_to_side)
    return encode_size(q, r, size, measure, orientation)


def encode_size(
    q: SupportsIndex,
    r: SupportsIndex,
    size: SupportsIndex,
    measure: SizeMeasure,
    orientation: Orientation,
) -> int:
    """`encode`, with the size already resolved -- the form internal callers holding a
    `CellIndex` use, so they carry its measure through rather than re-deciding it."""
    q = index(q)
    r = index(r)
    size = index(size)

    if not (1 <= size <= SIDE_LENGTH_MAX):
        raise ValueError(f"size must be in [1, {SIDE_LENGTH_MAX}], got {size}")

    q_enc = q + Q_OFFSET
    r_enc = r + R_OFFSET
    if not (0 <= q_enc <= Q_MASK):
        raise ValueError(f"q is out of representable range: {q}")
    if not (0 <= r_enc <= R_MASK):
        raise ValueError(f"r is out of representable range: {r}")

    cell_id = (
        (int(measure) << MEASURE_SHIFT)
        | (int(orientation) << ORIENTATION_SHIFT)
        | (size << SIDE_LENGTH_SHIFT)
        | (q_enc << Q_SHIFT)
        | r_enc
    )
    return cell_id & UINT64_MASK


def validate_cell_id(cell_id: SupportsIndex) -> tuple[int, int]:
    """Reject any id `encode` could not have produced, and return it with its size.

    The id comes back because coercing it (see the module docstring) happens here, so the caller
    would otherwise redo it before reading q/r out of the same bits.

    q/r fill their fields exactly, so every bit pattern is a valid coordinate and there is nothing
    to check there, and both values of the orientation and measure bits are meaningful. The other
    two fields are not self-validating: size's limit is a range check rather than its bit width,
    and the reserved field is only meaningful while it stays zero (see the module docstring -- a
    future flag there is what would finally make a Morton id distinguishable from a plain one,
    which requires decode to be looking at it).

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

    size = (cell_id >> SIDE_LENGTH_SHIFT) & SIDE_LENGTH_MASK
    if not (1 <= size <= SIDE_LENGTH_MAX):
        raise ValueError(f"cell_id has size {size}, outside the encodable [1, {SIDE_LENGTH_MAX}]")
    return cell_id, size


def decode(cell_id: SupportsIndex) -> CellIndex:
    """Recover (q, r, size, measure, orientation) from a cell id.

    Accepts numpy integers as well as `int` -- see the module docstring.

    Raises ValueError for any id `encode` could not have produced -- see `validate_cell_id`.
    """
    cell_id, size = validate_cell_id(cell_id)

    orientation = Orientation((cell_id >> ORIENTATION_SHIFT) & ORIENTATION_MASK)
    measure = SizeMeasure((cell_id >> MEASURE_SHIFT) & MEASURE_MASK)
    q_enc = (cell_id >> Q_SHIFT) & Q_MASK
    r_enc = (cell_id >> R_SHIFT) & R_MASK

    return CellIndex(
        q=q_enc - Q_OFFSET,
        r=r_enc - R_OFFSET,
        size=size,
        measure=measure,
        orientation=orientation,
    )
