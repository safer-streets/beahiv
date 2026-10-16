"""Cell identifier encoding: a 52-bit layout, so every id is exact as a float64.

Bit layout (MSB to LSB)::

    [63..52]  reserved     12 bits  (always zero)
    [51]      measure       1 bit   (see SizeMeasure)
    [50]      orientation   1 bit   (see Orientation)
    [49..38]  size         12 bits  (2-bit decimal exponent, 10-bit mantissa)
    [37..19]  q_enc        19 bits  (q + Q_OFFSET)
    [18..0]   r_enc        19 bits  (r + R_OFFSET)

Every id is below 2**52, and so below 2**53, the largest range in which float64 holds every
integer exactly. That is the point of the layout, not a side effect of it. Ids get pushed through
floats all the time: pandas turns an int64 column into float64 the moment it gains a NaN,
JSON treats every number as a double, and a merge or reindex can make the round
trip without anyone noticing. Under the previous 64-bit layout an id came back from that rounded,
yet still a well-formed id, so it decoded to a valid cell up to hundreds of kilometres away with
nothing to say so. Below 2**53 the round trip is exact, so there is nothing to detect.

Of the reserved bits, only bit 52 may ever be given a meaning: an id that sets any bit above it is
no longer exact as a float, which would bring the problem straight back (see FLOAT_EXACT_BITS).
Until then `encode` never sets it and `decode` refuses an id that has it set, which is what keeps it
free -- a decoder that ignored unknown bits could never start honouring one. Morton and plain ids
are still indistinguishable, so decoding one with the wrong function yields wrong q/r rather than
an error; the spare bit is the obvious candidate if that ever needs fixing.

The size is a whole number of metres with at most three significant digits: `mantissa * 10 **
exponent`, mantissa 1..999 and exponent 0..3. That spans 3m to 999km in 12 bits -- 25m, 1.5km, 14.5km
and 500km are all representable, 1234m is not -- where storing the metres directly needed 17 bits
for less range. Each size has exactly one encoding, the one with the smallest exponent, so a
mantissa below 100 with a non-zero exponent is rejected; otherwise 500m could be stored as either
500e0 or 50e1, giving one cell two ids.

The measure bit says what the size measures: the length of a side (0) or the distance between two
parallel sides (1). Callers choose with the mutually exclusive `side_length` / `side_to_side`
arguments (see `resolve_size`).

q/r get 19 bits each, offset to fit the National Grid box (0-700km east, 0-1300km north) at the
smallest cell allowed: a geometric side length of SIDE_LENGTH_MIN metres. That
minimum is what the 52-bit budget costs -- the old layout reached 1m. A smaller cell isn't simply
unreachable at the edges of GB; it is rejected outright by `encode`, rather than encoding near the
origin and raising further out.

Cell ids and coordinates are taken as `SupportsIndex` rather than `int`, and
coerced with `operator.index` on the way in. Ids reach the scalar API as numpy
integers routinely -- an element of a `decode_batch` result, a pandas column, a
DuckDB/Parquet BIGINT -- and numpy's fixed-width arithmetic is wrong for this
bit layout: a `q_enc - Q_OFFSET` that should go negative wraps around under
`np.uint64`. Coercing at these entry points is what keeps every value downstream
a plain Python int, so `neighbours`, `hierarchy` and `polyfill` need no numpy
awareness of their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from operator import index
from typing import SupportsIndex

from .coords import SQRT3, side_length_of
from .measure import SizeMeasure
from .orientation import Orientation

R_BITS = 19
Q_BITS = 19
SIDE_LENGTH_BITS = 12
ORIENTATION_BITS = 1
MEASURE_BITS = 1
RESERVED_BITS = 12

assert R_BITS + Q_BITS + SIDE_LENGTH_BITS + ORIENTATION_BITS + MEASURE_BITS + RESERVED_BITS == 64

R_SHIFT = 0
Q_SHIFT = R_BITS
SIDE_LENGTH_SHIFT = R_BITS + Q_BITS
ORIENTATION_SHIFT = SIDE_LENGTH_SHIFT + SIDE_LENGTH_BITS
MEASURE_SHIFT = ORIENTATION_SHIFT + ORIENTATION_BITS
RESERVED_SHIFT = MEASURE_SHIFT + MEASURE_BITS

# float64 has a 53-bit significand, so it holds every integer below 2**53 exactly. The reserved
# field's lowest bit is the only one that may ever be spent; this keeps ids exact even then.
FLOAT_EXACT_BITS = 53
assert RESERVED_SHIFT + 1 <= FLOAT_EXACT_BITS

R_MASK = (1 << R_BITS) - 1
Q_MASK = (1 << Q_BITS) - 1
SIDE_LENGTH_MASK = (1 << SIDE_LENGTH_BITS) - 1
ORIENTATION_MASK = (1 << ORIENTATION_BITS) - 1
MEASURE_MASK = (1 << MEASURE_BITS) - 1
RESERVED_MASK = (1 << RESERVED_BITS) - 1

# Not centred on zero: GB lies north-east of the grid origin, so a symmetric range would waste most
# of q/r's bits on the sea. These put the National Grid box, at a 3m side, near the middle of both
# fields under both orientations (test_cell_id checks its corners encode). The slack either side is
# large -- over 300km beyond every edge of the box even at 3m -- so nothing near GB is out of range.
Q_OFFSET = 260_000
R_OFFSET = 160_000

SIZE_MANTISSA_BITS = 10
SIZE_MANTISSA_MASK = (1 << SIZE_MANTISSA_BITS) - 1
SIZE_MANTISSA_MAX = 999
SIZE_EXPONENT_MAX = 3
assert SIZE_MANTISSA_MAX <= SIZE_MANTISSA_MASK
assert SIZE_EXPONENT_MAX < 1 << (SIDE_LENGTH_BITS - SIZE_MANTISSA_BITS)

# geometric side length, metres -- the smallest cell whose q/r still cover the whole box
SIDE_LENGTH_MIN = 3
SIDE_LENGTH_MAX = SIZE_MANTISSA_MAX * 10**SIZE_EXPONENT_MAX

# the smallest stored size meeting SIDE_LENGTH_MIN, indexed by SizeMeasure: side-to-side is
# sqrt(3) times the side length
SIZE_MIN = (SIDE_LENGTH_MIN, math.ceil(SIDE_LENGTH_MIN * SQRT3))

UINT64_MASK = (1 << 64) - 1

# an all-zero size field has mantissa 0, which encode never produces -- so an all-zero id is safe
# to use as an "invalid" sentinel.
INVALID_CELL_ID = 0


def pack_size(size: int, measure: SizeMeasure) -> int:
    """Return the 12-bit size field for `size` metres, or raise if no field can hold it."""
    if not (SIZE_MIN[measure] <= size <= SIDE_LENGTH_MAX):
        raise ValueError(
            f"size must be in [{SIZE_MIN[measure]}, {SIDE_LENGTH_MAX}] for {measure.name.lower()}, got {size}"
        )
    exponent = 0
    mantissa = size
    while mantissa > SIZE_MANTISSA_MAX and mantissa % 10 == 0:
        mantissa //= 10
        exponent += 1
    if mantissa > SIZE_MANTISSA_MAX:
        raise ValueError(
            f"size {size} is not representable: sizes have at most 3 significant digits (e.g. 1230, not 1234)"
        )
    return (exponent << SIZE_MANTISSA_BITS) | mantissa


def unpack_size(field: int) -> int:
    """Return the size a 12-bit size field holds, or 0 if `pack_size` could not have produced it."""
    exponent = field >> SIZE_MANTISSA_BITS
    mantissa = field & SIZE_MANTISSA_MASK
    if not (1 <= mantissa <= SIZE_MANTISSA_MAX) or (exponent and mantissa < 100):
        return 0
    return mantissa * 10**exponent


@dataclass(frozen=True, slots=True)
class CellIndex:
    """A decoded cell id. `size` is the stored whole-metre size, measured as `measure` says.

    `side_length` is the *geometric* side length that size implies -- the stored size itself for a
    SIDE_LENGTH cell, and `size / sqrt(3)` for a SIDE_TO_SIDE one. It is what every geometric
    formula wants; encoding and hierarchy arithmetic want `size`.
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
    between two parallel edges), in whole metres with at most 3 significant digits. Accepts numpy
    integers as well as `int` -- see the module docstring.
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
    size_field = pack_size(index(size), measure)

    q_enc = q + Q_OFFSET
    r_enc = r + R_OFFSET
    if not (0 <= q_enc <= Q_MASK):
        raise ValueError(f"q is out of representable range: {q}")
    if not (0 <= r_enc <= R_MASK):
        raise ValueError(f"r is out of representable range: {r}")

    return (
        (int(measure) << MEASURE_SHIFT)
        | (int(orientation) << ORIENTATION_SHIFT)
        | (size_field << SIDE_LENGTH_SHIFT)
        | (q_enc << Q_SHIFT)
        | r_enc
    )


def validate_cell_id(cell_id: SupportsIndex) -> tuple[int, int]:
    """Reject any id `encode` could not have produced, and return it with its size.

    The id comes back because coercing it (see the module docstring) happens here, so the caller
    would otherwise redo it before reading q/r out of the same bits.

    q/r fill their fields exactly, so every bit pattern is a valid coordinate and there is nothing
    to check there, and both values of the orientation and measure bits are meaningful. The other
    two fields are not self-validating: the size field has non-canonical and out-of-range patterns
    (see `unpack_size` and SIZE_MIN), and the reserved field is only meaningful while it stays zero.

    Shared by `decode` and `morton.decode_morton`, which differ only in how they read q/r.
    `batch.decode_batch` re-implements this vectorised rather than calling it per row.
    """
    cell_id = index(cell_id)
    if cell_id < 0:
        raise ValueError(f"cell_id must not be negative, got {cell_id}")
    if cell_id == INVALID_CELL_ID:
        raise ValueError("cell_id is INVALID_CELL_ID, the missing-input sentinel -- filter these out before decoding")
    if cell_id >> RESERVED_SHIFT:
        raise ValueError(
            f"cell_id has reserved bits set, so is not a valid cell id: {cell_id:#x} -- "
            "an id from before beahiv 1.0, or one that has been through a lossy conversion?"
        )

    size = unpack_size((cell_id >> SIDE_LENGTH_SHIFT) & SIDE_LENGTH_MASK)
    measure = SizeMeasure((cell_id >> MEASURE_SHIFT) & MEASURE_MASK)
    if size < SIZE_MIN[measure]:
        raise ValueError(f"cell_id has an invalid size field: {cell_id:#x}")
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
