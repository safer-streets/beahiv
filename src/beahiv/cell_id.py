"""Cell identifier encoding: a 52-bit layout, so every id is exact as a float64.

Bit layout (MSB to LSB)::

    [63..53]  reserved     11 bits  (always zero, permanently -- see FLOAT_EXACT_BITS)
    [52]      reserved      1 bit   (zero for now; kept for future use, likely a Morton flag)
    [51]      measure       1 bit   (see Measure)
    [50]      orientation   1 bit   (see Orientation)
    [49..38]  size         12 bits  (3-bit binary exponent, 9-bit mantissa)
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
free -- a decoder that ignored unknown bits could never start honouring one. Its likely use is a
Morton flag: Morton and plain ids are still indistinguishable, so decoding one with the wrong
function yields wrong q/r rather than an error, and setting bit 52 on Morton ids would let `decode`
tell them apart.

The size is a whole number of metres stored as a small binary float, the way IEEE 754 does it:
exponent 0 holds the mantissa as-is (0..511 metres, the "subnormal" range), and exponent e >= 1 holds
`(512 + mantissa) << (e - 1)`, a 10-bit significand with an implicit leading bit. So every whole
metre from 1 to 1023 is representable, then sizes with 10 significant bits up to 1023 << 6 = 65,472m
-- 1000m, 1500m, 2000m, 5000m and 10km all are, 1025m and 25km (2**3 * 3125) are not. Storing the
metres directly needed 17 bits for less range.

Binary rather than decimal because the factor-of-two hierarchy (`get_parent(s)`/`get_child(ren)`)
then always has a grid to go to: doubling a size only increments the exponent, and halving an even
one only decrements it (or halves the mantissa), so 2x and 0.5x are representable whenever they are
in range. A decimal three-significant-digit field, tried first, broke that for 1,200 sizes (501 ->
1002). The format also gives every size exactly one encoding and every field value a size, so
there are no non-canonical patterns to reject -- only a size below SIZE_MIN.

The measure bit says what the size measures: the length of a side (0) or the distance between two
parallel sides (1) -- `Measure.EDGE_LENGTH` or `Measure.EDGE_TO_EDGE`. Callers always name it, with the
required `measure=` argument that accompanies every `size=` (see `resolve_size`).

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
from .measure import Measure
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

SIZE_MANTISSA_BITS = 9
SIZE_EXPONENT_BITS = SIDE_LENGTH_BITS - SIZE_MANTISSA_BITS
SIZE_MANTISSA_MASK = (1 << SIZE_MANTISSA_BITS) - 1
SIZE_EXPONENT_MAX = (1 << SIZE_EXPONENT_BITS) - 1
# the implicit leading bit of a normal (exponent >= 1) size
SIZE_LEADING_BIT = 1 << SIZE_MANTISSA_BITS

# geometric side length, metres -- the smallest cell whose q/r still cover the whole box
SIDE_LENGTH_MIN = 3
SIDE_LENGTH_MAX = (SIZE_LEADING_BIT | SIZE_MANTISSA_MASK) << (SIZE_EXPONENT_MAX - 1)

# the smallest stored size meeting SIDE_LENGTH_MIN, indexed by Measure: edge-to-edge is
# sqrt(3) times the edge length
SIZE_MIN = (SIDE_LENGTH_MIN, math.ceil(SIDE_LENGTH_MIN * SQRT3))

UINT64_MASK = (1 << 64) - 1

# an all-zero size field is a size of 0, which encode never produces -- so an all-zero id is safe to
# use as an "invalid" sentinel.
INVALID_CELL_ID = 0


def pack_size(size: int, measure: Measure) -> int:
    """Return the 12-bit size field for `size` metres, or raise if no field can hold it."""
    if not (SIZE_MIN[measure] <= size <= SIDE_LENGTH_MAX):
        raise ValueError(
            f"size must be in [{SIZE_MIN[measure]}, {SIDE_LENGTH_MAX}] for {measure.name.lower()}, got {size}"
        )
    if size < SIZE_LEADING_BIT:
        return size
    shift = size.bit_length() - (SIZE_MANTISSA_BITS + 1)
    if size & ((1 << shift) - 1):
        below = size >> shift << shift
        raise ValueError(
            f"size {size} is not representable: above {2 * SIZE_LEADING_BIT - 1}m a size has 10 significant "
            f"bits -- the nearest are {below} and {below + (1 << shift)}"
        )
    return ((shift + 1) << SIZE_MANTISSA_BITS) | (size >> shift) - SIZE_LEADING_BIT


def unpack_size(field: int) -> int:
    """Return the size a 12-bit size field holds. Every field is some size; 0 is the only one
    `pack_size` never produces, and the caller still has to check SIZE_MIN."""
    exponent = field >> SIZE_MANTISSA_BITS
    mantissa = field & SIZE_MANTISSA_MASK
    if exponent == 0:
        return mantissa
    return (SIZE_LEADING_BIT | mantissa) << (exponent - 1)


@dataclass(frozen=True, slots=True)
class CellIndex:
    """A decoded cell id. `size` is the stored whole-metre size, measured as `measure` says.

    `side_length` is the *geometric* side length that size implies -- the stored size itself for an
    EDGE_LENGTH cell, and `size / sqrt(3)` for an EDGE_TO_EDGE one. It is what every geometric
    formula wants; encoding and hierarchy arithmetic want `size`.
    """

    q: int
    r: int
    size: int
    measure: Measure
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


def resolve_size(size: SupportsIndex, measure: Measure) -> tuple[int, Measure]:
    """Check the public `size` / `measure` arguments and return them as (int, Measure).

    Every public function that takes a cell size takes this pair, and funnels it through here.
    `measure` must be a `Measure` member: Measure and Orientation are both 0/1 IntEnums, so
    converting a plain int (or an Orientation passed by mistake) would pick a lattice silently.
    """
    if not isinstance(measure, Measure):
        raise TypeError(f"measure must be a Measure (Measure.EDGE_LENGTH or Measure.EDGE_TO_EDGE), got {measure!r}")
    return index(size), measure


def encode(
    q: SupportsIndex,
    r: SupportsIndex,
    *,
    size: SupportsIndex,
    measure: Measure,
    orientation: Orientation = Orientation.FLAT,
) -> int:
    """Encode an axial cell at a given size/orientation into a cell id.

    `size` is in whole metres, from SIZE_MIN to 65,472 (above 1023, with 10 significant bits -- see
    the module docstring), and `measure` says what it measures: `Measure.EDGE_LENGTH` (the length of an
    edge) or `Measure.EDGE_TO_EDGE` (the distance between two parallel edges). Both are required.
    Accepts numpy integers as well as `int` -- see the module docstring.
    """
    size, measure = resolve_size(size, measure)
    return encode_size(q, r, size, measure, orientation)


def encode_size(
    q: SupportsIndex,
    r: SupportsIndex,
    size: SupportsIndex,
    measure: Measure,
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
    two fields are not self-validating: the size field holds sizes below SIZE_MIN (0 among them), and
    the reserved field is only meaningful while it stays zero.

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
            "an id from before beahiv 0.1.0, or one that has been through a lossy conversion?"
        )

    size = unpack_size((cell_id >> SIDE_LENGTH_SHIFT) & SIDE_LENGTH_MASK)
    measure = Measure((cell_id >> MEASURE_SHIFT) & MEASURE_MASK)
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
    measure = Measure((cell_id >> MEASURE_SHIFT) & MEASURE_MASK)
    q_enc = (cell_id >> Q_SHIFT) & Q_MASK
    r_enc = (cell_id >> R_SHIFT) & R_MASK

    return CellIndex(
        q=q_enc - Q_OFFSET,
        r=r_enc - R_OFFSET,
        size=size,
        measure=measure,
        orientation=orientation,
    )
