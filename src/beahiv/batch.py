"""Vectorised numpy variants for encoding/decoding many points at once.

Single-point lookups (geo.py, cell_id.py) already meet the < 1us target
without numpy. This module exists for the bulk case explicitly called
out in the spec -- encoding a whole crime dataset at once -- where a
Python-level loop dominates and numpy vectorisation matters.
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

from .cell_id import (
    INVALID_CELL_ID,
    ORIENTATION_MASK,
    ORIENTATION_SHIFT,
    Q_MASK,
    Q_OFFSET,
    Q_SHIFT,
    R_MASK,
    R_OFFSET,
    RESERVED_MASK,
    RESERVED_SHIFT,
    SIDE_LENGTH_MASK,
    SIDE_LENGTH_MAX,
    SIDE_LENGTH_SHIFT,
)
from .coords import SQRT3
from .orientation import Orientation

_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# Mirrors geo._check_in_area_of_use / geo._LAT_MIN etc -- see there for why
# points outside this box (EPSG:27700's area of use) can't just be projected
# and encoded: PROJ extrapolates instead of erroring, which can overflow or
# silently wrap the q/r cell encoding depending on side_length.
_LAT_MIN, _LAT_MAX = 49.75, 61.01
_LON_MIN, _LON_MAX = -9.01, 2.01


def _check_in_area_of_use_batch(lons: np.ndarray, lats: np.ndarray, valid: np.ndarray) -> None:
    """Raise if any non-NaN (lon, lat) falls outside EPSG:27700's area of use."""
    in_bounds = (lons >= _LON_MIN) & (lons <= _LON_MAX) & (lats >= _LAT_MIN) & (lats <= _LAT_MAX)
    bad = valid & ~in_bounds
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"(lon={lons[i]}, lat={lats[i]}) at index {i} is outside EPSG:27700's "
            f"area of use (lon in [{_LON_MIN}, {_LON_MAX}], lat in [{_LAT_MIN}, {_LAT_MAX}]) "
            "-- check the arguments aren't swapped"
        )


def axial_to_cartesian_batch(
    q: np.ndarray,
    r: np.ndarray,
    side_length: float,
    orientation: Orientation = Orientation.FLAT,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    s = side_length
    if orientation == Orientation.POINTY:
        x = s * SQRT3 * (q + r / 2.0)
        y = 1.5 * s * r
    else:
        x = 1.5 * s * q
        y = s * SQRT3 * (r + q / 2.0)
    return x, y


def cartesian_to_axial_batch(
    x: ArrayLike,
    y: ArrayLike,
    side_length: float,
    orientation: Orientation = Orientation.FLAT,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = side_length

    if orientation == Orientation.POINTY:
        qf = (SQRT3 / 3.0 * x - y / 3.0) / s
        rf = (2.0 / 3.0 * y) / s
    else:
        qf = (2.0 / 3.0 * x) / s
        rf = (-1.0 / 3.0 * x + SQRT3 / 3.0 * y) / s

    cx, cz = qf, rf
    cy = -cx - cz

    rx = np.floor(cx + 0.5)
    ry = np.floor(cy + 0.5)
    rz = np.floor(cz + 0.5)

    x_diff = np.abs(rx - cx)
    y_diff = np.abs(ry - cy)
    z_diff = np.abs(rz - cz)

    fix_x = (x_diff > y_diff) & (x_diff > z_diff)
    fix_y = (~fix_x) & (y_diff > z_diff)

    rx = np.where(fix_x, -ry - rz, rx)
    ry = np.where(fix_y, -rx - rz, ry)
    rz = np.where(~(fix_x | fix_y), -rx - ry, rz)

    return rx.astype(np.int64), rz.astype(np.int64)


def encode_batch(
    q: np.ndarray,
    r: np.ndarray,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Encode axial coordinates into int64 cell ids -- see `cell_id.encode` for the layout."""
    q = np.asarray(q, dtype=np.int64)
    r = np.asarray(r, dtype=np.int64)

    # side_length doesn't fill its field, so an oversized one would overflow into
    # the orientation bit rather than being masked off -- one scalar check per
    # call, not per row, so the vectorised path pays nothing for it
    if not (1 <= side_length <= SIDE_LENGTH_MAX):
        raise ValueError(f"side_length must be in [1, {SIDE_LENGTH_MAX}], got {side_length}")

    q_enc = q + Q_OFFSET
    r_enc = r + R_OFFSET

    if np.any((q_enc < 0) | (q_enc > Q_MASK)) or np.any((r_enc < 0) | (r_enc > R_MASK)):
        raise ValueError("q or r out of representable range")

    # int64, never uint64: the reserved bits keep every id below 2**61, and a signed dtype is what
    # a BIGINT column holds without conversion. Mixing uint64 with Python ints/int64 also silently
    # promotes to float64 in numpy, which loses the low bits of an id.
    return (
        (np.int64(int(orientation)) << ORIENTATION_SHIFT)
        | (np.int64(side_length) << SIDE_LENGTH_SHIFT)
        | (q_enc << Q_SHIFT)
        | r_enc
    )


def decode_batch(cell_ids: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (q, r, side_length, orientation) arrays for a batch of cell ids.

    Rejects the same ids scalar `decode` does (see `cell_id.validate_cell_id`), vectorised: two
    array comparisons for the whole batch rather than a per-row call. Note that the *_to_cell
    functions emit INVALID_CELL_ID for missing input, so an array built from data with gaps must
    have them filtered out before it can be decoded.
    """
    try:
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
    except OverflowError as exc:
        # A Python int too big for int64 never reaches the bit checks below -- numpy raises during
        # the conversion, with a message about C longs. An id that large has bit 63 set, which is a
        # reserved bit, so the scalar path rejects it too: re-raise as the same ValueError rather
        # than letting the batch path differ in error type (see the rejection-parity rule).
        # An already-uint64 array needs none of this -- it wraps to a negative and the reserved
        # check below catches it.
        raise ValueError(
            "cell id(s) too large to be valid: every id encode produces is below 2**61, "
            "so anything that doesn't fit a signed int64 has reserved bits set"
        ) from exc

    orientation = ((cell_ids >> ORIENTATION_SHIFT) & ORIENTATION_MASK).astype(np.uint8)
    side_length = (cell_ids >> SIDE_LENGTH_SHIFT) & SIDE_LENGTH_MASK

    # a negative id has bit 63 set, which is a reserved bit -- the arithmetic shift keeps
    # those ones, so this rejects it rather than needing a sign check of its own
    reserved = (cell_ids >> RESERVED_SHIFT) & RESERVED_MASK
    if np.any(reserved):
        raise ValueError(f"{int(np.count_nonzero(reserved))} cell id(s) have reserved bits set")
    invalid = (side_length < 1) | (side_length > SIDE_LENGTH_MAX)
    if np.any(invalid):
        n_sentinel = int(np.count_nonzero(cell_ids == INVALID_CELL_ID))
        detail = f" ({n_sentinel} of them INVALID_CELL_ID)" if n_sentinel else ""
        raise ValueError(
            f"{int(np.count_nonzero(invalid))} cell id(s) have a side_length outside "
            f"the encodable [1, {SIDE_LENGTH_MAX}]{detail}"
        )

    q_enc = (cell_ids >> Q_SHIFT) & Q_MASK
    r_enc = cell_ids & R_MASK

    return q_enc - Q_OFFSET, r_enc - R_OFFSET, side_length, orientation


def lonlat_to_cell_batch(
    lons: ArrayLike,
    lats: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Encode each WGS84 (lon, lat) pair; NaN coordinates map to INVALID_CELL_ID.

    Raises if any non-NaN pair is outside EPSG:27700's area of use -- see `geo.lonlat_to_cell`.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    valid = ~(np.isnan(lons) | np.isnan(lats))
    _check_in_area_of_use_batch(lons, lats, valid)

    cell_ids = np.full(lons.shape, INVALID_CELL_ID, dtype=np.int64)
    if np.any(valid):
        x, y = _TO_BNG.transform(lons[valid], lats[valid])
        q, r = cartesian_to_axial_batch(x, y, side_length, orientation)
        cell_ids[valid] = encode_batch(q, r, side_length, orientation)
    return cell_ids


def latlon_to_cell_batch(
    lats: ArrayLike,
    lons: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Deprecated: use `lonlat_to_cell_batch(lons, lats, ...)` -- note the swapped argument order."""
    warnings.warn(
        "latlon_to_cell_batch(lats, lons, ...) is deprecated; use lonlat_to_cell_batch(lons, lats, ...) "
        "-- note the argument order",
        DeprecationWarning,
        stacklevel=2,
    )
    return lonlat_to_cell_batch(lons, lats, side_length, orientation)


def bng_to_cell_batch(
    x: ArrayLike,
    y: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Encode each EPSG:27700 (x, y) pair; NaN coordinates map to INVALID_CELL_ID.

    No area-of-use check: unlike the lat/lon path there is no projection to extrapolate, so a
    coordinate far outside GB is simply a cell far from the origin — and one far enough to exceed
    the q/r bit budget is rejected by ``encode_batch`` rather than wrapping silently.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))

    cell_ids = np.full(x.shape, INVALID_CELL_ID, dtype=np.int64)
    if np.any(valid):
        q, r = cartesian_to_axial_batch(x[valid], y[valid], side_length, orientation)
        cell_ids[valid] = encode_batch(q, r, side_length, orientation)
    return cell_ids


def cell_centre_batch(cell_ids: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return EPSG:27700 (x, y) centres for a batch of cell ids.

    Every cell must share the same side_length and orientation -- a single
    axial_to_cartesian_batch call can't mix them.
    """
    q, r, side_length, orientation = decode_batch(cell_ids)
    orientations = np.unique(orientation)
    if len(orientations) > 1:
        raise ValueError("cell_centre_batch requires a single orientation per call")

    lengths = np.unique(side_length)
    if len(lengths) > 1:
        raise ValueError("cell_centre_batch requires a single side_length per call")

    return axial_to_cartesian_batch(q, r, int(lengths[0]), Orientation(int(orientations[0])))


def cell_to_lonlat_batch(cell_ids: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return WGS84 (lon, lat) centres for a batch of cell ids -- lon first, as `_TO_WGS84` emits."""
    x, y = cell_centre_batch(cell_ids)
    return _TO_WGS84.transform(x, y)


def cell_to_latlon_batch(cell_ids: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated: use `cell_to_lonlat_batch`, which returns (lon, lat) -- the reverse of this."""
    warnings.warn(
        "cell_to_latlon_batch is deprecated; use cell_to_lonlat_batch, which returns (lon, lat)",
        DeprecationWarning,
        stacklevel=2,
    )
    lon, lat = cell_to_lonlat_batch(cell_ids)
    return lat, lon
