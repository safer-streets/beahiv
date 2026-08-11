"""Vectorised numpy variants for encoding/decoding many points at once.

Single-point lookups (geo.py, cell_id.py) already meet the < 1us target
without numpy. This module exists for the bulk case explicitly called
out in the spec -- encoding a whole crime dataset at once -- where a
Python-level loop dominates and numpy vectorisation matters.
"""

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
    UINT64_MASK,
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


def _check_in_area_of_use_batch(lats: np.ndarray, lons: np.ndarray, valid: np.ndarray) -> None:
    """Raise if any non-NaN (lat, lon) falls outside EPSG:27700's area of use."""
    in_bounds = (lats >= _LAT_MIN) & (lats <= _LAT_MAX) & (lons >= _LON_MIN) & (lons <= _LON_MAX)
    bad = valid & ~in_bounds
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"(lat={lats[i]}, lon={lons[i]}) at index {i} is outside EPSG:27700's "
            f"area of use (lat in [{_LAT_MIN}, {_LAT_MAX}], lon in [{_LON_MIN}, {_LON_MAX}]) "
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
    q = np.asarray(q, dtype=np.int64)
    r = np.asarray(r, dtype=np.int64)

    # side_length doesn't fill its field, so an oversized one would overflow into
    # the orientation bit rather than being masked off -- one scalar check per
    # call, not per row, so the vectorised path pays nothing for it
    if not (1 <= side_length <= SIDE_LENGTH_MAX):
        raise ValueError(f"side_length must be in [1, {SIDE_LENGTH_MAX}], got {side_length}")

    q_enc = (q + Q_OFFSET).astype(np.uint64)
    r_enc = (r + R_OFFSET).astype(np.uint64)

    if np.any(q_enc > Q_MASK) or np.any(r_enc > R_MASK):
        raise ValueError("q or r out of representable range")

    cell_ids = (
        (np.uint64(int(orientation)) << np.uint64(ORIENTATION_SHIFT))
        | (np.uint64(side_length) << np.uint64(SIDE_LENGTH_SHIFT))
        | (q_enc << np.uint64(Q_SHIFT))
        | r_enc
    )
    return cell_ids & np.uint64(UINT64_MASK)


def decode_batch(cell_ids: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (q, r, side_length, orientation) arrays for a batch of cell ids.

    Rejects the same ids scalar `decode` does (see `cell_id.validate_cell_id`), vectorised: two
    array comparisons for the whole batch rather than a per-row call. Note that the *_to_cell
    functions emit INVALID_CELL_ID for missing input, so an array built from data with gaps must
    have them filtered out before it can be decoded.
    """
    cell_ids = np.asarray(cell_ids, dtype=np.uint64)

    orientation = ((cell_ids >> np.uint64(ORIENTATION_SHIFT)) & np.uint64(ORIENTATION_MASK)).astype(np.uint8)
    side_length = ((cell_ids >> np.uint64(SIDE_LENGTH_SHIFT)) & np.uint64(SIDE_LENGTH_MASK)).astype(np.int64)

    reserved = (cell_ids >> np.uint64(RESERVED_SHIFT)) & np.uint64(RESERVED_MASK)
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

    q_enc = ((cell_ids >> np.uint64(Q_SHIFT)) & np.uint64(Q_MASK)).astype(np.int64)
    r_enc = (cell_ids & np.uint64(R_MASK)).astype(np.int64)

    return q_enc - Q_OFFSET, r_enc - R_OFFSET, side_length, orientation


def latlon_to_cell_batch(
    lats: ArrayLike,
    lons: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Encode each (lat, lon) pair; NaN coordinates map to INVALID_CELL_ID."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    valid = ~(np.isnan(lats) | np.isnan(lons))
    _check_in_area_of_use_batch(lats, lons, valid)

    cell_ids = np.full(lats.shape, INVALID_CELL_ID, dtype=np.uint64)
    if np.any(valid):
        x, y = _TO_BNG.transform(lons[valid], lats[valid])
        q, r = cartesian_to_axial_batch(x, y, side_length, orientation)
        cell_ids[valid] = encode_batch(q, r, side_length, orientation)
    return cell_ids


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

    cell_ids = np.full(x.shape, INVALID_CELL_ID, dtype=np.uint64)
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


def cell_to_latlon_batch(cell_ids: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x, y = cell_centre_batch(cell_ids)
    lon, lat = _TO_WGS84.transform(x, y)
    return lat, lon
