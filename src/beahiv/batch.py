"""Vectorised numpy variants for encoding/decoding many points at once.

Single-point lookups (geo.py, cell_id.py) already meet the < 1us target
without numpy. This module exists for the bulk case explicitly called
out in the spec -- encoding a whole crime dataset at once -- where a
Python-level loop dominates and numpy vectorisation matters.
"""

from collections.abc import Sequence
from operator import index
from typing import SupportsIndex
from warnings import deprecated

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

from .cell_id import (
    INVALID_CELL_ID,
    MEASURE_MASK,
    MEASURE_SHIFT,
    ORIENTATION_MASK,
    ORIENTATION_SHIFT,
    Q_MASK,
    Q_OFFSET,
    Q_SHIFT,
    R_MASK,
    R_OFFSET,
    RESERVED_SHIFT,
    SIDE_LENGTH_MASK,
    SIDE_LENGTH_SHIFT,
    SIZE_EXPONENT_MAX,
    SIZE_MANTISSA_BITS,
    SIZE_MANTISSA_MASK,
    SIZE_MANTISSA_MAX,
    SIZE_MIN,
    pack_size,
    resolve_size,
)
from .coords import ORIGIN_X, ORIGIN_Y, SQRT3, side_length_of
from .measure import SizeMeasure
from .orientation import Orientation

# A column of cell ids, as the batch decoders take it. ArrayLike alone has no room for a None among
# the ints, and a missing id is something they accept -- see `_as_cell_id_array`.
type CellIdsLike = ArrayLike | Sequence[SupportsIndex | None]

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
    """Vectorised `coords.axial_to_cartesian`; `side_length` is likewise the geometric one."""
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    s = side_length
    if orientation == Orientation.POINTY:
        x = s * SQRT3 * (q + r / 2.0)
        y = 1.5 * s * r
    else:
        x = 1.5 * s * q
        y = s * SQRT3 * (r + q / 2.0)
    return ORIGIN_X + x, ORIGIN_Y + y


def cartesian_to_axial_batch(
    x: ArrayLike,
    y: ArrayLike,
    side_length: float,
    orientation: Orientation = Orientation.FLAT,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64) - ORIGIN_X
    y = np.asarray(y, dtype=np.float64) - ORIGIN_Y
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
    *,
    side_length: int | None = None,
    side_to_side: int | None = None,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Encode axial coordinates into int64 cell ids -- see `cell_id.encode` for the layout.

    Give exactly one of `side_length` / `side_to_side`, as for `cell_id.encode`.
    """
    q = np.asarray(q, dtype=np.int64)
    r = np.asarray(r, dtype=np.int64)
    size, measure = resolve_size(side_length, side_to_side)
    # one scalar size for the whole call, so it is packed (and validated) once, not per row
    size_field = pack_size(size, measure)

    q_enc = q + Q_OFFSET
    r_enc = r + R_OFFSET

    if np.any((q_enc < 0) | (q_enc > Q_MASK)) or np.any((r_enc < 0) | (r_enc > R_MASK)):
        raise ValueError("q or r out of representable range")

    # int64, never uint64: a signed dtype is what a BIGINT column holds without conversion
    return (
        (np.int64(int(measure)) << MEASURE_SHIFT)
        | (np.int64(int(orientation)) << ORIENTATION_SHIFT)
        | (np.int64(size_field) << SIDE_LENGTH_SHIFT)
        | (q_enc << Q_SHIFT)
        | r_enc
    )


_POW10 = np.array([10**e for e in range(SIZE_EXPONENT_MAX + 1)], dtype=np.int64)
_SIZE_MIN = np.array(SIZE_MIN, dtype=np.int64)
# a float id is exact below 2**53 (every valid one is); anything at or above can't be one anyway
_FLOAT_ID_LIMIT = 2.0**53


def _float_ids(ids: np.ndarray) -> np.ndarray:
    whole = np.isfinite(ids) & (ids == np.floor(ids)) & (np.abs(ids) < _FLOAT_ID_LIMIT)
    return np.where(whole, ids, INVALID_CELL_ID).astype(np.int64)


def _object_ids(ids: np.ndarray) -> np.ndarray:
    out = np.full(ids.shape, INVALID_CELL_ID, dtype=np.int64)
    for i, value in np.ndenumerate(ids):
        if isinstance(value, (float, np.floating)):
            out[i] = _float_ids(np.array([value], dtype=np.float64))[0]
            continue
        try:
            value = index(value)
        except TypeError:  # None, pd.NA, or anything else that can't be an id
            continue
        # too big for int64 means bit 63 or above is set, which only a reserved bit could be
        if -(2**63) <= value < 2**63:
            out[i] = value
    return out


def _as_cell_id_array(cell_ids: CellIdsLike) -> np.ndarray:
    """Coerce an array-like of cell ids to int64, with every missing entry as INVALID_CELL_ID.

    Floats are accepted: every valid id is below 2**52, so one that has been through a float64 (a
    pandas int64 column that gained a NaN, a JSON number) comes back exact. NaN is missing, and a
    non-integral or out-of-range float can't be an id, so it is treated the same way. Nulls are
    still filled via the duck-typed `fillna` (pandas) / `fill_null` (pyarrow, polars) before numpy
    sees the data, which keeps a nullable integer column integer rather than relying on that.

    uint64 is reinterpreted as int64, which turns an id with bit 63 set negative, where the
    reserved-bit check in `decode_batch` sees it.
    """
    fill = getattr(cell_ids, "fillna", None) or getattr(cell_ids, "fill_null", None)
    if fill is not None:
        cell_ids = fill(INVALID_CELL_ID)
    ids = np.asarray(cell_ids)
    if isinstance(cell_ids, (list, tuple)) and ids.dtype.kind not in "iu":
        # numpy infers float64 for a list mixing ints with a NaN *or* with one int >= 2**63, which
        # would round the big int -- read as objects, the Python ints stay exact
        ids = np.asarray(cell_ids, dtype=object)
    if ids.dtype.kind in "iu":
        return ids.astype(np.int64, copy=False)
    if ids.dtype.kind == "f":
        return _float_ids(ids)
    if ids.dtype.kind == "O":
        return _object_ids(ids)
    raise TypeError(f"cell ids must be integers, got an array of {ids.dtype}")


def decode_batch(
    cell_ids: CellIdsLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (q, r, size, measure, orientation) arrays for a batch of cell ids.

    `size` is the stored field, as `CellIndex.size` -- *not* the geometric side length when the
    measure is SIDE_TO_SIDE (see `coords.side_length_of`).

    Unlike scalar `decode`, a bad row doesn't raise: every id `decode` would reject (see
    `cell_id.validate_cell_id`) -- INVALID_CELL_ID, a missing value, reserved bits set, an invalid
    size field -- decodes to all zeros instead, exactly as INVALID_CELL_ID's own bits read.
    `size == 0` is therefore the validity mask; no valid id has it.
    """
    cell_ids = _as_cell_id_array(cell_ids)

    orientation = ((cell_ids >> ORIENTATION_SHIFT) & ORIENTATION_MASK).astype(np.uint8)
    measure = ((cell_ids >> MEASURE_SHIFT) & MEASURE_MASK).astype(np.uint8)
    size_field = (cell_ids >> SIDE_LENGTH_SHIFT) & SIDE_LENGTH_MASK
    exponent = size_field >> SIZE_MANTISSA_BITS
    mantissa = size_field & SIZE_MANTISSA_MASK
    size = mantissa * _POW10[exponent]
    q = ((cell_ids >> Q_SHIFT) & Q_MASK) - Q_OFFSET
    r = (cell_ids & R_MASK) - R_OFFSET

    # mirrors cell_id.unpack_size and validate_cell_id. A negative id has bit 63 set, which the
    # arithmetic shift keeps, so `>> RESERVED_SHIFT` is non-zero for it without a sign check
    invalid = (
        ((cell_ids >> RESERVED_SHIFT) != 0)
        | (mantissa < 1)
        | (mantissa > SIZE_MANTISSA_MAX)
        | ((exponent > 0) & (mantissa < 100))
        | (size < _SIZE_MIN[measure])
    )
    if np.any(invalid):
        for field in (q, r, size, measure, orientation):
            field[invalid] = 0

    return q, r, size, measure, orientation


def lonlat_to_cell_batch(
    lons: ArrayLike,
    lats: ArrayLike,
    *,
    side_length: int | None = None,
    side_to_side: int | None = None,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Encode each WGS84 (lon, lat) pair; NaN coordinates map to INVALID_CELL_ID.

    Give exactly one of `side_length` / `side_to_side`, as for `cell_id.encode`.

    Raises if any non-NaN pair is outside EPSG:27700's area of use -- see `geo.lonlat_to_cell`.
    """
    s = side_length_of(*resolve_size(side_length, side_to_side))
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    valid = ~(np.isnan(lons) | np.isnan(lats))
    _check_in_area_of_use_batch(lons, lats, valid)

    cell_ids = np.full(lons.shape, INVALID_CELL_ID, dtype=np.int64)
    if np.any(valid):
        x, y = _TO_BNG.transform(lons[valid], lats[valid])
        q, r = cartesian_to_axial_batch(x, y, s, orientation)
        cell_ids[valid] = encode_batch(
            q, r, side_length=side_length, orientation=orientation, side_to_side=side_to_side
        )
    return cell_ids


@deprecated(
    "latlon_to_cell_batch(lats, lons, ...) is deprecated; use lonlat_to_cell_batch(lons, lats, ...) "
    "-- note the argument order"
)
def latlon_to_cell_batch(
    lats: ArrayLike,
    lons: ArrayLike,
    *,
    side_length: int | None = None,
    side_to_side: int | None = None,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Deprecated: use `lonlat_to_cell_batch(lons, lats, ...)` -- note the swapped argument order."""
    return lonlat_to_cell_batch(lons, lats, side_length=side_length, side_to_side=side_to_side, orientation=orientation)


def bng_to_cell_batch(
    x: ArrayLike,
    y: ArrayLike,
    *,
    side_length: int | None = None,
    side_to_side: int | None = None,
    orientation: Orientation = Orientation.FLAT,
) -> np.ndarray:
    """Encode each EPSG:27700 (x, y) pair; NaN coordinates map to INVALID_CELL_ID.

    No area-of-use check: unlike the lat/lon path there is no projection to extrapolate, so a
    coordinate far outside GB is simply a cell far from the origin — and one far enough to exceed
    the q/r bit budget is rejected by ``encode_batch`` rather than wrapping silently.

    Give exactly one of `side_length` / `side_to_side`, as for `cell_id.encode`.
    """
    s = side_length_of(*resolve_size(side_length, side_to_side))
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))

    cell_ids = np.full(x.shape, INVALID_CELL_ID, dtype=np.int64)
    if np.any(valid):
        q, r = cartesian_to_axial_batch(x[valid], y[valid], s, orientation)
        cell_ids[valid] = encode_batch(
            q, r, side_length=side_length, orientation=orientation, side_to_side=side_to_side
        )
    return cell_ids


def _cell_centres(cell_ids: CellIdsLike) -> tuple[np.ndarray, np.ndarray, float, Orientation]:
    """`cell_centre_batch`, plus the geometric side length and the orientation of the one grid it found.

    `geometry.cell_polygons` needs those too, and they are only knowable after decoding. With no
    valid row there is no grid at all, and the side length comes back 0.
    """
    q, r, size, measure, orientation = decode_batch(cell_ids)
    valid = size > 0

    orientations = np.unique(orientation[valid])
    if len(orientations) > 1:
        raise ValueError("cell_centre_batch requires a single orientation per call")
    sizes = np.unique(size[valid])
    if len(sizes) > 1:
        raise ValueError("cell_centre_batch requires a single size per call")
    measures = np.unique(measure[valid])
    if len(measures) > 1:
        raise ValueError("cell_centre_batch requires a single size measure per call")

    x = np.full(q.shape, np.nan)
    y = np.full(q.shape, np.nan)
    if not len(sizes):
        return x, y, 0.0, Orientation.FLAT
    s = side_length_of(int(sizes[0]), SizeMeasure(int(measures[0])))
    grid_orientation = Orientation(int(orientations[0]))
    x[valid], y[valid] = axial_to_cartesian_batch(q[valid], r[valid], s, grid_orientation)
    return x, y, s, grid_orientation


def cell_centre_batch(cell_ids: CellIdsLike) -> tuple[np.ndarray, np.ndarray]:
    """Return EPSG:27700 (x, y) centres for a batch of cell ids.

    A missing or invalid id (anything `decode_batch` zeroes) gets a NaN centre rather than raising.

    Every valid cell must share the same size, orientation and measure -- a single
    axial_to_cartesian_batch call can't mix them.
    """
    x, y, _, _ = _cell_centres(cell_ids)
    return x, y


def cell_to_lonlat_batch(cell_ids: CellIdsLike) -> tuple[np.ndarray, np.ndarray]:
    """Return WGS84 (lon, lat) centres for a batch of cell ids -- lon first, as `_TO_WGS84` emits.

    NaN for a missing or invalid id, as in `cell_centre_batch`.
    """
    x, y = cell_centre_batch(cell_ids)
    valid = ~np.isnan(x)
    lon = np.full(x.shape, np.nan)
    lat = np.full(x.shape, np.nan)
    if np.any(valid):
        lon[valid], lat[valid] = _TO_WGS84.transform(x[valid], y[valid])
    return lon, lat


@deprecated("cell_to_latlon_batch is deprecated; use cell_to_lonlat_batch, which returns (lon, lat)")
def cell_to_latlon_batch(cell_ids: CellIdsLike) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated: use `cell_to_lonlat_batch`, which returns (lon, lat) -- the reverse of this."""
    lon, lat = cell_to_lonlat_batch(cell_ids)
    return lat, lon
