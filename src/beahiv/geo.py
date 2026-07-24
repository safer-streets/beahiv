"""Geographic interface: WGS84 <-> EPSG:27700 <-> cell id.

No spatial queries, no polygon intersection, no R-tree -- coordinate
lookup is a direct projection + arithmetic transform.

Each function accepts either a plain scalar (int/float) or an array-like
(list, numpy array, ...) transparently: scalars take the pure-Python path
below (no numpy import, sub-microsecond per call); anything else dispatches
to the numpy-vectorised equivalent in `beahiv.batch`, which remains
available directly for callers who want an unambiguous vectorised call.
"""

from typing import overload

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

from .batch import (
    cartesian_to_axial_batch,
    cell_centre_batch,
    cell_to_latlon_batch,
    encode_batch,
    latlon_to_cell_batch,
)
from .cell_id import encode
from .coords import cartesian_to_axial
from .geometry import cell_centre
from .orientation import Orientation

_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# EPSG:27700 (British National Grid) area of use -- see
# pyproj.CRS.from_epsg(27700).area_of_use. Outside this box PROJ
# extrapolates instead of erroring, which can produce an (x, y) far
# outside GB and blow past the q/r cell encoding's bit budget (or,
# depending on side_length, silently wrap into a bogus but "valid" cell).
_LAT_MIN, _LAT_MAX = 49.75, 61.01
_LON_MIN, _LON_MAX = -9.01, 2.01


def _check_in_area_of_use(lat: float, lon: float) -> None:
    """Raise if (lat, lon) falls outside EPSG:27700's valid area of use."""
    if not (_LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX):
        raise ValueError(
            f"(lat={lat}, lon={lon}) is outside EPSG:27700's area of use "
            f"(lat in [{_LAT_MIN}, {_LAT_MAX}], lon in [{_LON_MIN}, {_LON_MAX}]) "
            "-- check the arguments aren't swapped"
        )


@overload
def latlon_to_cell(lat: float, lon: float, side_length: int, orientation: Orientation = Orientation.FLAT) -> int: ...
@overload
def latlon_to_cell(
    lat: ArrayLike, lon: ArrayLike, side_length: int, orientation: Orientation = Orientation.FLAT
) -> np.ndarray: ...
def latlon_to_cell(
    lat: ArrayLike,
    lon: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> int | np.ndarray:
    """Encode (lat, lon), scalar or array-like, to a cell id (or array of ids)."""
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        _check_in_area_of_use(lat, lon)
        x, y = _TO_BNG.transform(lon, lat)
        q, r = cartesian_to_axial(x, y, side_length, orientation)
        return encode(q, r, side_length, orientation)
    return latlon_to_cell_batch(lat, lon, side_length, orientation)


@overload
def bng_to_cell(x: float, y: float, side_length: int, orientation: Orientation = Orientation.FLAT) -> int: ...
@overload
def bng_to_cell(
    x: ArrayLike, y: ArrayLike, side_length: int, orientation: Orientation = Orientation.FLAT
) -> np.ndarray: ...
def bng_to_cell(
    x: ArrayLike,
    y: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> int | np.ndarray:
    """Encode an EPSG:27700 (x, y) point directly, with no WGS84 round trip.

    Accepts scalar or array-like (x, y), same dispatch as `latlon_to_cell`.
    """
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        q, r = cartesian_to_axial(x, y, side_length, orientation)
        return encode(q, r, side_length, orientation)
    q, r = cartesian_to_axial_batch(x, y, side_length, orientation)
    return encode_batch(q, r, side_length, orientation)


@overload
def centroid(cell_id: int, latlon: bool = False) -> tuple[float, float]: ...
@overload
def centroid(cell_id: ArrayLike, latlon: bool = False) -> tuple[np.ndarray, np.ndarray]: ...
def centroid(
    cell_id: ArrayLike,
    latlon: bool = False,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Return a cell's centre: EPSG:27700 (x, y) by default, or (lat, lon) if latlon=True.

    Accepts a scalar cell id or an array-like of cell ids (same dispatch as
    `latlon_to_cell`/`bng_to_cell`). For an array, every cell must share the
    same side_length and orientation (see `cell_centre_batch`).
    """
    if isinstance(cell_id, int):
        x, y = cell_centre(cell_id)
        if not latlon:
            return x, y
        lon, lat = _TO_WGS84.transform(x, y)
        return lat, lon
    if not latlon:
        return cell_centre_batch(cell_id)
    return cell_to_latlon_batch(cell_id)
