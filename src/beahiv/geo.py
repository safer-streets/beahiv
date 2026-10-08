"""Geographic interface: WGS84 <-> EPSG:27700 <-> cell id.

No spatial queries, no polygon intersection, no R-tree -- coordinate
lookup is a direct projection + arithmetic transform.

Each function accepts either a plain scalar (int/float) or an array-like
(list, numpy array, ...) transparently: scalars take the pure-Python path
below (no numpy import, sub-microsecond per call); anything else dispatches
to the numpy-vectorised equivalent in `beahiv.batch`, which remains
available directly for callers who want an unambiguous vectorised call.

WGS84 is always (lon, lat) -- x then y -- in arguments and results alike, matching shapely,
GeoJSON and pyproj's `always_xy`. `latlon_to_cell` is the deprecated lat-first spelling.
"""

from typing import TYPE_CHECKING, SupportsIndex, overload
from warnings import deprecated

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

from .batch import (
    bng_to_cell_batch,
    lonlat_to_cell_batch,
)
from .cell_id import decode, encode
from .coords import axial_to_cartesian, cartesian_to_axial
from .orientation import Orientation

if TYPE_CHECKING:  # only for the pyarrow overloads -- never imported at runtime from module scope
    import pyarrow as pa

_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# EPSG:27700 (British National Grid) area of use -- see
# pyproj.CRS.from_epsg(27700).area_of_use. Outside this box PROJ
# extrapolates instead of erroring, which can produce an (x, y) far
# outside GB and blow past the q/r cell encoding's bit budget (or,
# depending on side_length, silently wrap into a bogus but "valid" cell).
_LAT_MIN, _LAT_MAX = 49.75, 61.01
_LON_MIN, _LON_MAX = -9.01, 2.01


def _check_in_area_of_use(lon: float, lat: float) -> None:
    """Raise if (lon, lat) falls outside EPSG:27700's valid area of use."""
    if not (_LON_MIN <= lon <= _LON_MAX and _LAT_MIN <= lat <= _LAT_MAX):
        raise ValueError(
            f"(lon={lon}, lat={lat}) is outside EPSG:27700's area of use "
            f"(lon in [{_LON_MIN}, {_LON_MAX}], lat in [{_LAT_MIN}, {_LAT_MAX}]) "
            "-- check the arguments aren't swapped"
        )


def _is_arrow(value: object) -> bool:
    """True if ``value`` is a pyarrow array (Array, ChunkedArray, ...), without importing pyarrow."""
    return type(value).__module__.startswith("pyarrow")


def _match_arrow(cell_ids: np.ndarray, source: object) -> "np.ndarray | pa.Array":
    """Return the ids as a pyarrow array when the caller passed pyarrow in, else unchanged.

    The import is deliberately lazy and unguarded: ``source`` can only be a pyarrow object if the
    caller's process has already imported pyarrow, so this can never raise ImportError on a path
    that actually runs. That keeps pyarrow out of beahiv's runtime requirements -- the ``arrow``
    extra exists to pin a version and advertise the capability, not to make this import safe.
    Dispatching on the argument (rather than on whether pyarrow is importable) also keeps the return
    type a function of the input alone, so it can't change with how the package was installed.
    """
    if not _is_arrow(source):
        return cell_ids
    import pyarrow as pa

    return pa.array(cell_ids)


@overload
def lonlat_to_cell(lon: float, lat: float, side_length: int, orientation: Orientation = Orientation.FLAT) -> int: ...
@overload
def lonlat_to_cell(
    lon: "pa.Array | pa.ChunkedArray",
    lat: "pa.Array | pa.ChunkedArray",
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> "pa.Array": ...
@overload
def lonlat_to_cell(
    lon: ArrayLike, lat: ArrayLike, side_length: int, orientation: Orientation = Orientation.FLAT
) -> np.ndarray: ...
def lonlat_to_cell(
    lon: ArrayLike,
    lat: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> "int | np.ndarray | pa.Array":
    """Encode WGS84 (lon, lat), scalar or array-like, to a cell id (or array of ids).

    lon first: x then y, as everywhere else in this package.

    A pyarrow array in gives a pyarrow array back (nulls become NaN, and so INVALID_CELL_ID).

    Bounds-checked: raises ValueError if any non-NaN (lon, lat) is outside EPSG:27700's area of use
    (lon in [-9.01, 2.01], lat in [49.75, 61.01]). PROJ extrapolates rather than erroring there, so
    without the check a swapped or garbage coordinate would encode to a wrong cell. The two ranges
    don't overlap, so this also catches lon and lat passed the wrong way round. `bng_to_cell` has
    no such check.
    """
    if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
        _check_in_area_of_use(lon, lat)
        x, y = _TO_BNG.transform(lon, lat)
        q, r = cartesian_to_axial(x, y, side_length, orientation)
        return encode(q, r, side_length, orientation)
    return _match_arrow(lonlat_to_cell_batch(lon, lat, side_length, orientation), lon)


@overload
def latlon_to_cell(lat: float, lon: float, side_length: int, orientation: Orientation = Orientation.FLAT) -> int: ...
@overload
def latlon_to_cell(
    lat: "pa.Array | pa.ChunkedArray",
    lon: "pa.Array | pa.ChunkedArray",
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> "pa.Array": ...
@overload
def latlon_to_cell(
    lat: ArrayLike, lon: ArrayLike, side_length: int, orientation: Orientation = Orientation.FLAT
) -> np.ndarray: ...
@deprecated("latlon_to_cell(lat, lon, ...) is deprecated; use lonlat_to_cell(lon, lat, ...) -- note the argument order")
def latlon_to_cell(
    lat: ArrayLike,
    lon: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> "int | np.ndarray | pa.Array":
    """Deprecated: use `lonlat_to_cell(lon, lat, ...)` -- note the swapped argument order."""
    return lonlat_to_cell(lon, lat, side_length, orientation)


@overload
def bng_to_cell(x: float, y: float, side_length: int, orientation: Orientation = Orientation.FLAT) -> int: ...
@overload
def bng_to_cell(
    x: "pa.Array | pa.ChunkedArray",
    y: "pa.Array | pa.ChunkedArray",
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> "pa.Array": ...
@overload
def bng_to_cell(
    x: ArrayLike, y: ArrayLike, side_length: int, orientation: Orientation = Orientation.FLAT
) -> np.ndarray: ...
def bng_to_cell(
    x: ArrayLike,
    y: ArrayLike,
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> "int | np.ndarray | pa.Array":
    """Encode an EPSG:27700 (x, y) point directly, with no WGS84 round trip.

    Accepts scalar or array-like (x, y), same dispatch as `lonlat_to_cell`: a pyarrow array in
    gives a pyarrow array back (nulls become NaN, and so INVALID_CELL_ID).

    Not bounds-checked, unlike `lonlat_to_cell`: there is no projection here to go wrong. Any
    (x, y) encodes, wherever it is, unless it is far enough out to exceed the q/r bit budget, which
    `encode` rejects. A point outside GB just gets a cell outside GB. The catch is input in the
    wrong units: WGS84 degrees passed as (x, y) are read as metres and give valid but wrong cells
    near the grid origin, without raising.
    """
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        q, r = cartesian_to_axial(x, y, side_length, orientation)
        return encode(q, r, side_length, orientation)
    return _match_arrow(bng_to_cell_batch(x, y, side_length, orientation), x)


def _cell_centre(cell_id: SupportsIndex, lonlat: bool = False) -> tuple[float, float]:
    """Return one cell's centre as (x, y): EPSG:27700 metres, or (lon, lat) when lonlat=True.

    x/y order in both cases -- *not* (lat, lon). This is the numeric core behind
    `geometry.centroid`, whose `Point`s are x/y ordered, so WGS84 comes back lon-first per the
    shapely/GeoJSON convention (the same `always_xy` order pyproj is configured with above).
    Kept here rather than in `geometry.py` purely so the pyproj transformer stays in one module
    (see AGENTS.md on the CRS boundary). Shapely is no longer the reason -- this module may import
    it freely now -- so if that projection rule ever gives, `centroid` can move here whole and
    this helper disappears.

    Scalar only: the vectorised centre lookups are `batch.cell_centre_batch` and
    `batch.cell_to_lonlat_batch`, which `geometry.centroids` uses directly.
    """
    idx = decode(cell_id)
    x, y = axial_to_cartesian(idx.q, idx.r, idx.side_length, idx.orientation)
    if not lonlat:
        return x, y
    return _TO_WGS84.transform(x, y)  # always_xy=True, so this is (lon, lat)
