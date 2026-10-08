"""Encode shapely point geometries -- the geopandas-facing entry point.

`geo.bng_to_cell` takes separate x and y sequences; point data from geopandas arrives as a single
geometry column instead, and splitting it just to pass it back in is both awkward and an extra pass.

Coordinates are read as either EPSG:27700 metres or WGS84 `Point(lon, lat)` -- nothing else. Shapely
geometry carries no CRS of its own (the GEOS SRID slot exists but is unset, and geopandas never
populates it), so which of the two applies comes from the container's `.crs` when a geopandas
object declares one, and from the `lonlat` flag otherwise. The WGS84 leg is handed to
`lonlat_to_cell`/`lonlat_to_cell_batch`, so the projection itself (and its area-of-use check)
still lives only in `geo.py`/`batch.py`.
"""

from typing import overload

import numpy as np
import shapely
from numpy.typing import ArrayLike
from shapely import Point

from .batch import bng_to_cell_batch, lonlat_to_cell_batch
from .cell_id import INVALID_CELL_ID
from .geo import bng_to_cell, lonlat_to_cell
from .orientation import Orientation

_BNG_EPSG = 27700
_WGS84_EPSG = 4326

# shapely.GeometryType: POINT is 0, MISSING (a None entry) is -1, every real non-point type is > 0.
_POINT = 0


def _resolve_lonlat(points: object, lonlat: bool | None) -> bool:
    """Decide whether `points` holds WGS84 lon/lat or EPSG:27700 metres.

    A declared CRS is duck-typed on `.crs` rather than importing geopandas, which beahiv doesn't
    depend on -- and only containers carry one, never the geometry itself. Any CRS other than the
    two supported ones raises rather than being read as one of them: the failure is otherwise
    silent, e.g. WGS84 degrees read as metres put every point within a few metres of the grid
    origin, which encodes to a perfectly valid -- and completely wrong -- cell.
    """
    crs = getattr(points, "crs", None)
    if crs is None:
        return bool(lonlat)
    epsg = crs.to_epsg()
    if epsg == _BNG_EPSG:
        declared = False
    elif epsg == _WGS84_EPSG:
        declared = True
    else:
        raise ValueError(
            f"points are in {crs.name} (EPSG:{epsg if epsg is not None else 'unknown'}), "
            f"not EPSG:{_BNG_EPSG} or EPSG:{_WGS84_EPSG} -- reproject with .to_crs({_BNG_EPSG}) first"
        )
    if lonlat is not None and lonlat != declared:
        raise ValueError(f"lonlat={lonlat} contradicts the declared CRS EPSG:{epsg}")
    return declared


@overload
def point_to_cell(
    points: Point | None, side_length: int, orientation: Orientation = ..., lonlat: bool | None = ...
) -> int: ...
@overload
def point_to_cell(
    points: ArrayLike, side_length: int, orientation: Orientation = ..., lonlat: bool | None = ...
) -> np.ndarray: ...
def point_to_cell(
    points: "ArrayLike | Point | None",
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
    lonlat: bool | None = None,
) -> "int | np.ndarray":
    """Encode shapely point geometry, in EPSG:27700 or WGS84, to cell ids.

    Takes a single `Point` (returning an `int`) or a column of them (returning an `int64` numpy
    array): a geopandas `GeoDataFrame` (its active geometry column is used), a `GeoSeries`, a
    `GeometryArray`, an object ndarray, or a plain list.

    Coordinates are read as either British National Grid metres or WGS84 `Point(lon, lat)` -- the
    x/y order `centroid(lonlat=True)` returns. Which one depends on `lonlat` and on whether the
    input declares a CRS:

    - Declared CRS (a geopandas object with `.crs` set): EPSG:27700 is read as BNG and EPSG:4326 as
      lon/lat; any other CRS raises. Leave `lonlat` as None to go by the CRS. An explicit
      True/False is treated as an assertion and raises if it disagrees.
    - No CRS (a bare `Point`, a list or ndarray of them, or a geopandas object with `.crs` unset):
      shapely geometry carries no CRS, so nothing can be inferred. `lonlat=True` reads lon/lat;
      `lonlat=None` or `False` reads **EPSG:27700**.

    The no-CRS default fails silently for lon/lat input: forgetting `lonlat=True` reads degrees as
    metres, which lands every point within a few metres of the BNG origin and encodes to valid but
    wrong cells. The lon/lat path does not have this problem: it goes through `lonlat_to_cell`'s
    area-of-use check, which also rejects `Point(lat, lon)` written the wrong way round.

    The array form returns numpy rather than a `Series` so that assigning it back
    (`gdf["cell_id"] = point_to_cell(gdf, 100)`) is positional and cannot silently misalign against
    a non-default index.

    Missing (`None`) and empty points give `INVALID_CELL_ID`, matching the NaN handling in
    `bng_to_cell`. Any non-point geometry raises rather than encoding to all-invalid.

    A single point is served for completeness, not for speed: shapely attribute access costs several
    times the encode itself, so a hot scalar loop is better off calling `bng_to_cell(p.x, p.y, ...)`.
    """
    is_lonlat = _resolve_lonlat(points, lonlat)
    if points is None or isinstance(points, Point):
        return _encode_one(points, side_length, orientation, is_lonlat)
    return _encode_many(points, side_length, orientation, is_lonlat)


def _encode_one(point: Point | None, side_length: int, orientation: Orientation, lonlat: bool) -> int:
    if point is None or point.is_empty:
        return INVALID_CELL_ID
    if lonlat:
        return lonlat_to_cell(point.x, point.y, side_length, orientation)
    return bng_to_cell(point.x, point.y, side_length, orientation)


def _encode_many(points: ArrayLike, side_length: int, orientation: Orientation, lonlat: bool) -> np.ndarray:
    geoms = np.asarray(getattr(points, "geometry", points), dtype=object)

    type_ids = shapely.get_type_id(geoms)
    if np.any(type_ids > _POINT):
        found = ", ".join(shapely.GeometryType(int(i)).name for i in np.unique(type_ids[type_ids > _POINT]))
        raise ValueError(f"point_to_cell requires point geometries, got {found}")

    # get_x raises on an empty point, so the mask has to be applied before the read, not after.
    # get_coordinates would be the obvious alternative and is a trap: it drops empty geometries
    # entirely, silently returning fewer rows than there were points.
    present = (type_ids == _POINT) & ~shapely.is_empty(geoms)
    if present.all():
        # Reading straight through is ~2x the masked path below (an object-array gather plus a
        # scatter into pre-filled buffers), and a column with nothing missing is the normal case.
        x, y = shapely.get_x(geoms), shapely.get_y(geoms)
    else:
        x = np.full(geoms.shape, np.nan)
        y = np.full(geoms.shape, np.nan)
        x[present] = shapely.get_x(geoms[present])
        y[present] = shapely.get_y(geoms[present])

    if lonlat:
        return lonlat_to_cell_batch(x, y, side_length, orientation)
    return bng_to_cell_batch(x, y, side_length, orientation)
