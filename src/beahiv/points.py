"""Encode shapely point geometries -- the geopandas-facing entry point.

`geo.bng_to_cell` takes separate x and y sequences; point data from geopandas arrives as a single
geometry column instead, and splitting it just to pass it back in is both awkward and an extra pass.

Coordinates are always read as EPSG:27700 metres. Shapely geometry carries no CRS of its own (the
GEOS SRID slot exists but is unset, and geopandas never populates it), so there is nothing here to
reproject *from* -- a caller in another CRS uses `.to_crs(27700)`, or `latlon_to_cell` for WGS84.
The only CRS logic is a guard against a geopandas object that says outright it holds something else.

Along with `polyfill.py` this is one of only two modules allowed to import shapely.
"""

from typing import overload

import numpy as np
import shapely
from numpy.typing import ArrayLike
from shapely import Point

from .batch import bng_to_cell_batch
from .cell_id import INVALID_CELL_ID
from .geo import bng_to_cell
from .orientation import Orientation

_BNG_EPSG = 27700

# shapely.GeometryType: POINT is 0, MISSING (a None entry) is -1, every real non-point type is > 0.
_POINT = 0


def _check_crs(points: object) -> None:
    """Raise if `points` declares a CRS that isn't EPSG:27700.

    Duck-typed on `.crs` rather than importing geopandas, which beahiv doesn't depend on -- and only
    containers carry one, never the geometry itself. Worth the check because the failure is
    otherwise silent: WGS84 degrees read as metres put every point within a few metres of the grid
    origin, which encodes to a perfectly valid -- and completely wrong -- cell rather than erroring.
    """
    crs = getattr(points, "crs", None)
    if crs is None:
        return
    epsg = crs.to_epsg()
    if epsg != _BNG_EPSG:
        raise ValueError(
            f"points are in {crs.name} (EPSG:{epsg if epsg is not None else 'unknown'}), "
            f"not EPSG:{_BNG_EPSG} -- reproject with .to_crs({_BNG_EPSG}) first"
        )


@overload
def point_to_cell(points: Point | None, side_length: int, orientation: Orientation = ...) -> int: ...
@overload
def point_to_cell(points: ArrayLike, side_length: int, orientation: Orientation = ...) -> np.ndarray: ...
def point_to_cell(
    points: "ArrayLike | Point | None",
    side_length: int,
    orientation: Orientation = Orientation.FLAT,
) -> "int | np.ndarray":
    """Encode EPSG:27700 shapely point geometry to cell ids.

    Takes a single `Point` (returning an `int`) or a column of them (returning a `uint64` numpy
    array): a geopandas `GeoDataFrame` (its active geometry column is used), a `GeoSeries`, a
    `GeometryArray`, an object ndarray, or a plain list. Nothing is reprojected -- coordinates must
    already be British National Grid metres, and a geopandas object declaring any other CRS raises.

    The array form returns numpy rather than a `Series` so that assigning it back
    (`gdf["cell_id"] = point_to_cell(gdf, 100)`) is positional and cannot silently misalign against
    a non-default index.

    Missing (`None`) and empty points give `INVALID_CELL_ID`, matching the NaN handling in
    `bng_to_cell`. Any non-point geometry raises rather than encoding to all-invalid.

    A single point is served for completeness, not for speed: shapely attribute access costs several
    times the encode itself, so a hot scalar loop is better off calling `bng_to_cell(p.x, p.y, ...)`.
    """
    _check_crs(points)
    if points is None or isinstance(points, Point):
        return _encode_one(points, side_length, orientation)
    return _encode_many(points, side_length, orientation)


def _encode_one(point: Point | None, side_length: int, orientation: Orientation) -> int:
    if point is None or point.is_empty:
        return INVALID_CELL_ID
    return bng_to_cell(point.x, point.y, side_length, orientation)


def _encode_many(points: ArrayLike, side_length: int, orientation: Orientation) -> np.ndarray:
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

    return bng_to_cell_batch(x, y, side_length, orientation)
