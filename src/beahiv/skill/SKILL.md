---
name: beahiv
description: >-
  Use when writing or reviewing Python that indexes Great Britain spatial data with the `beahiv`
  hexagonal grid: turning points (lon/lat, EPSG:27700 x/y, Shapely or geopandas geometry) into
  cell ids, aggregating to hex cells, building cell polygons/centroids, neighbours and k-rings,
  filling a boundary polygon with cells, resizing between cell sizes, or using beahiv inside
  DuckDB. Also use when you see `import beahiv`, beahiv cell ids (int64), or "BEAHIV".
---

# beahiv

beahiv is a hexagonal spatial index for Great Britain, native to **EPSG:27700** (British National
Grid metres). Think of it as a local H3 with exact, flat-plane geometry. Each cell is a single
`int64` id that encodes `(q, r, size, measure, orientation)`, and every cell at a given size has
the same area.

```python
import beahiv
from beahiv import Orientation
```

## Rules to get right

1. **WGS84 is always (lon, lat), x then y.** That applies to arguments (`lonlat_to_cell(lon, lat, ...)`),
   returned tuples (`batch.cell_to_lonlat_batch` gives `(lon, lat)`) and `Point(lon, lat)`. Never
   pass lat first. `latlon_to_cell`, `batch.latlon_to_cell_batch` and `batch.cell_to_latlon_batch`
   are deprecated lat-first spellings. Don't use them in new code, and switch existing calls to the
   `lonlat` names, swapping the argument order as you do. `centroid`/`centroids` take `lonlat=`;
   the old `latlon=` keyword is gone and raises `TypeError`.
2. **Everything is EPSG:27700 metres unless you ask for WGS84.** `cell_polygon(s)` always returns
   EPSG:27700. `centroid(s)` does too, unless you pass `lonlat=True`. Set `crs=27700` on any GeoDataFrame you build from them.
3. **Pick the right entry point for your input:**
   - lon/lat numbers: `lonlat_to_cell`. It checks bounds and raises outside GB
     (lon −9.01–2.01, lat 49.75–61.01), which also catches swapped axes.
   - EPSG:27700 x/y numbers: `bng_to_cell`. It has **no bounds check**, so degrees passed in here
     silently give wrong cells near the grid origin.
   - Shapely or geopandas points: `point_to_cell`. A declared container CRS of 27700 or 4326 is
     honoured, and any other CRS raises (reproject with `.to_crs(27700)` first). Bare `Point`s
     and lists carry no CRS and are read as EPSG:27700 unless you pass `lonlat=True`.
4. **Size is keyword-only: exactly one of `side_length=` or `side_to_side=`.** `side_length` is
   the length of an edge and `side_to_side` the distance between two parallel edges. Either is a
   whole number of metres from 1 to 100,000, and must be an `int`, not a float. There is no
   resolution table: `side_length=200` means 200 m edges. Passing both, neither, or a bare
   positional number raises `TypeError`. `orientation` (default `Orientation.FLAT`) and
   `predicate` are keyword-only too. The id records which measure was used (`SizeMeasure`), so
   decoded cells keep it. Pick one size, measure and orientation per analysis and pass them
   everywhere.
5. **Different grids don't mix.** Never compare or combine ids across sizes, measures or
   orientations. FLAT and POINTY are different lattices, and so are `side_length=200` and
   `side_to_side=200`. `distance`, `cell_polygons` and `centroids` need every id to share one
   size, measure and orientation.
6. **Sizes don't nest.** Cells of different sizes don't tile each other. Use `resize_cell`
   for a covering at any other size. `get_parents`/`get_children` (overlapping cells at 2x/0.5x)
   and `get_parent`/`get_child` (same-centroid cell at 2x/0.5x, often `None`) are factor-of-two
   only.
7. **Missing input becomes `INVALID_CELL_ID` (0).** NaN coordinates and `None` or empty points
   encode to it rather than raising. `decode` and anything that decodes (`centroids`,
   `cell_polygons`, `get_neighbours`, ...) **rejects** it, so filter it out first:
   `ids[ids != beahiv.INVALID_CELL_ID]`.
8. **Scalar or array dispatch is automatic** for `lonlat_to_cell`, `bng_to_cell`, `point_to_cell`,
   `get_parents` and `get_children`. A Python `float`/`int` (or a single `Point`) gives a Python
   `int`. A list, ndarray, Series or pyarrow array gives an `int64` ndarray, or a pyarrow array
   if pyarrow went in. Arrays come back positional, never as a `Series`, so assign them back with
   `df["cell_id"] = ...`. Shapely-returning functions come in singular/plural pairs instead:
   `centroid` takes one id and `centroids` many, and the same goes for `cell_polygon` and
   `cell_polygons`.
9. **Store ids as signed 64-bit** (`int64`, DuckDB `BIGINT`, Arrow `int64`). Every id fits a
   signed int64. Scalar functions accept `np.int64` ids directly.
10. **Ids from beahiv before 0.1.0 are not compatible.** 0.1.0 moved the grid origin, so an older
    id still decodes, but to a different place. Regenerate stored ids rather than mixing them.

## Recipes

Points in a DataFrame to cells:

```python
df["cell_id"] = beahiv.lonlat_to_cell(df["lon"], df["lat"], side_length=200)
df["cell_id"] = beahiv.bng_to_cell(df["easting"], df["northing"], side_length=200)  # if already BNG
```

A GeoDataFrame of points to cells. The CRS must be 27700 or 4326:

```python
gdf["cell_id"] = beahiv.point_to_cell(gdf, side_length=200)
```

Aggregate to a hex GeoDataFrame:

```python
import geopandas as gpd
import pandas as pd

valid = df[df["cell_id"] != beahiv.INVALID_CELL_ID]
counts = valid.groupby("cell_id").size().rename("n").reset_index()
hexes = gpd.GeoDataFrame(counts, geometry=beahiv.cell_polygons(counts["cell_id"]), crs=27700)
```

Cover a boundary with cells, so that empty cells show up as zeros:

```python
boundary = area_gdf.to_crs(27700).union_all()  # polyfill needs EPSG:27700 Shapely geometry
cells = pd.DataFrame({"cell_id": beahiv.polyfill(boundary, side_length=200)})
full = cells.merge(counts, on="cell_id", how="left").fillna({"n": 0})
```

The `predicate` argument to `polyfill`/`bbox_fill`/`resize_cell` decides which cells are kept:
`"overlap"` (any overlap, the default for `polyfill`/`bbox_fill`), `"centre"`/`"center"` (the
cell centre is inside, the default for `resize_cell`) or `"full"` (wholly inside). To fill a
bounding box without building a polygon, use `bbox_fill(minx, miny, maxx, maxy, side_length=...)`.

Neighbourhoods and smoothing:

```python
beahiv.get_neighbours(cell_id)  # 6 ids
beahiv.k_ring(cell_id, 2)  # 1 + 3k(k+1) = 19 ids, including cell_id
beahiv.distance(cell_a, cell_b)  # hops; same grid only
```

Change resolution:

```python
beahiv.resize_cell(cell_id, new_side_length=50)  # any size, smaller or larger
beahiv.resize_cell(cell_id, new_side_to_side=100)  # or by side-to-side distance
beahiv.get_children(cell_id)  # 7 cells at half the size (needs an even size)
beahiv.get_parents(cell_ids)  # array in: deduplicated union of 2x cells
```

Centres and decoding:

```python
beahiv.centroid(cell_id)  # Point in EPSG:27700
beahiv.centroid(cell_id, lonlat=True)  # Point(lon, lat)
beahiv.centroids(cell_ids, lonlat=True)  # list[Point]
beahiv.decode(cell_id)  # CellIndex(q, r, size, measure, orientation); .side_length is geometric
beahiv.encode(q, r, side_length=200, orientation=Orientation.FLAT)
beahiv.encode(q, r, side_to_side=200)  # measure recorded as SizeMeasure.SIDE_TO_SIDE
```

For plain coordinate columns rather than `Point`s, use `beahiv.batch.cell_centre_batch(ids)`,
which returns `(x, y)` arrays, or `beahiv.batch.cell_to_lonlat_batch(ids)`, which returns
`(lon, lat)` arrays.

`encode_morton`/`decode_morton` produce Z-order ids with the same fields. They are useful as a
sort or cluster key, for example DuckDB or Parquet ordering. Morton ids are not interchangeable
with plain ids, and nothing can tell the two apart, so keep to one kind per column.

## DuckDB

Exchange geometry as WKB. For vectorised UDFs, use `type="arrow"` and pass the plural functions:

```python
import shapely
from duckdb.sqltypes import BIGINT, BLOB, DOUBLE

con.create_function(
    "bh_cell", lambda x, y: beahiv.bng_to_cell(x, y, side_length=200), [DOUBLE, DOUBLE], BIGINT, type="arrow"
)
con.create_function(
    "bh_cell_polygon", lambda ids: shapely.to_wkb(beahiv.cell_polygons(ids)), [BIGINT], BLOB, type="arrow"
)
# SQL: SELECT cell_id, ST_GeomFromWKB(bh_cell_polygon(cell_id)) AS geometry FROM ...
# then: gpd.GeoDataFrame.from_arrow(con.sql(...).arrow()).set_crs(27700)
```

The UDF has to return WKB as `BLOB` and be parsed with `ST_GeomFromWKB` in SQL, because DuckDB
can't cast a `BLOB` return to `GEOMETRY`. WKB carries no SRID, so set the CRS after the data
arrives in Python.

Bind the size and `orientation` in the lambda rather than passing them as UDF arguments. If the
size has to come from SQL, take the measure as a string equal to the Python keyword and forward it
as `**{measure: size}`, so that a misspelt measure raises instead of being read as a side length.
`bh_cell_polygon` raises if one chunk mixes grids, so use a per-row `cell_polygon` UDF if a
column can do that.

## Errors and what they mean

- `ValueError: ... outside EPSG:27700's area of use`: the lon/lat is outside GB, or the axes are swapped.
- `ValueError: points are in ... not EPSG:27700 or EPSG:4326`: reproject the GeoDataFrame first.
- `ValueError: lonlat=... contradicts the declared CRS`: drop the `lonlat` argument and let the
  container's CRS decide.
- `ValueError` from `decode`: the id is `INVALID_CELL_ID`, a Morton or foreign id, or corrupt.
- `TypeError: a cell size is required`, `... not both`, or `takes N positional arguments`: pass
  exactly one of `side_length=`/`side_to_side=`, by keyword.
- `ValueError: ... requires a single size / size measure / orientation per call` or `distance
  requires both cells to share ...`: ids from different grids were mixed.
- `ValueError: no 0.5x size cell ... is odd`: `get_child`/`get_children` need an even size.
- `TypeError` mentioning `centroids`/`centroid for one`: you used the singular form with an array, or the reverse.
