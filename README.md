# BEAHIV

**B**ritish National Grid **E**qual-**A**rea **H**exagonal **I**ndex that's **V**ersatile.

A fast, equal-area hexagonal spatial index native to the British
National Grid (BNG). BEAHIV is a local alternative to H3: it trades global
coverage for exact, deterministic geometry, and is built for regional
analysis within Great Britain — crime analysis, spatial statistics,
regional-scale modelling.

Each letter maps onto something the index actually does:

- **British** — native to BNG (**EPSG:27700**), no spherical geometry anywhere (BNG **does not** cover Northern Ireland
or the Channel Islands).
- **Equal-Area** — every cell has identical area in the BNG plane (on the ground they differ by
  up to ~0.25% — see [Technical details](#technical-details)).
- **Hexagonal** — a regular hexagonal lattice, pointy-top or flat-top.
- **Index** — O(1) coordinate → cell, O(1) neighbour, O(1) distance,
  with cell coordinates offset-biased directly into the id
  (`q_enc = q + Q_OFFSET`) so decoding needs no lookup table.
- **Versatile** — side length and the q/r offset are both parameters
  baked straight into the bits, not values looked up from a table, so
  neither is fixed by the format itself.


## Philosophy

BEAHIV intentionally rejects the compromises made by global spherical hex
systems. The grid is topologically regular, geometrically regular, equal
area, deterministic, and projection-native. For regional analysis in
Great Britain, equal area matters more than global continuity — the
result is a local, equal-area analogue of H3, optimised for statistical
analysis rather than global indexing.

## Design goals

- Equal-area cells — every hex has identical area in the projected plane
- Single neighbour relationship, six neighbours per cell
- Deterministic coordinate → cell mapping, no lookup tables
- No spatial joins, no point-in-polygon queries, no R-tree
- O(1) coordinate lookup, O(1) neighbour lookup, O(1) distance
- Compact 64-bit cell identifiers that survive conversion to float64, so
  JSON serialisation and pandas' coercion of an id column with missing
  values give back the same cell (see [Cell identifiers](#cell-identifiers))
- No spherical geometry, no pentagons, no varying cell sizes

This project deliberately avoids global coverage in favour of local
geometric correctness.

### Why not H3?

H3 is built to tile the whole globe, and every one of its quirks traces
back to that requirement — none of them are relevant once you only care
about Great Britain:

- **Unavoidable pentagons.** H3 projects a hexagonal grid onto an
  icosahedron and wraps that around the sphere. Euler's formula makes it
  topologically impossible to cover a sphere with hexagons alone — 12
  pentagon cells are forced in at every resolution. Pentagons have five
  neighbours, not six, breaking the single-neighbour-relationship
  guarantee and complicating any code that assumes uniform adjacency.
- **Cells aren't actually equal-area.** The icosahedron-to-sphere
  projection distorts differently depending on where a cell falls
  relative to icosahedron edges and vertices, so H3 cell areas vary by
  a non-trivial margin at a given resolution — "equal-area" is only ever
  approximate. A hex grid defined directly in a flat, projected CRS has
  no such distortion: area is exactly `(3√3/2)s²` everywhere, by
  construction.
- **Spherical geometry throughout.** Coordinate lookup, neighbour
  finding, and boundary generation all require great-circle math and
  icosahedron face bookkeeping. That's necessary complexity for global
  coverage, but pure overhead for a Great-Britain-only tool, and it
  rules out anything as simple as the closed-form axial ↔ Cartesian
  formulas this library uses.
- **No native EPSG:27700 support.** Every coordinate has to round-trip
  through H3's own spherical indexing scheme even though the source and
  target data almost always start and end in British National Grid
  metres. This library stays in EPSG:27700 the entire time — projection
  happens once, at the boundary, not as part of every lookup.

None of this is a knock on H3 — it's the right tool when you actually
need global coverage. It's the wrong tool when every input and output
is already in EPSG:27700 and the analysis never leaves Great Britain.

## Install

```bash
uv sync
```

Dependencies are `numpy`, `pyproj`, and `shapely`. `polyfill` (see below)
is the only thing that does a point-in-polygon query; `geometry.py`'s
`cell_polygon`/`cell_polygons` also use Shapely, but only to return cell
outlines as `Polygon` objects, not to run any spatial predicate.

There is one extra, `beahiv[arrow]`, which pins `pyarrow` for the
Arrow-in/Arrow-out encoding path (see [pyarrow](#pyarrow) below). It is
not needed for that path to work.

### Agent skill

beahiv ships an [agent skill](src/beahiv/skill/SKILL.md) that teaches
a coding agent how to use the library. It covers the rules that are easy to
get wrong (axis order, CRS, mixing grids, `INVALID_CELL_ID`), with recipes
and a guide to common errors. Install it from a project that depends on
beahiv, so it matches the installed version:

```bash
uv run beahiv-skill install .claude     # -> .claude/skills/beahiv/
uv run beahiv-skill install ~/.claude   # user-wide
uv run beahiv-skill remove .agents      # removes .agents/skills/beahiv/
```

The argument is the agent's config directory, and the skill goes in its
`skills/beahiv/` subdirectory. Rerunning `install` after an upgrade updates
the skill. `remove` deletes only the files the skill ships. If one of those
files has been edited locally, both commands refuse to change anything
unless you pass `--force`.

## Quickstart

```python
import beahiv
from beahiv import Orientation

# Trafalgar Square, London, indexed on a 500m flat-top hex grid.
cell_id = beahiv.lonlat_to_cell(-0.1278, 51.5074, size=500)

beahiv.decode(cell_id)
# CellIndex(q=706, r=-145, size=500, measure=<Measure.EDGE_LENGTH: 0>, orientation=<Orientation.FLAT: 1>)

# ...or size cells by the distance between parallel edges instead (see "Size and measure")
beahiv.lonlat_to_cell(-0.1278, 51.5074, size=500, measure=beahiv.Measure.EDGE_TO_EDGE)

beahiv.centroid(cell_id)  # POINT (530005.2 180366.6) in EPSG:27700 metres
beahiv.centroid(cell_id, lonlat=True)  # back to WGS84, as POINT (lon lat)
beahiv.cell_polygon(cell_id)  # 6 vertices, generated on demand
beahiv.get_neighbours(cell_id)  # 6 neighbouring cell ids
beahiv.k_ring(cell_id, 2)  # all 19 cells within 2 hops
beahiv.distance(cell_id, other)  # hex grid distance between two cells
```

See also: [examples notebook](https://github.com/safer-streets/exploratory-data-analysis/blob/main/introducing-beahiv.ipynb)

## Core concepts

### Axis order

WGS84 coordinates are always **(lon, lat)** — x then y — everywhere in
beahiv: function arguments (`lonlat_to_cell(lon, lat, ...)`), returned
tuples (`batch.cell_to_lonlat_batch`), and Shapely points (`Point(lon, lat)`).
This is the shapely/GeoJSON convention and pyproj's `always_xy` order, and
the same x-then-y order as EPSG:27700's `(x, y)`.

`latlon_to_cell`, `batch.latlon_to_cell_batch` and `batch.cell_to_latlon_batch`
are the old lat-first spellings. They still work but are marked
`@deprecated`: they emit a `DeprecationWarning`, and type checkers flag
them. `centroid`/`centroids` take `lonlat=`; their old `latlon=` keyword
has been removed and now raises `TypeError`. Swapping the
order is caught rather than silent: GB's latitude and longitude ranges
don't overlap, so lat/lon passed the wrong way round fails the
area-of-use check.

### Cell size

`size` must be a whole number of metres from **3 m to 65,472 m**, not
an index into a fixed resolution table. Unlike H3's 16 discrete resolutions,
there are 4,093 valid sizes: **every whole metre from 3 to 1,023**, then
sizes with 10 significant *binary* digits — steps of 2 m up to 2,046 m, 4 m up
to 4,092 m, and so on, doubling each time, to 64 m steps at the top. So 25 m,
202 m, 350 m, 1 km, 1.5 km, 5 km and 10 km are all valid grids, in either
orientation. 1,025 m and 14.5 km are not; the error names the nearest valid
sizes (14,496 m and 14,512 m). `encode`/`encode_batch` raise
`ValueError` for anything else.

Binary rather than round-decimal steps because then **2x and 0.5x of a valid
size are always valid too** (while in range), so the parent/child lookups
below always have a grid to go to. A decimal three-significant-digit size was
tried first and broke that for 1,200 sizes (501 m → 1,002 m).

Both limits come from the bit budget (see "Cell identifiers" below). The
binary format is what lets the size fit 12 bits. The 3 m floor is the
smallest cell whose q/r still cover the whole National Grid, so a smaller one
is rejected outright rather than encoding near the origin and raising further
out. For `EDGE_TO_EDGE` the floor is 6 m, a geometric side of at least 3 m.

Two things this does *not* enforce:

- **Whole metres only.** `size` must be an `int` -- a float such
  as `500.5` isn't rejected with a helpful message, it fails at the bit
  shift (`TypeError`, since Python won't `<<` a float) inside `encode`.
  Round or truncate before calling.
- **No cross-resolution nesting for arbitrary ratios.** Cells at different
  `side_length` values are independent lattices scaled by `s`, not levels
  of a hierarchy, and this isn't fixed by choosing a "clean" ratio: a
  regular hexagon can't be exactly tiled by smaller regular hexagons of
  the same orientation for an *arbitrary* integer factor, the way a
  square cleanly quarters for a quadtree. Picking `size=300` as
  "3x" a `size=100` grid doesn't produce 9 small cells per big
  cell either -- most straddle the big cell's boundary instead of
  nesting inside it. Mixing side lengths in one dataset is fine as long
  as each cell id carries its own (which it does, by construction), but
  don't expect `k_ring`/aggregation across arbitrary different side
  lengths.
- **Same-centroid 2x/0.5x lookups are the one exception** -- see
  "Parent/child at 2x/0.5x `side_length`" below.

### Size and measure

Every function that takes a cell size takes it as two **keyword-only**
arguments: `size=` (required) and `measure=`, which says what that number
measures and defaults to `Measure.EDGE_LENGTH`. Leave out `size`, or pass a
`measure` that isn't a `Measure`, and it raises `TypeError`. There is
deliberately no positional form (`orientation` and `predicate` are
keyword-only too):

- `Measure.EDGE_LENGTH` (the default) — `size` is the length of one edge.
- `Measure.EDGE_TO_EDGE` — `size` is the distance between two opposite,
  parallel edges, which is `sqrt(3)` edge lengths.

```python
beahiv.bng_to_cell(530034, 180381, size=200)  # 200m edges, the same as measure=beahiv.Measure.EDGE_LENGTH
beahiv.bng_to_cell(530034, 180381, size=200, measure=beahiv.Measure.EDGE_TO_EDGE)  # 200m across, edge ~115.5m
```

`measure` is a plain value, so it can be passed through a function or a
config like any other argument. A plain `0`/`1` is refused rather than
converted: `Measure` and `Orientation` are both 0/1 enums, so converting would
let `measure=Orientation.FLAT` through as `EDGE_TO_EDGE` without a word.

Either way the stored size follows the same rules (whole metres, 3 m to
65,472 m, 10 significant bits above 1,023 m), and which one it measures is recorded in the cell id
itself (the measure bit, below) — so everything downstream of an id
(`decode`, geometry, neighbours, hierarchy) needs no size arguments at all,
and an edge-to-edge cell's relatives are edge-to-edge cells. `CellIndex`
carries both halves: `size` is the stored value and `measure` the
`Measure` it is in, while `CellIndex.side_length` is the *geometric*
side length they imply (`size` itself for an `EDGE_LENGTH` cell).
The two measures are different lattices even at the same number, like the
two orientations: never compare q/r across them. `resize_cell` is the one
exception to the default: there `measure` defaults to the source cell's own
(as `orientation` does), and either may be overridden.

### Parent/child at 2x/0.5x `side_length`

`axial_to_cartesian` is linear in `(q, r)`, so a cell at `side_length`
shares its *exact* centroid with a cell at `2 * side_length` if and only
if `(q / 2, r / 2)` is itself an integer pair (`q` and `r` both even),
and with a cell at `side_length / 2` if and only if `side_length` is
itself even (that child is always `(2q, 2r)` -- doubling is always an
integer, unlike halving). Neither direction needs rounding or a
Cartesian round trip: the same-centroid partner either exists as an
exact integer solution, or it doesn't exist at all.

Singular and plural answer two different questions, and the name is the
whole distinction. **`get_parent`/`get_child`** return that same-centroid
partner, or nothing -- pure parity arithmetic, no geometry, a cheap way
to ask whether a cell happens to line up exactly with the 2x/0.5x grid:

```python
cell_id = beahiv.encode(4, -6, size=100)
parent = beahiv.get_parent(cell_id)  # side_length=200, same centroid
child = beahiv.get_child(cell_id)  # side_length=50, same centroid
assert beahiv.get_child(parent) == cell_id

odd_q = beahiv.encode(3, -6, size=100)
beahiv.get_parent(odd_q)  # None -- odd q, no same-centroid parent
```

**`get_parents`/`get_children`** return every cell at the target
`side_length` that *overlaps* this one, as a tuple. That set always
exists and always covers the cell, and is what you want when resizing by
a factor of two:

```python
beahiv.get_parents(beahiv.encode(3, -6, size=100))  # the 2 straddled 200m cells
beahiv.get_parents(cell_id)  # 1 -- an even-q/r cell lies wholly inside its same-centroid parent
beahiv.get_children(cell_id)  # 7 x 50m: the same-centroid child, plus its k_ring of 1
```

`get_parents` returns 1 id when `q` and `r` are both even (a cell sharing
a 2x cell's centroid lies wholly inside it, with nothing to straddle) and
2 otherwise; it is never empty. `get_children` always returns 7 -- the
same-centroid child and the six partially contained cells ringing it,
which cover the cell but **overspill** it by 75% of its area: half-size
hexes cover a hexagon, they don't partition it.

So neither pair is a hierarchical tiling. The same-centroid relation is
partial (most cells have no parent), and the overlapping relation shares
cells rather than partitioning them. For an actual covering at an
arbitrary new size, see `resize_cell` below.

An odd `side_length` raises `ValueError` from both child functions,
rather than returning nothing the way `get_parent` does on odd `q`/`r`:
side lengths are integer metres, so an odd one means the 0.5x grid
doesn't exist at all and there's nothing to be inside -- a different
failure from a cell that merely doesn't line up with a grid that does
exist. `get_parent`/`get_parents` likewise raise when doubling
`side_length` would exceed `SIDE_LENGTH_MAX`.

### Coordinate system

All geometry is computed in EPSG:27700 projected metres. Latitude/longitude
is accepted as input only, and immediately projected via `pyproj` — there
is no spherical geometry anywhere in the indexing path.

### Axial coordinates and orientation

Every hex is addressed by signed integer axial coordinates `(q, r)`, plus
a `side_length` (metres) and an `orientation`:

- **POINTY** — a vertex points up/down, the left/right edges are vertical.
- **FLAT** — POINTY rotated 30°: a vertex points left/right, top/bottom
  edges are horizontal.

```python
x = x0 + side_length * sqrt(3) * (q + r / 2)  # POINTY
y = y0 + 1.5 * side_length * r

x = x0 + 1.5 * side_length * q  # FLAT
y = y0 + side_length * sqrt(3) * (r + q / 2)
```

`side_length` here is the geometric one (`size / sqrt(3)` for an
`EDGE_TO_EDGE` cell). `(x0, y0)` is the grid origin — the EPSG:27700 point
axial `(0, 0)` centres on, shared by every size, orientation and measure.
It is `ORIGIN_X`/`ORIGIN_Y` in `coords.py`, the single place it is defined
for both the scalar and batch paths. It is *not* stored in
cell ids, so changing it re-maps every existing id to a different place:
ids are only comparable between builds that agree on it.

Cartesian → axial is the inverse of these formulas followed by standard
cube-coordinate rounding — this is what guarantees an exact, deterministic
cell for every point, with no ambiguity at cell boundaries.

### Cell identifiers

Every cell is a single integer, reversible with no lookup table, and
always **below `2**52`**:

```text
bits 63-53  reserved     11 bits   always zero, permanently (see "Why 52 bits")
bit 52      reserved      1 bit    always zero for now; kept for future use
bit 51      measure       1 bit    0 = EDGE_LENGTH, 1 = EDGE_TO_EDGE
bit 50      orientation   1 bit    0 = POINTY, 1 = FLAT
bits 49-38  size         12 bits   3-bit binary exponent, 9-bit mantissa
bits 37-19  q (offset)   19 bits   q + Q_OFFSET
bits 18-0   r (offset)   19 bits   r + R_OFFSET
```

**Why 52 bits.** float64 holds every integer below `2**53` exactly, so an id
that passes through a float comes back unchanged. That happens all the time
without anyone asking: pandas turns an `int64` column into `float64` the
moment it gains a `NaN`, and JavaScript's `JSON.parse` reads every number as
a double. Under the earlier 64-bit layout such an id came back rounded but
still well-formed, so it decoded to a valid cell up to hundreds of
kilometres away, with no error. Now there is nothing to detect: the round
trip is exact. The ids still go in a plain `int64`/`BIGINT` column; every
array of ids beahiv returns (numpy or Arrow) is `int64`, never `uint64`.

**Reserved bits.** Bit 52, the 53rd bit, is reserved for future use. It is
the only one that may ever be given a meaning; setting any bit
above it would push ids past `2**53` and bring the problem back. No `encode`
path sets it, and `decode` *rejects* an id with any reserved bit set rather
than ignoring it — that strictness is what keeps the bit free to be given a
meaning later, since a decoder that quietly masked unknown bits off could
never start honouring one.

**Size.** A tiny binary float, as in IEEE 754. Exponent 0 stores the
mantissa as-is (0–511 m). Exponent `e` ≥ 1 stores `(512 + mantissa) << (e - 1)`,
a 10-bit significand with an implicit leading bit. Each size has exactly one
encoding and every field value is a size, so the only invalid size fields are
those below the 3 m minimum.

**q/r.** 19 bits each, stored with an offset (`Q_OFFSET`, `R_OFFSET`) so both
fields decode with plain arithmetic and no sign handling. The offsets aren't
centred on zero: they put the National Grid box (700 km × 1300 km) near the
middle of both fields at the 3 m minimum, in both orientations, because GB lies
entirely north-east of the grid origin. That leaves over 300 km of room beyond
every edge of the box, even at 3 m.

**Ids from earlier layouts** (0.0.x, or an untagged pre-renumbering 1.0.1
build) are not compatible. `decode` rejects every old FLAT id, every old side-to-side (now `EDGE_TO_EDGE`) id and
every old POINTY id with a side of 512 m or more, as "reserved bits set". Old
POINTY side-length ids with smaller sides fall inside the new range and
decode to a different cell. Regenerate stored ids from their coordinates
rather than mixing them.

### Neighbours, distance, rings

Neighbours are pure axial arithmetic — no geometry is constructed:

```python
NEIGHBOUR_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
```

Distance converts to cube coordinates and applies the standard hex grid
distance formula. `k_ring(cell_id, k)` returns all cells within `k` hops,
satisfying `N(k) = 1 + 3k(k+1)`.

### Geometry

BEAHIV uses `shapely` geometry types which also supplies conversion functions
to convert to/from WKT or WKB format. The latter is essential when implementing UDFs
for duckdb.


Polygons are never stored — `cell_polygon` regenerates six vertices on
demand from the cell centre, at 30/90/150/210/270/330° (POINTY) or
0/60/120/180/240/300° (FLAT).

`cell_polygons` is the vectorised sibling for many cells at once, built on
`batch.cell_centre_batch` for the centre lookup, and taking the same
array-like input it does — a list, a numpy array, a pandas Series, a pyarrow
array:

```python
beahiv.cell_polygons(cell_ids)  # one Polygon per cell id, same order
```

Every cell passed in must share the same `side_length` and `orientation` —
same restriction as `cell_centre_batch`, since a single angle set and radius
only apply to one grid at a time. An empty input returns `[]`. A missing
(null) or invalid id, `INVALID_CELL_ID` included, gives `None` in its place
rather than raising, so the result still lines up row for row: geopandas
reads `None` as a missing geometry, and `shapely.to_wkb` turns it back into
a null.

`centroid`/`centroids` mirror that pair for cell centres, returning Shapely
`Point`s rather than coordinate tuples:

```python
beahiv.centroid(cell_id)  # POINT, EPSG:27700
beahiv.centroids(cell_ids)  # one Point per cell id, same order
```

A `Point` carries no CRS, so with `lonlat=True` the result follows the
shapely/GeoJSON axis convention — `Point(lon, lat)`, **x is longitude** —
and drops straight into a GeoSeries with `crs=4326`. For plain coordinate
columns, `batch.cell_centre_batch` and `batch.cell_to_lonlat_batch` are the
direct route and are what `centroids` wraps.

## Bulk operations

`lonlat_to_cell` and `bng_to_cell` accept arrays transparently —
pass a list, a numpy array, or a pandas Series and get one back
(`centroid` is scalar-only; its array form is `centroids`):

```python
cell_ids = beahiv.lonlat_to_cell(lons, lats, size=500)  # lons/lats: array-like
```

A plain scalar still takes the pure-Python path with no numpy call, at
about 1µs a call from EPSG:27700 and 2µs from WGS84; anything else dispatches to the numpy-vectorised
implementation in `beahiv.batch`, which also remains available directly
for callers who want an unambiguous vectorised call:

```python
from beahiv.batch import lonlat_to_cell_batch, bng_to_cell_batch, cell_to_lonlat_batch

cell_ids = lonlat_to_cell_batch(lons, lats, size=500)
```

A missing coordinate (`NaN`) is an absent point rather than an error: it
encodes to `INVALID_CELL_ID` (`0`), which no valid `encode` can produce.

The batch functions that take cell ids tolerate gaps the same way. A null
(`None`, `NaN`, pandas `<NA>`, an Arrow null), `INVALID_CELL_ID`, or any
other id scalar `decode` would reject becomes a missing value in the output
rather than an error:

| Function | Missing/invalid row gives |
| --- | --- |
| `batch.decode_batch` | all-zero `(q, r, size, measure, orientation)` — `size == 0` is the mask |
| `batch.cell_centre_batch`, `batch.cell_to_lonlat_batch` | `NaN` |
| `cell_polygons`, `centroids` | `None` |
| `get_parents`, `get_children` (array input) | nothing — the id adds no cells to the union |

Only the scalar functions (`decode`, `cell_polygon`, `centroid`, ...) still
raise for a bad id. Mixing grids still raises in the batch functions too:
that is a mistake in the call, not a gap in the data.

A float column of ids is fine. Every id is below `2**52`, so one that has
been through a float64 (an `int64` column that gained a `NaN`, a JSON number)
is still exact, and decodes to the right cell. `NaN` is a missing row. A float
that can't be an id (non-integral, infinite, or `2**53` or above) is treated as
invalid like any other bad row.

The scalar functions take numpy integers as well as `int`, so a single id
picked out of a batch result (or read from a pandas column, or a DuckDB
`BIGINT`) can be passed straight back in:

```python
beahiv.k_ring(cell_ids[0], 1)  # cell_ids[0] is an np.int64, not an int
```

They coerce it on the way in and always hand back plain Python `int` ids —
nothing downstream inherits numpy's fixed-width overflow behaviour.

The overlapping hierarchy lookups also take many ids at once -- a list,
numpy array, or pandas Series -- and return the union: a flat, sorted
`int64` array of every overlapping cell, each once, since neighbouring cells
share most of their parents and children. The same-centroid
`get_parent`/`get_child` are scalar-only.

```python
parents = beahiv.get_parents(cell_ids)  # 1-D int64, duplicates dropped
children = beahiv.get_children(cell_ids)  # likewise
```

### pyarrow

`lonlat_to_cell` and `bng_to_cell` also take a pyarrow `Array` or
`ChunkedArray`, and give an `int64` `Array` back — Arrow in, Arrow out.
Nulls arrive as `NaN` and so encode to `INVALID_CELL_ID`:

```python
import pyarrow as pa

cell_ids = beahiv.bng_to_cell(pa.array(xs), pa.array(ys), size=202)
```

This is what makes beahiv usable as a vectorised UDF in an Arrow-native
query engine (DuckDB, Polars), where a per-row Python call would dominate:

```python
con.create_function(
    "beahiv_cell",
    lambda x, y: beahiv.bng_to_cell(x, y, size=202),
    [DOUBLE, DOUBLE],
    BIGINT,
    type="arrow",
)
```

Note `side_length` is closed over rather than declared as a UDF parameter.
Under `type="arrow"` *every* declared parameter arrives as a `ChunkedArray`,
including one that is a constant at every call site, and `bng_to_cell` wants
a scalar `side_length` — passing the array raises `TypeError: '<=' not
supported between instances of 'int' and 'pyarrow.lib.ChunkedArray'` from the
range check. Take x/y as parameters and bind the grid into the function.

pyarrow is **not** a runtime dependency: the import happens lazily, on a
branch only reachable when the caller has already handed us a pyarrow
object. The `beahiv[arrow]` extra exists to pin a version and advertise
the capability, not to make the feature work.

See [introducing-beahiv.ipynb](https://github.com/safer-streets/exploratory-data-analysis/blob/main/introducing-beahiv.ipynb)
for more examples of duckdb UDFs

### Shapely and geopandas

Point data usually arrives as a single geometry column rather than as
separate x and y sequences. `point_to_cell` takes that column directly:

```python
gdf["cell_id"] = beahiv.point_to_cell(gdf, size=202)
```

It accepts a `GeoDataFrame` (its active geometry column is used), a
`GeoSeries`, a `GeometryArray`, an object ndarray of Shapely points, or a
plain list, and returns an `int64` numpy array — never a `Series`, so
assigning it back is positional and can't realign against a non-default
index. A single `Point` returns a plain `int`, the same scalar/array
dispatch the rest of the API uses.

Going through the column is also the faster route, because it skips
building the intermediate `Series` that `.x`/`.y` produce — on 500k
points:

| | |
| --- | --- |
| `point_to_cell(gdf, side_length=202)` | 50 ms |
| `bng_to_cell(gdf.geometry.x, gdf.geometry.y, side_length=202)` | 89 ms |

Missing (`None`) and empty points encode to `INVALID_CELL_ID`, as NaN
coordinates do elsewhere. A non-point geometry raises rather than
encoding to all-invalid.

#### EPSG:27700 or WGS84

`point_to_cell` reads coordinates as either British National Grid metres
or WGS84 `Point(lon, lat)` — the x/y order `centroid(lonlat=True)` returns.
A `GeoDataFrame` or `GeoSeries` says which through its `.crs`:

```python
beahiv.point_to_cell(gdf, size=202)  # gdf.crs is EPSG:27700 or EPSG:4326
```

Any other declared CRS raises rather than being silently misread — WGS84
degrees interpreted as metres, say, put every point within a few metres of
the grid origin, encoding to a perfectly valid and completely wrong cell.
Reproject first with `gdf.to_crs(27700)`.

Bare Shapely geometry is different: a `Point` holds two numbers and
nothing else. It has no `.crs`, and while GEOS keeps an SRID slot
(`shapely.get_srid`) it defaults to `0` and geopandas never populates it —
so there is nothing to infer a CRS *from*. A `Point`, list or ndarray (or a
`GeoSeries` with an unset `.crs`) is read as EPSG:27700 unless you pass
`lonlat=True`:

```python
beahiv.point_to_cell(Point(-0.1246, 51.5007), size=202, lonlat=True)
```

Passing a `lonlat` that contradicts a declared CRS raises. WGS84 input
goes through `lonlat_to_cell`, so it gets the same EPSG:27700 area-of-use
check — which also catches `Point(lat, lon)` written the wrong way round.

geopandas is **not** a runtime dependency — `point_to_cell` duck-types on
`.geometry` and `.crs`, and only ever calls Shapely, which beahiv already
requires.

### Filling a polygon with cells

`beahiv.polyfill` covers a Shapely polygon with hex cells — the one bulk
operation that does need a point-in-polygon query, so it's kept out of
the core indexing modules (`geo.py`, `coords.py`, ...), which never do
spatial joins:

```python
# polygon must be in EPSG:27700
cell_ids = beahiv.polyfill(polygon, size=500, orientation=Orientation.FLAT)
```

`predicate` (default `"overlap"`) controls what counts as covering,
matching h3's `contain` vocabulary: `"overlap"` (any part of the hex
touches the polygon), `"center"` (hex centre inside the polygon), or
`"full"` (hex entirely inside the polygon).

The predicates run vectorised over chunks of candidate cells, so a fill
costs roughly 0.1–0.6µs per returned cell for `"overlap"`/`"centre"` (a
100×100km box at 25m — 6 million cells — takes about half a second) and
about 1µs for `"full"`, which has to test each cell's whole hexagon.

For the common case of an axis-aligned bounding box rather than an
arbitrary polygon, `beahiv.bbox_fill` is a thin convenience wrapper —
same `predicate` semantics, no need to build a Shapely `Polygon` first:

```python
cell_ids = beahiv.bbox_fill(minx, miny, maxx, maxy, size=500, orientation=Orientation.FLAT)
```

For resizing a cell you already have -- covering its hexagon with cells
at a different size -- `beahiv.resize_cell` is seeded by the
cell's own outline (`cell_polygon`) instead of an arbitrary polygon:

```python
finer = beahiv.resize_cell(cell_id, size=50)  # a finer covering
coarser = beahiv.resize_cell(cell_id, size=500)  # a coarser covering -- also permitted
```

`size` may be smaller or larger than `cell_id`'s own size -- both directions are ordinary `polyfill` calls against
the cell's polygon, same `predicate` semantics. This is the general,
geometry-based replacement for `get_parents`/`get_children` (previous
section): unlike those it isn't restricted to a factor of two, doesn't
require an even `q`/`r`/`side_length`, and always returns *something*
covering the cell.

`measure` and `orientation` default to `cell_id`'s own, but either can be
overridden to cover the cell with a grid of another family instead:

```python
beahiv.resize_cell(cell_id, size=50, measure=beahiv.Measure.EDGE_TO_EDGE, orientation=Orientation.FLAT)
```

### DuckDB spatial

Use WKB format to exchange geometry data between python and duckdb. Some examples of vectorised UDFs are below.

```py
import duckdb
from duckdb.sqltypes import BIGINT, BLOB, DOUBLE, VARCHAR
import shapely


GEOMETRY = duckdb.type("GEOMETRY")

con = duckdb.connect(...)
con.execute("INSTALL spatial;LOAD spatial;")

con.create_function(
    "bh_polyfill",
    lambda wkb, measure, size, orientation, predicate: bh.polyfill(
        shapely.from_wkb(wkb),
        size=size,
        measure=bh.Measure[measure],
        orientation=bh.Orientation(orientation),
        predicate=predicate,
    ),
    [GEOMETRY, VARCHAR, BIGINT, BIGINT, VARCHAR],
    duckdb.list_type(BIGINT),
)
con.create_function(
    "bh_cell_polygon",
    lambda cell_ids: shapely.to_wkb(bh.cell_polygons(cell_ids)),
    [BIGINT],
    BLOB,
    type="arrow",
)
con.create_function(
    "bh_point_to_cell",
    lambda x, y: bh.bng_to_cell(x, y, size=size, measure=bh.Measure.EDGE_TO_EDGE, orientation=bh.Orientation.POINTY),
    [DOUBLE, DOUBLE],
    BIGINT,
    type="arrow",
)
```

`bh_cell_polygon` takes the whole chunk at once (`cell_polygons`, not
`cell_polygon`) — about 2x faster than the per-row form on a 15k-cell
polyfill, and the reason to spend the `type="arrow"` declaration. It
inherits `cell_polygons`' restriction that every id in a call shares one
size, `orientation` and measure, which holds for the output of a single
`polyfill` but would raise on a column mixing grids; use the per-row
`cell_polygon` if a query can do that. A `NULL` (or `0`) id gives a `NULL`
geometry.

`bh_polyfill` stays per-row: it consumes one boundary polygon per call and
returns a list, so there is nothing to vectorise. `bh_point_to_cell` binds
the size/`orientation` rather than declaring them as parameters, for the
`ChunkedArray` reason given above.

SQL has no enums, so `bh_polyfill` takes the measure as the member's name
(`'EDGE_LENGTH'` or `'EDGE_TO_EDGE'`) and looks it up with `bh.Measure[measure]`:
one UDF covers both measures, the SQL call site stays explicit
(`bh_polyfill(geom, 'EDGE_TO_EDGE', 350, ...)`), and a misspelt measure fails
with a `KeyError` rather than being read as an edge length. The closed-over
`bh_point_to_cell` just names its measure. `bh_cell_polygon` needs nothing: the measure travels in the id.

And to use duckdb for spatial computations and query the results into a geopandas `GeoDataFrame`, adapt the pattern in
the example below, which generates BEAHIV cells within a boundary polygon:

```py
import geopandas as gpd

beahiv_cells = gpd.GeoDataFrame.from_arrow(
    con.sql(
        """
        WITH c AS (
            SELECT unnest(bh_polyfill(geom, 'EDGE_TO_EDGE', ?, ?, 'centre')) AS spatial_id
            FROM boundary_table
            WHERE name = ?
        )
        SELECT spatial_id, ST_GeomFromWKB(bh_cell_polygon(spatial_id)) AS geometry FROM c
        """,
        params=(size, bh.Orientation.POINTY, boundary_name),
    ).arrow()
).set_crs("epsg:27700")
```

Every value is a `?` placeholder — note the query string is deliberately not
an f-string, so `boundary_name` reaches DuckDB as a bound parameter rather
than as interpolated SQL.

`ST_GeomFromWKB` is doing real work and can't be dropped by declaring the
UDF's return type as `GEOMETRY`: DuckDB rejects that with `Conversion Error:
Unimplemented type for cast (BLOB -> GEOMETRY)`. WKB out of Python, parsed on
the SQL side, is the route.

The CRS is set on arrival rather than carried through: beahiv is EPSG:27700
throughout and WKB has nowhere to put an SRID, so nothing upstream of
`set_crs` knows the units are British National Grid metres.

## Testing

```bash
uv run pytest
```

Property tests cover:

- **Round trip** — `decode(encode(q, r, s, o)) == (q, r, s, o)` for random
  and boundary coordinates, including the corners of the National Grid box
  at the 3 m minimum, in both orientations.
- **Float exactness** — every id survives `int(float(id))` unchanged,
  including the largest the fields allow.
- **Neighbour symmetry** — if B is a neighbour of A, A is a neighbour of B.
- **Area equality** — random cells all have identical area, matching
  `A = (3√3/2)s²`, within 1e-9 relative tolerance.
- **Coordinate stability** — a random point, converted to a cell and back
  to a polygon, always lies within that polygon.

## API reference

| Function | Description |
| --- | --- |
| `encode(q, r, *, size, measure=EDGE_LENGTH, orientation=FLAT)` | Build a cell id. Every size-taking function below takes `size` and `measure` the same way |
| `decode(cell_id)` | Recover `CellIndex(q, r, size, measure, orientation)`; raises for any id `encode` couldn't have produced |
| `Measure` | `EDGE_LENGTH` / `EDGE_TO_EDGE` — what a cell's `size` measures |
| `Orientation` | `POINTY` / `FLAT` (the default) — two different lattices, not a relabelling; never mix their ids |
| `INVALID_CELL_ID` | The all-zero sentinel emitted for missing input — scalar `decode` raises for it; the batch functions give a missing value |
| `lonlat_to_cell(lon, lat, *, size, measure, orientation)` | WGS84 → cell id (scalar, array-like, or pyarrow). Raises outside EPSG:27700's area of use (lon −9.01–2.01, lat 49.75–61.01) |
| `latlon_to_cell(lat, lon, *, size, measure, orientation)` | Deprecated lat-first spelling of `lonlat_to_cell` |
| `bng_to_cell(x, y, *, size, measure, orientation)` | EPSG:27700 → cell id, no WGS84 round trip (scalar, array-like, or pyarrow). No bounds check: any (x, y) within the bit budget encodes |
| `point_to_cell(points, *, size, measure, orientation, lonlat=None)` | Shapely point(s) — a `Point`, or a geopandas `GeoDataFrame`/`GeoSeries` — → cell id(s). EPSG:27700, or WGS84 `Point(lon, lat)` via a declared EPSG:4326 or `lonlat=True` |
| `centroid(cell_id, *, lonlat=False)` | Cell centre as a Shapely `Point` — EPSG:27700 (default), or WGS84 as `Point(lon, lat)` (`lonlat=True`) |
| `centroids(cell_ids, *, lonlat=False)` | Vectorised `centroid` — one same-grid cell id list in, one `Point` per cell out (`None` for a missing/invalid id) |
| `cell_polygon(cell_id)` | Cell outline as a Shapely `Polygon`, in EPSG:27700 |
| `cell_polygons(cell_ids)` | Vectorised `cell_polygon` — one same-grid cell id list in, one `Polygon` per cell out (`None` for a missing/invalid id) |
| `get_neighbours(cell_id)` | Six neighbouring cell ids |
| `NEIGHBOUR_OFFSETS` | The six axial `(dq, dr)` steps `get_neighbours` takes, in its order |
| `distance(cell_a, cell_b)` | Hex grid distance |
| `k_ring(cell_id, k)` | All cells within `k` hops |
| `get_parent(cell_id)` | Cell at 2x `side_length` sharing this cell's exact centroid, or `None` if `q`/`r` aren't both even; single id only |
| `get_child(cell_id)` | Cell at `side_length / 2` sharing this cell's exact centroid — always `(2q, 2r)`; single id only |
| `get_parents(cell_id)` | Every cell at 2x `side_length` overlapping this one — 1 if it nests exactly, else the 2 it straddles; array input gives the deduplicated union, skipping missing/invalid ids |
| `get_children(cell_id)` | Every cell at `side_length / 2` overlapping this one — always 7, covering it with 75% overspill; array input gives the deduplicated union, skipping missing/invalid ids |
| `polyfill(polygon, *, size, measure, orientation, predicate)` | Every cell id covering a Shapely polygon |
| `bbox_fill(minx, miny, maxx, maxy, *, size, measure, orientation, predicate)` | Every cell id covering an axis-aligned bounding box |
| `resize_cell(cell_id, *, size, measure=None, orientation=None, predicate)` | Every `size` cell id covering `cell_id`'s hexagon -- smaller or larger than its own; `measure`/`orientation` default to the cell's |

## Technical details

### WGS84 → British National Grid needs the OSTN15 grid file

`lonlat_to_cell`, `point_to_cell` with WGS84 input, and `centroid(..., lonlat=True)` go through
pyproj, and the cells they produce are only as reproducible as that transformation. The accurate
WGS84 ↔ OSGB36 transformation is Ordnance Survey's **OSTN15**, a correction grid published as
`uk_os_OSTN15_NTv2_OSGBtoETRS.tif` (about 3MB). The pyproj wheels **don't include it**, and when it
is missing, PROJ **silently** uses a 7-parameter Helmert transformation instead. This raises no
error and no warning, and points shift by roughly 0.5m (Edinburgh) to 3–4m (Cornwall and the
Western Isles).

At small cell sizes that changes which cell a point lands in. The share of random points whose cell
changes between the two transformations:

| Area | 25m cells | 100m cells | 250m cells |
| --- | --- | --- | --- |
| London | 4.9% | 1.3% | 0.5% |
| Cornwall | 10.8% | 2.7% | 1.1% |

So the same `(lon, lat)` can give a different cell id on a machine that has the grid than on one
that doesn't, such as CI, a fresh container or a colleague's laptop. `bng_to_cell` and everything
else that starts from EPSG:27700 coordinates is unaffected: there is no datum transformation there.

**Installing it.** Either of these works with the pyproj wheel:

```sh
# download once, into PROJ's user data directory (pyproj.datadir.get_user_data_dir())
uv run pyproj sync --file uk_os_OSTN15_NTv2_OSGBtoETRS.tif
```

```sh
# or let PROJ fetch the parts of the grid it needs from cdn.proj.org on demand,
# caching them in the same directory
export PROJ_NETWORK=ON
```

In a container or CI image, run the `pyproj sync` line at build time (`--target-directory DIR`
puts it elsewhere; point `PROJ_USER_WRITABLE_DIRECTORY` at that directory at runtime). Note that the
pyproj wheel uses its own bundled data directory and the user directory, so a copy installed for a
system PROJ (e.g. under `/usr/share/proj`) isn't seen.

**Checking it.** This prints `True` only when the best transformation, OSTN15, is usable:

```python
from pyproj.transformer import TransformerGroup

print(TransformerGroup("EPSG:4326", "EPSG:27700").best_available)
```

### "Equal-area" holds in the projected plane

Every cell has exactly the same area, `(3√3/2)s²`, **measured in British National Grid
metres**. BNG is a Transverse Mercator projection, which preserves shape locally but not area: its
scale factor is 0.9996 on the central meridian (2°W) and grows with distance east or west. So the
same cell covers slightly different areas *on the ground* in different places:

| Location | BNG area ÷ ground area |
| --- | --- |
| Central meridian (2°W) | 0.99920 |
| London | 0.99963 |
| Lowestoft (furthest east) | 1.00079 |
| Outer Hebrides (furthest west) | 1.00167 |

The spread across Great Britain is about 0.25%. A cell on the central meridian covers about 0.25%
more ground than one in the Outer Hebrides. For counts, densities and rates this is far below any
other source of error, but a cell is equal-area in the map, not exactly on the ellipsoid.
