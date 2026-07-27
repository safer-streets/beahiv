# BEAHIV

**B**ritish National Grid **E**qual-**A**rea **H**exagonal **I**ndex that's **V**ersatile.

A fast, equal-area hexagonal spatial index native to **EPSG:27700** (British
National Grid). BEAHIV is a local alternative to H3: it trades global
coverage for exact, deterministic geometry, and is built for regional
analysis within Great Britain — crime analysis, spatial statistics,
regional-scale modelling.

Each letter maps onto something the index actually does:

- **British** — native to EPSG:27700, no spherical geometry anywhere.
- **Equal-Area** — every cell has identical area, always.
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

- Equal-area cells — every hex has identical area, always
- Single neighbour relationship, six neighbours per cell
- Deterministic coordinate → cell mapping, no lookup tables
- No spatial joins, no point-in-polygon queries, no R-tree
- O(1) coordinate lookup, O(1) neighbour lookup, O(1) distance
- Compact 64-bit cell identifiers
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

Dependencies are `numpy`, `pyproj`, and `shapely`. Only `polyfill` (see
below) touches Shapely -- the rest of the indexing code never does any
point-in-polygon query.

There is one extra, `beahiv[arrow]`, which pins `pyarrow` for the
Arrow-in/Arrow-out encoding path (see [pyarrow](#pyarrow) below). It is
not needed for that path to work.

## Quickstart

```python
import beahiv
from beahiv import Orientation

# Trafalgar Square, London, indexed on a 500m flat-top hex grid.
cell_id = beahiv.latlon_to_cell(51.5074, -0.1278, side_length=500)

beahiv.decode(cell_id)
# CellIndex(q=707, r=-145, side_length=500, orientation=<Orientation.FLAT: 1>)

beahiv.centroid(cell_id)  # (530250.0, 180566.3) in EPSG:27700 metres
beahiv.centroid(cell_id, latlon=True)  # back to (lat, lon)
beahiv.cell_polygon(cell_id)  # 6 vertices, generated on demand
beahiv.get_neighbours(cell_id)  # 6 neighbouring cell ids
beahiv.k_ring(cell_id, 2)  # all 19 cells within 2 hops
beahiv.distance(cell_id, other)  # hex grid distance between two cells
```

## Core concepts

### Side length

`side_length` must be a whole number of metres, not an index into a fixed
resolution table -- unlike H3's 16 discrete resolutions, any integer
satisfying `1 <= side_length <= 1,048,575` (`SIDE_LENGTH_MAX` in
`cell_id.py`, the 20-bit ceiling) is a valid grid to index against, in
either orientation. `encode`/`encode_morton` raise `ValueError` outside
that range: `side_length <= 0` isn't a size, and anything above the
ceiling doesn't fit the 20-bit field.

Two things this does *not* enforce:

- **Whole metres only.** `side_length` must be an `int` -- a float such
  as `500.5` isn't rejected with a helpful message, it fails at the bit
  shift (`TypeError`, since Python won't `<<` a float) inside `encode`.
  Round or truncate before calling.
- **No cross-resolution nesting -- not even at integer multiples.**
  Cells at different `side_length` values are independent lattices
  scaled by `s`, not levels of a hierarchy, and this isn't fixed by
  choosing a "clean" ratio: a regular hexagon can't be exactly tiled by
  smaller regular hexagons of the same orientation for *any* integer
  factor, the way a square cleanly quarters for a quadtree. Picking
  `side_length=300` as "3x" a `side_length=100` grid doesn't produce
  9 small cells per big cell either -- most straddle the big cell's
  boundary instead of nesting inside it. There is no parent/child
  relationship here at all, clean-multiple or not, unlike H3's fixed
  resolution hierarchy. Mixing side lengths in one dataset is fine as
  long as each cell id carries its own (which it does, by construction),
  but don't expect `k_ring`/aggregation across different side lengths.

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
x = side_length * sqrt(3) * (q + r / 2)  # POINTY
y = 1.5 * side_length * r

x = 1.5 * side_length * q  # FLAT
y = side_length * sqrt(3) * (r + q / 2)
```

Cartesian → axial is the inverse of these formulas followed by standard
cube-coordinate rounding — this is what guarantees an exact, deterministic
cell for every point, with no ambiguity at cell boundaries.

### Cell identifiers

Every cell is a single `uint64`, reversible with no lookup table:

```text
bit 63      orientation   1 bit    0 = POINTY, 1 = FLAT
bits 62-43  side_length  20 bits   whole metres, 1..1,048,575
bits 42-22  q (offset)   21 bits   q + Q_OFFSET
bits 21-0   r (offset)   22 bits   r + R_OFFSET
```

`side_length` is stored directly as a literal metre value rather than an
index into a resolution table, so any grid spacing that fits the bit
budget is usable without registering it anywhere first. q/r are stored
offset-biased (shifted into an unsigned range) so both fields decode with
plain arithmetic and no sign handling. 21/22 bits per axial coordinate is
sized generously: even at a **1 metre** side length — far finer than
typically needed — the whole British National Grid extent (700km ×
1300km) indexes fully, in both orientations, with room to spare.

An optional Morton-coded variant (`encode_morton` / `decode_morton`) packs
the same fields but bit-interleaves q/r for spatial locality, useful for
DuckDB clustering, Parquet sort-order pruning, and range scans.

### Neighbours, distance, rings

Neighbours are pure axial arithmetic — no geometry is constructed:

```python
NEIGHBOUR_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
```

Distance converts to cube coordinates and applies the standard hex grid
distance formula. `k_ring(cell_id, k)` returns all cells within `k` hops,
satisfying `N(k) = 1 + 3k(k+1)`.

### Geometry

Polygons are never stored — `cell_polygon` regenerates six vertices on
demand from the cell centre, at 30/90/150/210/270/330° (POINTY) or
0/60/120/180/240/300° (FLAT).

## Bulk operations

`latlon_to_cell`, `bng_to_cell`, and `centroid` accept arrays transparently —
pass a list, a numpy array, or a pandas Series and get one back:

```python
cell_ids = beahiv.latlon_to_cell(lats, lons, side_length=500)  # lats/lons: array-like
```

A plain scalar still takes the pure-Python path with no numpy import and
sub-microsecond latency; anything else dispatches to the numpy-vectorised
implementation in `beahiv.batch`, which also remains available directly
for callers who want an unambiguous vectorised call:

```python
from beahiv.batch import latlon_to_cell_batch, bng_to_cell_batch, cell_to_latlon_batch

cell_ids = latlon_to_cell_batch(lats, lons, side_length=500)
```

A missing coordinate (`NaN`) is an absent point rather than an error: it
encodes to `INVALID_CELL_ID` (`0`), which no valid `encode` can produce.

### pyarrow

`latlon_to_cell` and `bng_to_cell` also take a pyarrow `Array` or
`ChunkedArray`, and give a `uint64` `Array` back — Arrow in, Arrow out.
Nulls arrive as `NaN` and so encode to `INVALID_CELL_ID`:

```python
import pyarrow as pa

cell_ids = beahiv.bng_to_cell(pa.array(xs), pa.array(ys), side_length=202)
```

This is what makes beahiv usable as a vectorised UDF in an Arrow-native
query engine (DuckDB, Polars), where a per-row Python call would dominate:

```python
con.create_function(
    "beahiv_cell",
    lambda x, y: beahiv.bng_to_cell(x, y, 202),
    [DOUBLE, DOUBLE], UBIGINT, type="arrow",
)
```

pyarrow is **not** a runtime dependency: the import happens lazily, on a
branch only reachable when the caller has already handed us a pyarrow
object. The `beahiv[arrow]` extra exists to pin a version and advertise
the capability, not to make the feature work.

### Filling a polygon with cells

`beahiv.polyfill` covers a Shapely polygon with hex cells — the one bulk
operation that does need a point-in-polygon query, so it's kept out of
the core indexing modules (`geo.py`, `geometry.py`, ...), which never do
spatial joins:

```python
# polygon must be in EPSG:27700
cell_ids = beahiv.polyfill(polygon, side_length=500, orientation=Orientation.FLAT)
```

`predicate` (default `"overlap"`) controls what counts as covering,
matching h3's `contain` vocabulary: `"overlap"` (any part of the hex
touches the polygon), `"center"` (hex centre inside the polygon), or
`"full"` (hex entirely inside the polygon).

## Testing

```bash
uv run pytest
```

Property tests cover:

- **Round trip** — `decode(encode(q, r, s, o)) == (q, r, s, o)` for random
  and boundary coordinates, including the 1m/full-GB-extent case above.
- **Neighbour symmetry** — if B is a neighbour of A, A is a neighbour of B.
- **Area equality** — random cells all have identical area, matching
  `A = (3√3/2)s²`, within 1e-9 relative tolerance.
- **Coordinate stability** — a random point, converted to a cell and back
  to a polygon, always lies within that polygon.

## API reference

| Function | Description |
| --- | --- |
| `encode(q, r, side_length, orientation)` | Build a cell id |
| `decode(cell_id)` | Recover `CellIndex(q, r, side_length, orientation)` |
| `latlon_to_cell(lat, lon, side_length, orientation)` | WGS84 → cell id (scalar, array-like, or pyarrow) |
| `bng_to_cell(x, y, side_length, orientation)` | EPSG:27700 → cell id, no WGS84 round trip (scalar, array-like, or pyarrow) |
| `centroid(cell_id, latlon=False)` | Cell centre → EPSG:27700 (default) or WGS84 (`latlon=True`) |
| `cell_centre(cell_id)` | Cell centre in EPSG:27700 |
| `cell_polygon(cell_id)` | Six vertices in EPSG:27700 |
| `get_neighbours(cell_id)` | Six neighbouring cell ids |
| `distance(cell_a, cell_b)` | Hex grid distance |
| `k_ring(cell_id, k)` | All cells within `k` hops |
| `encode_morton` / `decode_morton` | Z-order variant of `encode`/`decode` |
| `polyfill(polygon, side_length, orientation, predicate)` | Every cell id covering a Shapely polygon |

