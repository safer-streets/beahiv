# Journal — `beahiv`

The task/design log for this repo, newest first. Every task/PR gets an entry recording **Why**,
**What**, **Design decisions**, and **Follow-ups** — see
[Task & Design Summaries](AGENTS.md#task--design-summaries) in [AGENTS.md](AGENTS.md) for the rules.
Write the entry as part of the change, not after the fact.

<!-- New entries go directly below this line. -->

## Remove geopandas as a dependency entirely — test only the plain-Shapely surface

- **Why** — a plain `uv sync` pulled in geopandas (and transitively pandas, pyogrio) purely to run
  `tests/test_points.py`'s duck-typing checks, even for changes nowhere near `points.py`. This is
  the exact risk flagged as a follow-up when geopandas was first added ("CI installs are heavier
  across the three-OS matrix... if it bites, the geopandas-specific tests could move behind an
  `importorskip`"). Two intermediate approaches were tried and dropped: an `importorskip`'d
  dependency-group still needed installing to actually run those tests anywhere, and a
  `_FakeGeoSeries`/`_FakeGeoDataFrame` stand-in (implementing `.crs`/`.geometry`/`__array__`) added
  real weight to the test file for a container interface no test in the suite is required to cover.
  Settled on simply not testing the `.crs`/`.geometry` container-duck-typing branches at all —
  `point_to_cell` accepts plain `Point`s and lists/ndarrays of them independently of that path, and
  that's the surface this suite now covers.
- **What**
  - `pyproject.toml`: no geopandas anywhere, dev group or otherwise.
  - [tests/test_points.py](tests/test_points.py): dropped every test that needed a `.crs`/`.geometry`
    container — `test_accepts_geoseries_geodataframe_geometryarray_ndarray_and_list` (replaced with
    a slimmer `test_accepts_list_and_ndarray_of_points`), `test_non_default_index_does_not_shift_results`,
    `test_rejects_a_declared_crs_that_is_not_bng`, `test_reprojecting_to_bng_first_is_accepted`, and
    `test_crs_guard_accepts_bng_declared_without_an_epsg_code`. Every remaining test now builds
    input from a plain `list[Point]`/`np.ndarray` via a local `_points()` helper (replacing the old
    `_geoseries()`). Two cases (`test_missing_and_empty_points_become_invalid_cell_id`, the masked-
    path test) pass an explicit `np.array(..., dtype=object)` rather than a raw `list[Point | None]`
    — `ty` doesn't accept a list containing `None` alongside `Point` against `ArrayLike`'s nested-
    sequence branch, only its `_SupportsArray` branch (which an object ndarray satisfies directly).
  - [.github/workflows/lint-test.yml](.github/workflows/lint-test.yml): back to a plain `uv sync`.
  - `AGENTS.md`: reviewer checklist item 6 updated — geopandas isn't imported anywhere including
    `points.py`'s tests, and the container-duck-typing branches are explicitly noted as uncovered
    by this suite (not silently assumed correct).
- **Follow-ups** — `points.py`'s `.crs`/`.geometry` duck-typing (the actual `_check_crs` raising
  path, `GeoDataFrame`-style `.geometry` extraction, index-independence) has no test coverage at
  all now. Deliberately accepted rather than deferred — flagging here so it isn't mistaken for an
  oversight if that logic ever breaks.

## `cell_polygon`/`cell_polygons` return Shapely `Polygon` objects, not vertex tuples

- **Why** — callers needing an actual `Polygon` (`polyfill.py`, tests) had to wrap the output
  themselves with `Polygon(cell_polygon(cell_id))`. Worse, a raw `list[tuple[float, float]]` has
  no guarantee of forming a valid, closed ring — a `Polygon` enforces that structure; the tuple
  list was just deferring that guarantee to every call site.
- **What**
  - [geometry.py](src/beahiv/geometry.py): `cell_polygon` returns `shapely.Polygon` instead of
    `list[tuple[float, float]]`; `cell_polygons` returns `list[Polygon]`. Both now import `shapely`.
  - [polyfill.py](src/beahiv/polyfill.py): dropped the now-redundant `Polygon(cell_polygon(...))`
    wrapping — `cell_polygon` already returns a `Polygon`. Removed the now-unused `Polygon` import.
  - `tests/test_geometry.py`/`tests/test_polyfill.py`: updated for the new return type; added a
    local `_vertices()` helper in `test_geometry.py` that unwraps a `Polygon` back to a plain
    vertex list (via `.exterior.coords`, dropping the closing repeated vertex) before handing it to
    the Shapely-free helpers in `tests/_geom_helpers.py`, which are unchanged and still operate on
    plain tuples.
  - `AGENTS.md`/`README.md`: updated the Shapely-import boundary — `geometry.py` is now a third
    module (with `polyfill.py`/`points.py`) allowed to import Shapely, since it needs `Polygon` to
    construct its return value. It still runs no spatial predicate itself, so the "core indexing
    code does no point-in-polygon query" rule (the actual design goal) is unchanged; only the
    narrower "no Shapely import at all" reading of it was reversed.
- **Design decisions**
  - **`geometry.py` may now import Shapely, but still may not query it.** The previous rule
    conflated two separate things: not doing spatial joins/predicates outside `polyfill.py`, and
    not importing Shapely at all outside `polyfill.py`/`points.py`. Only the first is an actual
    design goal (see README's "Why not H3?" / dependency-light rationale); the second was
    incidental and is what changed here.
  - **`tests/_geom_helpers.py` stays Shapely-free.** Its job is independent verification math
    (shoelace area, ray-casting point-in-polygon) that doesn't lean on Shapely's own
    `.area`/`.contains` to check `cell_polygon`'s output — that's unrelated to whether `cell_polygon`
    itself returns a `Polygon`, so callers unwrap to plain tuples before calling in.

## Implement `cell_polygons`, the vectorised sibling of `cell_polygon`

- **Why** — `geometry.py` shipped a `cell_polygons(cell_ids)` stub (docstring only, no body) as a
  placeholder for a bulk polygon lookup; nothing called it yet.
- **What**
  - [geometry.py](src/beahiv/geometry.py): implemented `cell_polygons`. Decodes/centres the whole
    batch via `batch.cell_centre_batch` (one call, not a Python loop over `cell_polygon`), then
    builds all six vertices per cell with a single vectorised `cos`/`sin` pass over the one
    orientation's angle set. Empty input returns `[]` before touching `cell_centre_batch` (which
    itself raises `IndexError` on an empty array — not something worth working around inside it).
  - `__init__.py`: exported `cell_polygons` alongside `cell_polygon`.
  - `README.md`/`AGENTS.md`: documented the new function and its restriction.
  - `tests/test_geometry.py`: parity test against `cell_polygon` per id (mirrors `test_batch.py`'s
    `*_matches_scalar` pattern), plus empty-input, mixed-orientation, and mixed-side_length cases.
- **Design decisions**
  - **Same-side_length/orientation restriction as `cell_centre_batch`, not a per-cell mixed-grid
    version.** A single angle set and radius only apply to one grid at a time; supporting mixed
    grids would mean re-deriving `axial_to_cartesian` per-element in numpy (formula duplication
    `batch.py` already carries) for a use case nothing calls. Reusing `cell_centre_batch` keeps
    the vectorised centre formula in one place instead of a second copy in `geometry.py`.
  - **Delegates to `batch.cell_centre_batch` rather than looping over `cell_polygon`.** Per the
    "Scalar and batch implementations are intentionally separate code" rule, but the loop-over-
    scalar shortcut here would specifically defeat the point of a vectorised version — `geo.py`'s
    `centroid` already establishes the precedent of a non-`batch.py` module delegating to
    `batch.py` for its array branch.

## Cap `side_length` at 100km, reserve the freed bits at the MSB end

- **Why** — the 20-bit literal-metres field allowed `1..1,048,575`, so a 100km and a 99.999km
  grid were both valid. Those are distinct non-nesting lattices that no operation can combine
  (`distance` raises across side lengths, `k_ring` can't span them), for a difference no analysis
  could see. The ceiling was also larger than Great Britain.
- **What**
  - [cell_id.py](src/beahiv/cell_id.py): `SIDE_LENGTH_BITS` 20 → 17 and `SIDE_LENGTH_MAX` →
    `100_000`; new `RESERVED_BITS`/`RESERVED_SHIFT`/`RESERVED_MASK` occupying bits 63-61.
    `ORIENTATION_SHIFT` moves 63 → 60; q/r shifts and widths are untouched.
  - [morton.py](src/beahiv/morton.py): validated against `SIDE_LENGTH_MASK` where `cell_id.py`
    used `SIDE_LENGTH_MAX` — the two were equal before, so the drift was invisible. Now the cap.
  - [batch.py](src/beahiv/batch.py): `encode_batch` never validated `side_length` at all, so the
    array paths accepted what the scalar path rejected. One scalar check per call, not per row.
  - Tests: reserved bits never set, ids fit a signed int64, the cap holds below the field width,
    and batch/scalar reject the same side lengths. `test_arrow.py` asserted FLAT ids exceed
    `2**63` — true only while orientation was bit 63; it now checks the orientation bit by shift
    and asserts ids stay under `2**63`.
- **Design decisions**
  - **Reserved bits at the MSB end, not spent on q/r.** Puts every id below `2**61`, so ids fit a
    *signed* int64 — consumers get a plain `BIGINT` column instead of an unsigned type or a hex
    string. Spending them on q/r would buy nothing: 21/22 already covers GB down to a 0.445m side
    length (bound by FLAT's `q`, needing `700000/1.5` cells against a capacity of 1,048,575),
    which is well under the 1m floor.
  - **The cap is a range check, not a field width.** 100,000 sits below the 17-bit
    `SIDE_LENGTH_MASK` of 131,071, so nothing masks an oversized value off — it would overflow
    into the orientation bit. That's why every encode path checks, and why an `assert` ties
    `SIDE_LENGTH_MAX <= SIDE_LENGTH_MASK`.
  - **Reserved bits are masked off on decode, not validated.** No encode can set them, and a
    check would cost `decode_batch` a whole extra pass over the array.
- **Follow-ups**
  - **Every FLAT cell id changes value** (orientation moved from bit 63 to 60); POINTY ids are
    unchanged for any side length within the new cap. `safer-streets-tooling` keeps `SIDE_LENGTH
    = 202` and needs no code change, but persisted `crime_counts_beahiv_202` / `beahiv_202_*`
    tables hold stale ids and must be rebuilt. Its tests recompute expectations from beahiv's own
    encoder rather than hard-coding ids, so they pass after a rebuild.
  - Now that ids fit int64, `beahiv_counts` storing `spatial_id` as zero-padded hex could become a
    `BIGINT` column. Not done here — it's a downstream schema change, and the H3 tables use the
    same hex convention for a consistent `spatial_id` type across grids.
  - A significant-figures ladder (3 digits + a power-of-ten exponent, so 1234m and 99,999m are
    rejected outright rather than merely capped) was prototyped and backed out in favour of this
    smaller change. It cost ~25-35ns per encode via a 2800-entry lookup table, or 39-167ns
    computed arithmetically, against a ~250ns baseline encode.

## `point_to_cell` — encode Shapely/geopandas point geometry

- **Why** — point data arrives from geopandas as a geometry column, not as separate x/y sequences.
  Splitting it to call `bng_to_cell` is both awkward and slower than reading the coordinates once.
- **What**
  - New [points.py](src/beahiv/points.py) with `point_to_cell(points, side_length, orientation)`,
    exported from `__init__.py`. Takes a single `Point` (→ `int`) or a
    `GeoDataFrame`/`GeoSeries`/`GeometryArray`/object ndarray/list (→ `uint64` ndarray).
  - Input must already be EPSG:27700. `_check_crs` raises when a geopandas container declares
    anything else; undeclared geometry is taken at its word.
  - geopandas added to the dev group so [tests/test_points.py](tests/test_points.py) exercises the
    duck-typed branches for real — same rationale as pyarrow for `test_arrow.py`.
  - README gets a "Shapely and geopandas" section (including a subsection on why the EPSG:27700
    assumption exists) and an API-table row; AGENTS.md gains a rule pinning the no-reprojection
    decision, and `points.py` is named as the second module allowed to import Shapely.
- **Design decisions**
  - **Not an overload of `bng_to_cell`.** A geometry form is `(geom, side_length)` against the
    existing `(x, y, side_length)`, so the second positional argument would mean two different
    things — `bng_to_cell(p, 100)` binds `100` to `y`. Fixing that needs either a keyword-only
    `side_length` (breaks every positional call) or `*args`, which discards the `@overload` +
    `ArrayLike` typing the reviewer checklist requires. A separate function has neither problem.
  - **One dispatching function, not `point_to_cell` + `points_to_cell`.** Scalar-or-array on a
    single argument is the convention every other entry point here follows, and two public names a
    letter apart for near-identical behaviour is exactly the duplication the `cell_centre` removal
    below was about.
  - **No reprojection — EPSG:27700 in, or an error.** A working version that reprojected from the
    container's `.crs` (and from an explicit `crs=` for bare geometry) was built and then reverted,
    because the premise doesn't hold at the level the function operates on: Shapely geometry has no
    CRS. There is no `.crs` on a `Point`, and the GEOS SRID slot (`shapely.get_srid`) is always `0`
    since geopandas never populates it — so a `crs=` argument is the caller asserting something the
    data cannot confirm, and the geopandas-container case doesn't justify the machinery on its own.
    `.to_crs(27700)` is one call, and `latlon_to_cell` already covers WGS84 with an area-of-use
    check the reprojecting version had to route through WGS84 to inherit.
  - **The CRS *guard* survives the revert**, because it validates rather than infers, needs no
    `pyproj` (duck-typed `crs.to_epsg()`), and prevents a silent wrong answer: WGS84 degrees read as
    metres land every point within a few metres of the grid origin and encode to a valid, completely
    wrong cell — the same failure class as `geo._check_in_area_of_use`.
  - **Returns numpy, never a `Series`.** `gdf["cell_id"] = point_to_cell(gdf, 202)` is then
    positional and can't realign against a non-default index. Matches the `batch.py` decoders.
  - **Non-point geometries raise.** `shapely.get_x` returns NaN for a polygon, so without the
    `get_type_id` check a polygon column would encode to all-`INVALID_CELL_ID` silently.
  - **`get_x`/`get_y`, never `get_coordinates`** — the latter drops empty geometries entirely,
    returning fewer rows than there were points and misaligning everything after the first empty.
  - **Fast path when nothing is missing.** `get_x` raises on an empty point, so gaps force a masked
    gather-and-scatter; that costs ~2x a straight read (26ms vs 13ms per 500k), ~20% of the whole
    call, and a column with no gaps is the normal case. Both branches are covered by
    `test_masked_path_agrees_with_the_all_present_fast_path`.
- **Follow-ups**
  - The scalar form is for completeness, not speed: ~5.7us against 0.94us for
    `bng_to_cell(x, y, ...)`, essentially all of it Shapely attribute access (`p.x`/`p.y` 2.7us,
    `is_empty` 1.3us). Documented in the docstring rather than optimised.
  - If reprojection is ever wanted back, the reverted design is recorded above: route non-BNG
    through WGS84 into `latlon_to_cell` (inheriting the area-of-use guard), cache
    `Transformer.from_crs` (~19ms to build), and note that a BNG → WGS84 → BNG round trip moves a
    point ~1mm, flipping the cell for 0.0005–0.007% of points sitting that close to a hex edge.
  - geopandas in the dev group pulls in pandas and pyogrio, so CI installs are heavier across the
    three-OS matrix. Acceptable for now; if it bites, the geopandas-specific tests could move
    behind an `importorskip`.

## Remove `cell_centre` — `centroid` already covers it

- **Why** — `beahiv.cell_centre(cell_id)` and `beahiv.centroid(cell_id)` returned exactly the same
  EPSG:27700 `(x, y)`; the public API had two names for one operation.
- **What**
  - `geometry.cell_centre` deleted, and dropped from `beahiv/__init__.py`'s imports and `__all__`.
    `geometry.py` is now just `cell_polygon` (which never called `cell_centre` — it already
    inlined `decode` + `axial_to_cartesian`).
  - `geo.centroid`'s scalar branch inlines the same two lines instead of delegating; `geo.py` no
    longer imports `geometry`.
  - `polyfill`'s `"center"` predicate calls `axial_to_cartesian(q, r, ...)` on the loop variables
    it already has, rather than encoding a cell id and decoding it straight back out.
  - Tests updated to use `centroid`. `test_centroid_defaults_to_bng` was
    `centroid(cell_id) == cell_centre(cell_id)`, which would now be a tautology — it instead
    reprojects the default `(x, y)` and checks it against the `latlon=True` result.
  - `README.md` API table row removed; `AGENTS.md` module table and repo-layout comment updated.
- **Design decisions**
  - Kept `centroid` (not `cell_centre`) because it's the one that also does WGS84 and array/pyarrow
    dispatch; `cell_centre` was the strict subset.
  - `polyfill` was *not* pointed at `geo.centroid` — that would drag a `pyproj`-importing module
    into the one module that's meant to stay at arm's length from the core indexing path, to
    recompute a centre from q/r it was holding anyway.
- **Follow-ups**
  - `batch.cell_centre_batch` keeps its name (it still backs `centroid`'s array branch), so the
    scalar name it mirrored no longer exists. Renaming it to `centroid_batch` would be a separate
    public break — not done here.

## pyarrow round-trip for `latlon_to_cell`/`bng_to_cell`; NaN handling on the BNG path

- **Why** — encoding a whole crime dataset inside a DuckDB vectorised (`type="arrow"`) UDF, where
  the caller had to unwrap pyarrow to numpy on the way in and re-wrap on the way out. Arrow input
  already worked (numpy consumes it via the buffer protocol); the return type and nulls did not.
  Separately, `bng_to_cell` had no NaN handling at all, unlike `latlon_to_cell_batch` — a nullable
  column raised `ValueError: q or r out of representable range` from `encode_batch`.
- **What**
  - `batch.bng_to_cell_batch` — the BNG counterpart of `latlon_to_cell_batch`, NaN → `INVALID_CELL_ID`.
    `geo.bng_to_cell`'s array branch now delegates to it instead of inlining
    `cartesian_to_axial_batch` + `encode_batch`, so both public entry points have a matching
    `*_batch` and the NaN semantics live in one place.
  - `geo._is_arrow` / `geo._match_arrow` — a pyarrow array in gives a `uint64` pyarrow array back,
    for both `latlon_to_cell` and `bng_to_cell`. Numpy and scalar returns are unchanged.
  - `pyproject.toml`: an `arrow` extra (`pyarrow>=17`), and `pyarrow` in the dev group so the
    round-trip is actually exercised.
  - `tests/test_arrow.py` (new) and three `bng_to_cell_batch` tests in `test_batch.py`
    (scalar parity, NaN, bit-budget rejection). README: a "pyarrow" subsection under Bulk
    operations, the extra in Install, and the two API-reference rows.
- **Design decisions**
  - *Lazy, unguarded `import pyarrow` inside the wrap, not a module-level `try: import`.* A caller
    can only hand us a pyarrow object if pyarrow is already imported in their process, so the import
    can never fail where it runs — pyarrow stays out of the runtime requirements entirely. The extra
    pins a version and advertises the capability; it is not what makes the feature work.
  - *Dispatch on the argument, never on whether pyarrow is importable.* With a module-level guarded
    import, `bng_to_cell(chunked_array, ...)` would return `ndarray` without the extra and
    `pa.Array` with it — the same call returning different types depending on how the package was
    installed. Keying on the input makes that impossible, and means no second CI configuration is
    needed to keep the non-Arrow path honest.
  - *No area-of-use check on the BNG path.* The lat/lon guard exists because PROJ extrapolates
    silently outside EPSG:27700's box; there is no projection here, so a far-off coordinate is just
    a far-off cell, and one far enough to matter is caught by `encode_batch`'s q/r range check.
- **Follow-ups**
  - `centroid` (and the `batch.py` decoders) still take Arrow input but always return numpy — the
    round-trip is only on the two `*_to_cell` functions. Returning a struct/tuple of Arrow arrays
    is a bigger design question, deferred until something needs it.
  - The scalar path raises a bare `ValueError` from `int(nan)` for a NaN coordinate rather than
    returning `INVALID_CELL_ID`, so scalar and batch disagree on what a missing coordinate means.
    Left as-is: `latlon_to_cell` has the same asymmetry (its scalar guard raises "outside area of
    use" for NaN), and changing it is an API decision, not a bug fix.

## README: side length validity, non-hierarchy clarification; Morton default issue

- **Why** — README didn't document what values `side_length` actually accepts, nor make clear
  that the lack of a fixed resolution table means no parent/child hierarchy at all — including
  the tempting-but-wrong assumption that an integer multiple (e.g. 300m vs 100m) would nest.
- **What**
  - Added a "Side length" subsection under Core concepts: valid range `[1, SIDE_LENGTH_MAX]`
    (1,048,575), what `encode`/`encode_morton` reject and why, and two unenforced edges verified
    by hand — a float `side_length` fails inside `encode` with a bare `TypeError` (Python won't
    `<<` a float) rather than a clean `ValueError`, and cells never nest across `side_length`
    values, not even at integer ratios (verified empirically: tiling a 300m cell's footprint with
    100m cells gives 5 fully-inside vs 11 straddling the boundary, out of 16 total — a regular
    hexagon has no exact same-orientation subdivision into smaller regular hexagons, unlike a
    square's quadtree split).
  - Filed [safer-streets/beahiv#1](https://github.com/safer-streets/beahiv/issues/1): should the
    default `encode`/`decode` layout be Morton-coded instead of plain concatenation? Benchmarked
    scalar `encode`/`decode` vs `encode_morton`/`decode_morton` (`timeit`, 200k iters): Morton is
    11.9x slower to encode (244ns → 2.9us) and 4.5x slower to decode (837ns → 3.8us), because
    `_interleave`/`_deinterleave` are pure-Python bit-by-bit loops. Also noted there: no
    vectorised (`batch.py`) Morton path exists yet at all.
- **Follow-ups** — resolution of issue #1 is pending; if Morton becomes default (or even just a
  commonly-recommended opt-in), `encode_morton_batch`/`decode_morton_batch` need writing first.

## Consolidate dev tooling into `[dependency-groups]`

- **Why** — `uv add --dev pre-commit` had created a `[dependency-groups].dev` alongside the
  existing `dev` extra; plain `uv sync` includes the group (default) but not the extra, and
  exact-syncs, so it silently uninstalled pytest/ruff/ty from the venv.
- **What** — moved pytest/ruff/ty into `[dependency-groups].dev` and deleted the
  `[project.optional-dependencies]` dev extra; install command is now plain `uv sync`
  (README, AGENTS.md toolchain table, CI workflow all updated).
- **Design decisions** — dependency group over extra: it's the PEP 735 convention for dev
  tooling, `uv sync` needs no flags, and dev tools are no longer published to consumers as an
  installable `beahiv[dev]` extra.

## CI workflow: strip safer-streets-tooling leftovers

- **Why** — `lint-test.yml` was copied from `safer-streets-tooling` and still checked out
  `safer-streets-core`, which is not a dependency of this repo.
- **What** — removed the `safer-streets-core` checkout and the sibling-directory arrangement
  that existed only to support it (`path:`/`working-directory: safer-streets-tooling`); fixed
  `uv sync --group dev` → `--extra dev` (dev is an optional extra here, not a dependency
  group); aligned the lint/test steps with the AGENTS.md quality gates (added
  `ruff format --check`, scoped ruff/ty to `src/ tests/`); dropped the `SAFER_STREETS_DATA_DIR`
  env var and the `htmlcov/` coverage upload (nothing here generates coverage). Updated
  AGENTS.md's "no CI configured" note.

## Pre-initial-commit review fixes

- **Why** — review pass before the first commit found stale docs and small consistency gaps.
- **What**
  - `README.md`: quickstart `decode`/`centroid` output re-captured against the current FLAT
    default (was stale POINTY-era output); install/test commands aligned with AGENTS.md
    (`uv sync --extra dev`, `uv run pytest`); dropped the free-floating `Version: 0.1` line
    (pyproject is the single source of truth).
  - `__init__.py`: docstring acronym expansion aligned with README/AGENTS.md.
  - `batch.decode_batch`: replaced hardcoded `(1 << 20) - 1` with `SIDE_LENGTH_MASK` and `0b1`
    with `ORIENTATION_MASK`; same `ORIENTATION_MASK` fix in `morton.decode_morton` — these
    literals would have silently drifted if the bit layout ever changed.
  - Added `src/beahiv/py.typed` so downstream type checkers see the annotations.
  - `.gitignore`: added `.ruff_cache/`, `dist/`, `.claude/settings.local.json`.
- **Follow-ups**
  - `centroid` dispatches scalars via `isinstance(cell_id, int)`, so a `np.uint64` scalar
    (e.g. one element of a batch result) takes the batch path and returns 0-d arrays; callers
    currently cast with `int(...)`. Decide whether to accept `np.integer` or document the cast.
  - No LICENSE file and no packaging metadata (`license`, `readme`, `authors`, `urls` in
    pyproject) — needed before any PyPI release.
  - README doesn't state that parent/child hierarchy is deliberately out of scope (free-form
    `side_length` means no fixed resolution set, so no exact nesting) — worth a sentence, since
    it's the first thing an H3 user will look for.

## Initial implementation (backfilled)

- **Why** — a local, equal-area alternative to H3 for analysis confined to Great Britain, native
  to EPSG:27700 (see README "Why not H3?").
- **What** — full initial library: `orientation`, `coords`, `cell_id`, `morton`, `geo`,
  `geometry`, `neighbours`, `batch`, `polyfill`, plus one test module per source module and
  Shapely-free geometry helpers in `tests/_geom_helpers.py`.
- **Design decisions**
  - 64-bit id layout 1/20/21/22 (orientation/side_length/q/r): side_length stored as literal
    metres rather than a resolution-table index; q/r offset-biased so decode is plain
    arithmetic. 21/22 bits covers the full GB extent at a 1m side length (tested).
  - FLAT is the default orientation throughout.
  - Scalar and batch paths are deliberately separate implementations (sub-microsecond scalar
    target rules out numpy on the scalar path); parity is enforced by `*_matches_scalar` tests,
    not by code sharing.
  - `shapely` is a core dependency despite only `polyfill` using it — eager top-level export of
    `polyfill` was chosen over a lazy/optional import so `beahiv.polyfill(...)` just works.
  - lat/lon validated against EPSG:27700's area of use before projecting, because PROJ
    extrapolates rather than erroring outside it.
- **Follow-ups** — no CI or pre-commit yet; quality gates (`ruff check`, `ruff format --check`,
  `ty check`, `pytest`) run manually per AGENTS.md.

