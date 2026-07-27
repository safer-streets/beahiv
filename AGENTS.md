# Agent Guidelines for `beahiv`

This file instructs AI agents acting as developer, reviewer, and QA for this repository.

## Project Overview

`beahiv` (**B**ritish **E**qual-**A**rea **H**exagonal, **I**ndex, that's **V**ersatile) is a
small, dependency-light hexagonal spatial index native to **EPSG:27700** (British National Grid).
It's a local alternative to H3: exact closed-form geometry instead of spherical/icosahedral
approximation, in exchange for only covering Great Britain. See [README.md](README.md) for the
full design rationale ("Why not H3?"), design goals, and API reference — read it before making
architectural changes.

The library is small enough to read in full; do that before extending it. Key modules under
[src/beahiv/](src/beahiv/):

| File | Role |
| ---- | ---- |
| [orientation.py](src/beahiv/orientation.py) | `Orientation` enum (POINTY/FLAT) — see the "Orientation" gotcha below |
| [coords.py](src/beahiv/coords.py) | Scalar axial ↔ Cartesian (EPSG:27700 metres) conversion, cube-coordinate rounding |
| [cell_id.py](src/beahiv/cell_id.py) | 64-bit cell id bit layout: `encode`/`decode`, `CellIndex` |
| [morton.py](src/beahiv/morton.py) | Z-order (Morton) variant of `encode`/`decode` — same fields, bit-interleaved for spatial locality |
| [geo.py](src/beahiv/geo.py) | Public geographic interface: `latlon_to_cell`, `bng_to_cell`, `centroid` (WGS84 ↔ EPSG:27700 ↔ cell id) |
| [geometry.py](src/beahiv/geometry.py) | On-demand cell geometry: `cell_polygon` (nothing stored) |
| [neighbours.py](src/beahiv/neighbours.py) | Pure axial arithmetic: `get_neighbours`, `distance`, `k_ring` |
| [batch.py](src/beahiv/batch.py) | numpy-vectorised equivalents of the scalar API, for bulk encode/decode |
| [points.py](src/beahiv/points.py) | `point_to_cell` — Shapely/geopandas point geometry (EPSG:27700 only) → cell ids (needs Shapely) |
| [polyfill.py](src/beahiv/polyfill.py) | `polyfill(polygon, ...)` — the one function that does point-in-polygon queries (needs Shapely) |

Tests are in [tests/](tests/), one `test_*.py` per module plus [tests/_geom_helpers.py](tests/_geom_helpers.py)
(deliberately Shapely-free geometry helpers — see below).

## Toolchain

| Tool | Command |
| ---- | ------- |
| Package manager | `uv` |
| Linter / formatter | `ruff` (`uv run ruff check`, `uv run ruff format`) |
| Type checker | `ty` (`uv run ty check src/ tests/`) |
| Tests | `uv run pytest` |
| Install dev deps | `uv sync` (the `dev` dependency group is included by default) |

CI ([.github/workflows/lint-test.yml](.github/workflows/lint-test.yml)) runs the same gates on
push/PR, but still run them locally before considering a change complete. There is no pre-commit
configured.

## Quality Gates

All of the following must pass before any change is considered complete:

```sh
uv run ruff check src/ tests/          # zero lint errors
uv run ruff format --check src/ tests/ # zero formatting issues
uv run ty check src/ tests/            # zero type errors
uv run pytest                          # all tests pass
```

Scope `ruff`/`ty` to `src/ tests/`, not `.` — `ruff format` also reformats Python fences inside
`README.md`, which is not wanted (those blocks are illustrative, not real importable code).

There is no coverage gate configured. The whole point of this library is exact, closed-form
arithmetic — every new code path (a new orientation branch, a new predicate, a new bit-layout
field) should get a directly corresponding test rather than being covered incidentally.

## Developer Rules

- **EPSG:27700 is the native CRS; WGS84 is accepted only at the boundary.** All indexing, geometry,
  and arithmetic happens in EPSG:27700 metres. Only `geo.py` and its vectorised mirror `batch.py`
  import `pyproj` and project; everything else — `cell_id.py`, `coords.py`, `geometry.py`,
  `neighbours.py`, `morton.py`, `points.py`, `polyfill.py` — never does and never should.
- **Nothing outside `geo.py`/`batch.py` reprojects, including `points.py`.** Shapely geometry
  carries no CRS — no `.crs`, and the GEOS SRID slot is always `0` because geopandas doesn't set
  it — so there is nothing to reproject *from* and inferring one would be a guess. `point_to_cell`
  therefore requires EPSG:27700 input and only *validates*: `points._check_crs` raises when a
  geopandas container declares any other CRS, duck-typed on `.crs` so no geopandas (or `pyproj`)
  import is needed. Don't "improve" this into automatic reprojection; callers use `.to_crs(27700)`,
  or `latlon_to_cell` for WGS84.
- **Validate lat/lon against EPSG:27700's area of use before projecting.** Outside
  `lat ∈ [49.75, 61.01]`, `lon ∈ [-9.01, 2.01]` (`pyproj.CRS.from_epsg(27700).area_of_use`), PROJ
  *extrapolates* rather than erroring — a swapped or garbage lat/lon can produce an (x, y) millions
  of metres off, which then either overflows the q/r bit budget (loud) or silently wraps into a
  bogus but "valid" cell (quiet, and worse) depending on `side_length`. `geo._check_in_area_of_use`
  guards the scalar path and `batch._check_in_area_of_use_batch` mirrors it for arrays — keep both
  in sync if the bounds ever change, and don't remove either guard to "simplify" a call site.
- **FLAT and POINTY are different lattices, not a relabelling of the same one.** FLAT's axial basis
  is POINTY's rotated +30° about the origin. The same `(q, r, side_length)` centres on a
  *different physical point* under each orientation, and the two only agree at `(0, 0)`. Never
  compare, reuse, or mix `q`/`r` (or cell ids) across orientations. See the docstrings in
  [coords.py](src/beahiv/coords.py) and [orientation.py](src/beahiv/orientation.py).
- **Scalar and batch implementations are intentionally separate code, not one calling the other.**
  `batch.py` re-implements the same formulas from `coords.py`/`cell_id.py` in numpy rather than
  looping over the scalar functions, because single-point lookups meet a sub-microsecond target
  that a numpy call (even on one element) would blow. When you fix a bug or change a formula in
  one, check whether the same bug/formula exists in the other — nothing enforces they stay in
  sync. (This is exactly how `cartesian_to_axial_batch`'s POINTY branch once drifted from the
  scalar version — a "simplification" that swapped `qf`/`rf` instead of recomputing them looked
  equivalent but wasn't; `ty` doesn't catch it, only a same-orientation batch-vs-scalar test does.)
- **`latlon_to_cell`, `bng_to_cell`, and `centroid` dispatch transparently on scalar vs array-like
  input.** A plain `int`/`float` takes the pure-Python path (no numpy import); anything else
  (`list`, `np.ndarray`, pandas `Series`, ...) dispatches to the `beahiv.batch` equivalent. This is
  implemented with `@overload` + `numpy.typing.ArrayLike`, not a `float | np.ndarray` union — a
  plain union return type makes every call site's return type ambiguous to `ty`/pyright, and a
  compound `isinstance(a, T) and isinstance(b, T)` check does **not** let a type checker narrow
  the negative (else) branch, because "not (A and B)" doesn't imply "not A" for either variable
  individually. If you add a fourth dispatching function, follow the existing `@overload` pattern
  in [geo.py](src/beahiv/geo.py) rather than a bare union.
- **`shapely` is a core dependency, not optional**, despite only `polyfill.py` using it — `polyfill`
  is exported eagerly from `beahiv/__init__.py`, so `import beahiv` always needs it installed. This
  was a deliberate tradeoff (flat top-level API over a shapely-free `import beahiv`); don't
  "fix" it by moving `shapely` back to an extra without also reverting the eager top-level export,
  or `import beahiv` will break for everyone relying on `beahiv.polyfill`.
- **`beahiv.polyfill` is both a submodule name and a top-level function name.** `from beahiv import
  polyfill` and `beahiv.polyfill(...)` (after `import beahiv`) both correctly resolve to the
  function. The only broken pattern is `import beahiv.polyfill` followed by
  `beahiv.polyfill.polyfill(...)` (expecting module-then-attribute) — don't write code or docs that
  assume that access pattern.
- **No comments explaining *what* the code does.** Only add one when the *why* is non-obvious — a
  hidden constraint, a subtle invariant, a workaround for specific behaviour. This is the existing
  style throughout `cell_id.py`, `coords.py`, `geo.py`; match it.
- **Don't add abstractions ahead of need.** This codebase favours a few explicit lines over a
  premature helper (e.g. `_check_in_area_of_use` is duplicated in scalar/batch form rather than
  factored through a shared numpy-only helper that the scalar path would then have to import numpy
  to use).

## Reviewer Checklist

When reviewing a PR or diff, check:

1. **CRS discipline** — no new `pyproj` import outside `geo.py` and `batch.py`; any new geometry
   stays in EPSG:27700 metres.
2. **Orientation correctness** — any new formula involving `q`/`r` is derived per-orientation from
   first principles (or delegates to the existing `coords.py`/`batch.py` functions), never by
   algebraically "simplifying" one orientation's formula from the other's.
3. **Scalar/batch parity** — a change to a scalar formula (`coords.py`, `cell_id.py`,
   `geometry.py`) has a matching change in `batch.py`, and vice versa, with a test that compares
   them directly (see `test_batch.py`'s `*_matches_scalar` tests) — not just independent test
   coverage of each.
4. **Bit budget** — any change to `Q_BITS`/`R_BITS`/`SIDE_LENGTH_BITS` in `cell_id.py` keeps the
   64-bit total exact (there's an `assert` for this — don't relax it) and re-validates the
   full-GB-extent-at-1m-resolution test in `test_cell_id.py`.
5. **Dispatch typing** — a new scalar/array dual-mode function uses `@overload` +
   `numpy.typing.ArrayLike`, matching `latlon_to_cell`/`bng_to_cell`/`centroid`, not a bare
   `X | np.ndarray` union return type.
6. **Optional-dependency boundary** — `polyfill.py` and `points.py` are the only modules allowed to
   import Shapely; don't let a Shapely import creep into `cell_id.py`, `coords.py`, `geo.py`,
   `geometry.py`, `neighbours.py`, `morton.py`, or `batch.py`. Nothing may import geopandas at all,
   including those two — `points.py` duck-types on `.geometry`/`.crs` instead, and geopandas is a
   dev dependency purely so [tests/test_points.py](tests/test_points.py) can exercise those
   branches against the real thing.
7. **Docs** — if the public API, dependency list, or design rationale changes, update
   [README.md](README.md) (quickstart, API reference table, and "Core concepts" as relevant).

## QA Rules

- Run the full gate suite (`ruff check`, `ruff format --check`, `ty check`, `pytest`) before
  declaring any task done.
- New tests belong next to the module they cover (`tests/test_<module>.py`), following the
  existing style: plain `pytest` functions, no fixtures/mocking framework, no external data or
  network access — this library has none of either, and should stay that way.
- Prefer property-style tests (round trip, symmetry, invariants) over hard-coded expected values
  where the underlying math makes that possible — see `test_cell_id.py`'s round-trip and extremes
  tests, and `_geom_helpers.py`'s independent (non-Shapely) area/point-in-polygon checks used to
  verify `cell_polygon` from the outside.
- If a test is skipped or marked `xfail`, leave a comment explaining why and when it can be
  removed.

## Task & Design Summaries

Every development task — feature, fix, refactor, rename, or non-trivial investigation — gets a
corresponding entry in [JOURNAL.md](JOURNAL.md). This is not optional and not just for large
changes: it's the record of *why* the code looks the way it does, for the next agent (or the next
you) who won't otherwise have this conversation's context. Write the entry as part of the change,
not backfilled afterwards.

Newest entries go at the top, directly below the `<!-- New entries go directly below this line.
-->` marker. Each entry records:

- **Why** — the motivating request or problem, in a sentence or two.
- **What** — the actual change: files touched, functions added/renamed/removed.
- **Design decisions** — any non-obvious tradeoff and the reasoning behind it (e.g. "chose eager
  export over lazy `__getattr__` for `polyfill` because ..."). Omit this if the change was purely
  mechanical (a rename, a formatting pass).
- **Follow-ups** — anything deliberately deferred, flagged but not acted on, or left inconsistent
  on purpose (e.g. "`JOURNAL.md`'s own header still said `safer-streets-tooling` — fixed here").

Keep entries terse — a few bullet points, not prose paragraphs. If a change doesn't warrant a
Design decisions or Follow-ups bullet, leave it out rather than padding it.

## Repository Layout

```text
src/
  beahiv/
    __init__.py       # public API surface (__all__) -- keep in sync with what's exported
    orientation.py     # Orientation enum (POINTY/FLAT)
    coords.py          # scalar axial <-> Cartesian conversion
    cell_id.py         # 64-bit cell id encode/decode, bit layout, CellIndex
    morton.py          # Z-order variant of encode/decode
    geo.py             # WGS84 <-> EPSG:27700 <-> cell id (latlon_to_cell, bng_to_cell, centroid)
    geometry.py         # cell_polygon (generated on demand, nothing stored)
    neighbours.py       # get_neighbours, distance, k_ring
    batch.py            # numpy-vectorised equivalents of the scalar API
    points.py           # Shapely/geopandas points -> cell ids, EPSG:27700 only (needs Shapely)
    polyfill.py          # polygon -> hex grid (needs Shapely)
tests/
  _geom_helpers.py      # Shapely-free area / point-in-polygon helpers, test-only
  test_*.py             # one per module
README.md
AGENTS.md
JOURNAL.md             # task/design log -- see Task & Design Summaries above
pyproject.toml
uv.lock
```

## Workflow

1. Read [README.md](README.md) first if the change touches public API or design goals — it's the
   source of truth for what this library is trying to be, and changes here should usually be
   reflected there too.
2. Make changes under [src/beahiv/](src/beahiv/).
3. Add or update tests in [tests/](tests/) — a scalar-only change needs a batch-parity check too if
   a batch equivalent exists (see Developer Rules above).
4. Run the full gate suite locally (`ruff check`, `ruff format --check`, `ty check`, `pytest`).
5. If the public API or dependency list changed, update [README.md](README.md).
6. Add a [JOURNAL.md](JOURNAL.md) entry for the task (see Task & Design Summaries above) — do this
   as part of the change, not as cleanup afterwards.
