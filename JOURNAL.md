# Journal — `beahiv`

The task/design log for this repo, newest first. Every task/PR gets an entry recording **Why**,
**What**, **Design decisions**, and **Follow-ups** — see
[Task & Design Summaries](AGENTS.md#task--design-summaries) in [AGENTS.md](AGENTS.md) for the rules.
Write the entry as part of the change, not after the fact.

<!-- New entries go directly below this line. -->

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

