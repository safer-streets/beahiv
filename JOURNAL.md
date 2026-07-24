# Journal — `beahiv`

The task/design log for this repo, newest first. Every task/PR gets an entry recording **Why**,
**What**, **Design decisions**, and **Follow-ups** — see
[Task & Design Summaries](AGENTS.md#task--design-summaries) in [AGENTS.md](AGENTS.md) for the rules.
Write the entry as part of the change, not after the fact.

<!-- New entries go directly below this line. -->

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

