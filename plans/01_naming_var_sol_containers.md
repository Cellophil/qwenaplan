# plan_01_naming — `obj.var.x_t` and `obj.sol.x_t` containers (+ `_pu_t` views)

## Context

Today every component has two parallel naming conventions for its time-varying
data:

| Today | What it really is |
|---|---|
| `gen.p` | a `pyoframe.Variable` (symbolic, used in constraints) |
| `gen.p_t` | a `polars.DataFrame` (the solved value) |
| `bus.theta` / `bus.theta_t` | same pattern |
| `line.p` / `line.p_t` | same pattern |
| `storage.soc` / `storage.soc_t` etc. | same pattern |

This is asymmetric and ambiguous: `gen.p` reads like a value but is a Variable,
and `gen.p_t` reads like a Variable but is a DataFrame. There's also no place to
hang related quantities that *aren't* the primary variable but are derived from
it (per-unit views, duals, etc.).

Decision (from the chat): introduce **two containers per component**, both using
the `_t` suffix for time-vectorized things (`_t` makes "vector over snapshots"
explicit on every read site):

- **`obj.var.<name>_t`** — the pyoframe `Variable` (or expression). Used in
  constraint construction, custom user constraints, etc.
- **`obj.sol.<name>_t`** — the solved value as a tidy Polars DataFrame
  (snapshot index + value column). Replaces today's `obj.<name>_t`.

Plus a small set of derived per-unit views where they make sense, available on
*both* containers as expressions / DataFrames:

- `gen.var.p_pu_t` and `gen.sol.p_pu_t` — `p / p_nom`. Capacity factor view.
- `storage.var.soc_pu_t` and `storage.sol.soc_pu_t` — `soc / e_nom`. Fill level.
- `storage.var.p_pu_t` and `storage.sol.p_pu_t` — `(p_out - p_in) / max(p_nom_in, p_nom_out)`
  net power as fraction of nameplate. (Sign: + = discharging.)

These are pure expressions / pure transforms; they don't add new variables to
the LP. They give the user a cheap, consistent way to compare units of
different scale.

## Goals

1. Migrate every place that constructs `pf.Variable` from `self.x = pf.Variable(...)`
   to `self.var.x_t = pf.Variable(...)`, where `self.var` is a small attribute
   bag.
2. Migrate every solution accessor `obj.x_t` from a top-level property to
   `obj.sol.x_t`, with the same return shape (Polars DataFrame keyed by
   snapshot, value column named after the variable).
3. Add `_pu_t` views on `var` and `sol` for generator output, storage SOC, and
   storage net power.
4. Update every test, the README, and `TEST_GUIDELINES.md` so the new shape is
   the only documented one.
5. **No physics or LP changes.** Pure renaming + container introduction. The
   78-test suite must stay green throughout.

## Non-goals

- No change to the parameter convention (`p_nom`, `marginal_cost`, etc. stay
  on the component itself — they're scalar / static, not time-vectorized).
- No new optimization features (no extendable, no UC, etc.).
- No backwards-compat shims. The old `gen.p` / `gen.p_t` names go away in one
  commit; the codebase is small enough that a single mechanical sweep is
  cleaner than a deprecation period.
- Loads keep `load.p_t` returning the parameter — `Load` has no Variable,
  so there is no `load.var.p_t`. Put the parameter view on `load.sol.p_t` for
  symmetry, and document the asymmetry.

## Why this naming

- **`_t` everywhere** for time-vectorized fields. Eliminates the asymmetry
  where `gen.p` is a vector but reads scalar.
- **`var` / `sol`** instead of `vars` / `sols` because singular reads better at
  call sites: `gen.var.p_t * gen.marginal_cost` vs `gen.vars.p_t * ...`.
- **Static parameters stay on the component**: `gen.p_nom`, `gen.marginal_cost`,
  `gen.bus`, `gen.carrier`, `bus.v_nom`, `bus.x`, `bus.y`, `line.x_pu`,
  `line.s_nom`, `link.efficiency`, `storage.e_nom`, etc. They aren't
  vectorized; they don't belong in `var` / `sol`.

## Inventory — what migrates

Time-varying fields per component (current name → new name):

### `Bus`
- `bus.theta` → `bus.var.theta_t`
- `bus.theta_t` → `bus.sol.theta_t`

### `ACLine`
- `line.p` → `line.var.p_t`
- `line.p_t` → `line.sol.p_t`

### `Link`
- `link.p` → `link.var.p_t`
- `link.p_t` → `link.sol.p_t`

### `Generator`
- `gen.p` → `gen.var.p_t`
- `gen.p_t` → `gen.sol.p_t`
- (new) `gen.var.p_pu_t` — `gen.var.p_t / gen.p_nom` (pyoframe expression)
- (new) `gen.sol.p_pu_t` — DataFrame, `p` column divided by `p_nom`

### `Load`
- `load.p_t` → `load.sol.p_t` (parameter; symmetric placement, documented)
- No `load.var` (loads have no decision variables).

### `_StorageBase` (and via inheritance, `StorageUnit`)
- `s.soc` → `s.var.soc_t`
- `s.p_in` → `s.var.p_in_t`
- `s.p_out` → `s.var.p_out_t`
- `s.soc_t` → `s.sol.soc_t`
- `s.p_in_t` → `s.sol.p_in_t`
- `s.p_out_t` → `s.sol.p_out_t`
- (new) `s.var.soc_pu_t` — `soc / e_nom`
- (new) `s.sol.soc_pu_t` — `soc / e_nom` (DataFrame)
- (new) `s.var.p_pu_t` — `(p_out - p_in) / nameplate` where
  `nameplate = max(p_nom_in or 0, p_nom_out or 0)`. If both `None`, this view
  is undefined — raise on access with a clear message.
- (new) `s.sol.p_pu_t` — same, evaluated.

`StorageUnit` keeps the legacy aliases `s.p_store`, `s.p_dispatch`,
`s.eff_store`, `s.eff_dispatch` — these are **not** time-vectorized variable
names, they're alternative names for the *same* attributes, and they're
user-facing. Keep them as scalar property delegates.

### Composites (`Battery`, `PumpedHydroStorage`)

The composites delegate their `var` / `sol` containers to the inner storage
(and, for PHS, the inner generator), so:

- `battery.var.soc_t` ← `battery._storage.var.soc_t`
- `battery.var.p_in_t` ← `battery._storage.var.p_in_t`
- `battery.var.p_out_t` ← `battery._storage.var.p_out_t`
- `battery.var.p_t` — **expression**, `p_out_t - p_in_t` (currently lives at
  `battery.p`). Net power.
- `battery.sol.soc_t`, `battery.sol.p_in_t`, `battery.sol.p_out_t`,
  `battery.sol.p_t` (computed; replaces today's `battery.p_t`).
- `battery.var.p_store_t` / `battery.var.p_dispatch_t` — alias views into
  `var.p_in_t` / `var.p_out_t`.
- `battery.sol.p_store_t` / `battery.sol.p_dispatch_t` — same on solution side.
- (new) `battery.var.soc_pu_t` and `battery.sol.soc_pu_t` (delegate to inner).
- (new) `battery.var.p_pu_t` and `battery.sol.p_pu_t` (uses
  `nameplate = battery.p_nom` directly — easier than the `_StorageBase` case).

For PumpedHydroStorage:
- `phs.var.soc_t` ← `phs._storage.var.soc_t`
- `phs.var.p_t` ← `phs._generator.var.p_t` (electrical output is the gen's
  variable, post-coupling). Today this lives at `phs.p`.
- `phs.var.p_store_t` (pump) ← `phs._storage.var.p_in_t`
- `phs.var.p_dispatch_t` (water) ← `phs._storage.var.p_out_t`
- `phs.sol.*` mirror.
- (new) `phs.var.soc_pu_t` / `phs.sol.soc_pu_t` (delegate to inner storage).
- (new) `phs.var.p_pu_t` / `phs.sol.p_pu_t` — `p / p_nom_turbine`.

## Implementation strategy

Two passes — one in src, one in tests/docs. Single commit per pass keeps the
diff reviewable.

### Pass 1: introduce containers in `src/qwenaplan/`

1. **Add a tiny `_VarContainer` and `_SolContainer` helper** in
   `src/qwenaplan/base.py`. They're plain attribute bags with a clear `__repr__`
   listing the fields they hold. Could literally be `types.SimpleNamespace` if
   we want zero ceremony, but a named class makes inspection cleaner.

2. **`Component` gains two attributes** at construction time:
   - `self.var = _VarContainer()` — empty until `setup_variables` populates it.
   - `self.sol = _SolContainer(self)` — receives `self` so it can lazily
     resolve solution DataFrames by reading `self.var.<name>_t.solution`.
     Implementing `sol` as **dynamic** (computes on access) avoids stale
     solutions across re-solves.

3. **Migrate `setup_variables` in every component** to assign into
   `self.var.<name>_t = pf.Variable(...)` instead of `self.<name> = ...`. Keep
   `setup_variables_for_model` aligned (it uses `self.var.<name>_t` now).

4. **Migrate `setup_constraints` and `setup_objective`** to read from
   `self.var.<name>_t` instead of `self.<name>`. Same change in
   `physics.py::DCPhysics.apply_kirchhoff_*` (read `bus.var.theta_t`,
   `line.var.p_t`).

5. **Replace today's `_t` properties** (`gen.p_t`, `line.p_t`,
   `storage.soc_t`, etc.) with equivalents on `sol`. Centralised
   `_solution_as` (already in `base.py`) keeps doing the rename of pyoframe's
   `solution` column to the variable's friendly name.

6. **Add `_pu_t` views** on both containers. For `var`, return a pyoframe
   expression (`self.var.p_t / self.p_nom`). For `sol`, return a DataFrame
   with the `p`/`soc` column divided by the relevant scalar.

7. **Composites**: rebuild `battery.var` / `phs.var` (and `.sol`) as
   property delegates onto the inner components. The composite's container is
   *not* an independent bag — it's a window onto the inner storage / generator
   so users always see one source of truth.

8. **Remove the now-deprecated top-level names** (`gen.p`, `gen.p_t`,
   `bus.theta`, `bus.theta_t`, `line.p`, `line.p_t`, `link.p`, `link.p_t`,
   `storage.soc`, `storage.p_in`, `storage.p_out`, `storage.soc_t`,
   `storage.p_in_t`, `storage.p_out_t`, `battery.p`, `battery.soc`,
   `battery.p_t`, `battery.soc_t`, `battery.p_store`, `battery.p_dispatch`,
   `battery.p_store_t`, `battery.p_dispatch_t`, `phs.p`, `phs.soc`,
   `phs.p_t`, `phs.soc_t`, `phs.p_store`, `phs.p_dispatch`,
   `phs.p_store_t`, `phs.p_dispatch_t`).

   **Exception worth keeping**: `Component.get_p_net()` returns an expression
   used by the KCL builder. Keep it (it's a method, not an attribute, and it
   composes ad-hoc — `p_out - p_in` for storage, `-p_set` for load). Internally
   it now reads `self.var.p_in_t`, etc.

9. **Update `importers.py`** if it touches any renamed attributes (it
   shouldn't — it works on `Network` registry dicts, not on solution-time
   data).

10. **Run the test suite** at this point. Expected: lots of red — every test
    that reads `gen.p_t`, `gen.p`, `storage.soc`, etc. is broken until pass 2.
    That's fine; this is a refactor commit, not a feature commit.

### Pass 2: migrate tests, docs

1. **Update every test in `tests/`** with mechanical replacements. Recommended
   regex sweep (manually verify each hit):
   - `\.p\b` → `.var.p_t` (constraint construction context) or
     `.sol.p_t["p"]` (result-reading context). The two contexts are distinct;
     don't blanket-replace.
   - `\.p_t\b` → `.sol.p_t`
   - `\.soc\b` → `.var.soc_t` (constraint context) or `.sol.soc_t["soc"]`
     (result context).
   - `\.soc_t\b` → `.sol.soc_t`
   - `\.p_in\b`, `\.p_out\b`, `\.theta\b` → analogous.

   In practice the call sites are easy to recognise: constraint contexts pass
   the variable into a pyoframe constraint; result contexts call
   `.to_list()`, index `["p"]`, etc.

2. **Add focused tests for the new `_pu_t` views**, in
   `tests/test_pu_views.py`:
   - `gen.sol.p_pu_t` for a generator running at 30/100 returns 0.3.
   - `storage.sol.soc_pu_t` for SOC=50 in a 100 MWh storage returns 0.5.
   - `storage.var.soc_pu_t` is a usable pyoframe expression in a constraint
     (build `soc_pu_t <= 0.8` and verify the LP enforces it).
   - `_StorageBase` with both `p_nom_in=None` and `p_nom_out=None` raises
     a clear error on `var.p_pu_t` access.

3. **Update `tests/conftest.py`** if any fixture reads renamed attributes —
   most don't, but verify.

4. **Refresh `tests/TEST_GUIDELINES.md`** — replace the "Reading solutions"
   section with the new container shape, and update example snippets.

5. **Refresh `README.md` Quick Start** — `print(gen.sol.p_t)`,
   `gen.sol.p_pu_t`, etc.

6. **Run the full suite** — expect 78 + new pu tests passing.

## Verification

End-to-end on the existing fixtures:

```python
# Build path uses var.*
gen.var.p_t * gen.marginal_cost  # constraint / objective term

# Read path uses sol.*
n.optimize()
df = gen.sol.p_t          # DataFrame: time, p
cf = gen.sol.p_pu_t       # DataFrame: time, p_pu  (= p / p_nom)
```

The full pytest must report:
- The 78 existing tests still pass after both passes.
- The new `tests/test_pu_views.py` adds ~5 focused tests.

Run:
```bash
./conda/bin/python -m pytest -q
```

## Critical files

Source:
- `src/qwenaplan/base.py` — add `_VarContainer` / `_SolContainer`, hook up on
  `Component.__init__`. The existing `_solution_as` helper stays.
- `src/qwenaplan/components.py` — every component class. Largest churn here.
- `src/qwenaplan/physics.py` — KCL/KVL builders read `bus.var.theta_t`,
  `line.var.p_t`.
- `src/qwenaplan/network.py` — `create_model`'s component-iteration loop is
  unchanged, but the `_objective_cost_weight_param`/`_snapshot_*_param`
  helpers stay where they are (these are network-level, not component-level).

Tests / docs:
- All `tests/test_*.py` — mechanical rename pass.
- `tests/conftest.py` — verify nothing reads renamed attrs.
- New `tests/test_pu_views.py`.
- `tests/TEST_GUIDELINES.md`.
- `README.md`.

## Out of scope (do **not** mix into this commit)

- No new components.
- No solver tweaks.
- No new constraints.
- No physics changes.
- No optimization features.

If something tempts you mid-refactor — write it down for a follow-up plan.

## Estimated diff size

- src: ~250 line diff (mostly inside `components.py`).
- tests: ~150 line diff across 9 files (mechanical).
- docs: ~50 line diff.

Single session, clean execution. Should be doable in two commits (one src,
one tests + docs) so review can split the mechanical from the substantive.
