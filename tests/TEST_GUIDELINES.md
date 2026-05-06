# qwenaplan test guidelines

These are the conventions the suite uses. Keep them when adding tests so
the file remains scannable.

## Workflow under test

```python
n = qp.Network()
# ... add(Bus / Generator / Load / Line / Link / Storage) ...
n.set_snapshots(snapshots, duration=1.0, weighting=1.0)
n.create_model()           # builds variables, constraints, objective
status = n.optimize()      # returns pyoptinterface TerminationStatusCode
assert status == poi.TerminationStatusCode.OPTIMAL
gen.p_t["p"].to_list()     # read solution
n.objective_value           # total cost
```

Three things that used to be the norm and are now wrong:

1. **Do not write `n.model.minimize = …` after `create_model()`**. Pyoframe
   forbids reassigning `minimize`, and `create_model` already builds the
   objective from each component's `setup_objective` contribution
   (Generator marginal-cost × duration × weighting, etc.). If a test needs
   an extra term, use `+=` / `-=`. If it needs a fundamentally different
   objective, build it before `create_model` is called and arrange the
   network so no other component contributes.
2. **Do not use `bus.p_net`** — there is no such variable. Demand is a
   real `Load` component now. Add `qp.Load(name, bus, p_set)`. If you
   want load shedding, add a high-marginal-cost generator (e.g.
   `marginal_cost=10_000`) and let the LP trade shed cost against
   generation cost.
3. **Do not hand-roll `gen.p + battery.p >= load_param`** as a balance
   constraint. KCL is automatic; just add `Load` and the bus balance
   closes naturally.

## Fixtures (in `conftest.py`)

Each fixture returns a network with components added but `set_snapshots`
not yet called, so a test can still mutate parameters or add more
components.

| Fixture | Topology |
|---|---|
| `network` | empty |
| `snapshots` | 4-step `pl.Series("time", [0,1,2,3])` |
| `two_bus_network` | Bus1—ACLine—Bus2; one gen at Bus1; no load |
| `three_bus_network` | three buses, two AC lines, one Link |
| `storage_test_network` | one bus + gen + StorageUnit + Load |
| `battery_test_network` | one bus + gen + Battery + Load |
| `phs_test_network` | one bus + gen + PumpedHydroStorage + Load |

## Test structure

Each component's tests live in `test_<component>.py` and follow this
shape:

```python
class TestComponentInit:
    def test_default_parameters(self, network): ...
    def test_custom_parameters(self, network): ...
    def test_repr(self, network): ...

class TestComponent<Behaviour>:
    def test_<analytical truth>(self, snapshots): ...
```

What we keep:
- **Init tests** that verify constructor parameters land where expected.
- **Validation tests** for inputs that should raise.
- **Numerical behaviour tests** that build a small LP whose optimal
  dispatch you can compute on paper, then assert the LP returns it.

What we don't keep:
- Variable/constraint *existence* checks. `hasattr(model, "gen_lower_X")`
  tests internal naming, not behaviour, and locks us in to internal
  details. Replace with a numerical assertion that proves the constraint
  is doing its job.
- Hand-rolled balance constraints. Add a `Load`.
- "Smoke tests that just verify model runs without error." Use a
  numerical assertion or an infeasibility assertion (see
  `tests/test_infeasibility.py`).

## Reading solutions

Every component exposes `<var>_t` properties that return Polars frames
keyed by snapshot:

```python
gen.p_t           # cols: time, p
load.p_t          # cols: time, p (parameter, available pre-solve too)
line.p_t          # cols: time, p
storage.soc_t     # cols: time, soc
storage.p_in_t    # cols: time, p_in
battery.p_t       # cols: time, p   (computed: p_dispatch - p_store)
phs.p_t           # cols: time, p   (= generator.p)
phs.p_dispatch_t  # cols: time, p_dispatch
```

Use these instead of `gen.p.solution["solution"].to_list()` — same data,
clearer intent.

## Termination status

```python
import pyoptinterface as poi
assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
```

For infeasibility tests, accept any non-OPTIMAL status — HiGHS reports
different codes depending on presolve. The contract is "the LP did not
silently report OPTIMAL."

## Snapshot duration & weighting

`n.set_snapshots(snapshots, duration=…, weighting=…)`:
- `duration` (hours per snapshot) multiplies storage SOC dynamics
  (energy = power × duration) **and** the objective.
- `weighting` (occurrence count) multiplies the objective only — handy
  for representative-period models (e.g. one typical day weighted 365).
- Defaults are 1.0 / 1.0; the per-snapshot semantics from before remain
  valid.

A test that wants to check duration scaling should use a 6-hour
snapshot and assert the SOC delta is 6× the power, not 1×.
