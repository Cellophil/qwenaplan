# qwenaplan test guidelines

These are the conventions the suite uses. Keep them when adding tests so
the file remains scannable.

## Workflow under test

```python
n = qp.Network()
# ... add(Bus / Generator / Load / Line / Link / Storage) ...
n.set_snapshots(snapshots, duration=1.0, weighting=1.0)
n.create_model()                  # builds variables, constraints, objective
status = n.optimize()             # returns pyoptinterface TerminationStatusCode
assert status == poi.TerminationStatusCode.OPTIMAL
gen.sol.p_t["p"].to_list()        # read solution (sol = solved values)
gen.var.p_t * gen.marginal_cost   # build a custom constraint expression (var = symbolic)
n.objective_value                  # total cost
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

Solved values live on `obj.sol.<name>_t` and return Polars frames keyed
by snapshot:

```python
gen.sol.p_t           # cols: time, p
gen.sol.p_pu_t        # cols: time, p_pu      (= p / p_nom)
load.sol.p_t          # cols: time, p         (parameter, available pre-solve too)
line.sol.p_t          # cols: time, p
storage.sol.soc_t     # cols: time, soc
storage.sol.soc_pu_t  # cols: time, soc_pu    (= soc / e_nom)
storage.sol.p_in_t    # cols: time, p_in
storage.sol.p_pu_t    # cols: time, p_pu      (net = (p_out - p_in) / nameplate)
battery.sol.p_t       # cols: time, p         (computed: p_dispatch - p_store)
phs.sol.p_t           # cols: time, p         (= generator.var.p_t)
phs.sol.p_dispatch_t  # cols: time, p_dispatch
```

Symbolic counterparts (use these for custom constraints / objective
terms — no new variables introduced):

```python
gen.var.p_t            # pyoframe Variable
gen.var.p_pu_t         # pyoframe expression
storage.var.soc_pu_t   # = soc_t / e_nom (expression)
storage.var.p_pu_t     # = (p_out_t - p_in_t) / nameplate (expression)
n.model.cap = storage.var.soc_pu_t <= 0.8     # caps SOC at 80% of e_nom
```

Use these instead of `gen.var.p_t.solution["solution"].to_list()` —
same data, clearer intent. `Load` has no `var` (parameter, no decision
variable); only `load.sol.p_t`.

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
