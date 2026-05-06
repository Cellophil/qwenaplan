# qwenaplan

A modern power system optimization framework built with Polars and PyOframe.

A deliberately scoped-down conceptual rewrite of [PyPSA](https://github.com/PyPSA/PyPSA): same problem space (DC linear OPF for planning studies), same component vocabulary, but every line is different and several PyPSA features are omitted on purpose. See [plans/00_roadmap.md](plans/00_roadmap.md) for the long-term direction.

## Installation

### Development

```bash
# A local conda env at ./conda is the convention.
conda install -p ./conda -c conda-forge python=3.12 polars highspy pytest pytest-cov dill pip
./conda/bin/pip install pyoframe
./conda/bin/pip install -e ".[dev]"
```

### Regular

```bash
pip install qwenaplan   # not yet on PyPI; install editable for now
```

## Quick start

```python
import polars as pl
import qwenaplan as qp

n = qp.Network()

# Buses
bus1 = n.add(qp.Bus, "Bus1", v_nom=1.0)
bus2 = n.add(qp.Bus, "Bus2", v_nom=1.0)

# Generation and demand
n.add(qp.Generator, "Cheap", bus=bus1, p_nom=100.0, marginal_cost=10.0)
n.add(qp.Generator, "Expensive", bus=bus2, p_nom=100.0, marginal_cost=100.0)
n.add(qp.Load, "Demand", bus=bus2, p_set=40.0)

# Transmission
line = n.add(qp.ACLine, "L", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=200.0)

# Time axis (with optional duration in hours and weighting in occurrences)
n.set_snapshots(pl.Series("time", [0, 1, 2, 3]), duration=1.0, weighting=1.0)

# Build and solve
n.create_model()        # vars, constraints, objective from marginal_cost × duration × weighting
status = n.optimize()   # returns pyoptinterface TerminationStatusCode

# Inspect
print(n.objective_value)            # total annualised cost
print(n.generators["Cheap"].p_t)    # tidy DataFrame keyed by snapshot
print(line.p_t)                      # line flow per snapshot (from_bus → to_bus)
```

### Components

- **`Bus`** — node; carries a phase-angle variable for KVL.
- **`Generator`** — supply with `p_nom`, `marginal_cost`, optional `p_min_pu` / `p_max_pu` (scalar or per-snapshot Series), optional `ramp_limit_up` / `ramp_limit_down`.
- **`Load`** — demand at a bus, `p_set` (scalar or per-snapshot Series). Parameter, not a decision variable.
- **`ACLine`** — DC linear power flow with reactance `x_pu` and thermal limit `s_nom`.
- **`Link`** — controllable inter-bus flow with `p_nom` and `efficiency`.
- **`StorageUnit`** — generic storage (state of charge, charge/discharge efficiency, optional influx).
- **`Battery`** — composite over a StorageUnit: charge-and-discharge with one shared `p_nom`.
- **`PumpedHydroStorage`** — composite of a reservoir + a generator coupled by `gen_efficiency`.

### Notes vs. PyPSA

- **No `bus.p_net` slack.** KCL is closed: generators inject, loads withdraw, storage does both. If you want unmet-demand behaviour, add a high-marginal-cost generator explicitly (e.g. `marginal_cost=10_000`).
- **`network.optimize()` returns the termination code**, and `network.objective_value` returns the solved cost. Don't reach into `n.model` unless you really mean to.
- **`snapshot_duration` and `snapshot_weighting`** are kept distinct: duration enters the SOC physics (and the objective), weighting only enters the objective. A "6-hour overnight block" uses `duration=6.0` so a 30 MW battery actually moves 180 MWh during it.

## Running tests

```bash
./conda/bin/python -m pytest
```

Tests assert numerical answers against analytical truth where possible. See [tests/TEST_GUIDELINES.md](tests/TEST_GUIDELINES.md) for the conventions used.

## Project layout

```
qwenaplan/
├── pyproject.toml
├── README.md                  # this file
├── plans/                     # long-term roadmap
│   └── 00_roadmap.md
├── src/
│   └── qwenaplan/
│       ├── __init__.py
│       ├── base.py            # Component / PowerElement / BranchElement ABCs
│       ├── components.py      # Bus / Generator / Load / ACLine / Link / Storage*
│       ├── network.py         # Network: registries, set_snapshots, create_model, optimize
│       ├── physics.py         # DC KCL / KVL builders
│       └── importers.py       # PyPSA → qwenaplan
└── tests/
    ├── conftest.py
    ├── TEST_GUIDELINES.md
    └── test_*.py
```

## Dependencies

- Python ≥ 3.10 (we use `float | None` syntax in a couple of places)
- `polars`
- `pyoframe` (≥ 1.4 — the solver kwarg is required)
- A solver — defaults to HiGHS via `highsbox` (installed transitively from `pyoptinterface[highs]`)

## License

MIT
