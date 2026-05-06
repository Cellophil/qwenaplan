# qwenaplan — Long-Term Roadmap

## Context

`qwenaplan` is a deliberate, scoped-down conceptual rewrite of [PyPSA](https://github.com/PyPSA/PyPSA) built on **Polars** + **pyoframe**. The current state is a clean, well-architected MVP: DC linear OPF with generators, AC lines, links, and three storage flavours (`StorageUnit`, `Battery`, `PumpedHydroStorage`), plus a strict-mode PyPSA importer. The code is internally consistent and the tests pass — but several foundational pieces are missing or quietly broken, and the surface area below the MVP is much smaller than what a planning workflow needs day-to-day.

This is a **long-term to-do list, not a concrete plan of action**. Each chunk is sized so a weaker model can pick it up in isolation without holding the whole architecture in its head. Chunks are deliberately ordered: **Tier 0 fixes pre-existing latent bugs**, then we add the missing primitives (Load, results), then I/O, statistics, capacity expansion. Multi-year investment is deferred to integrate with the separate Benders project.

Each tier has a one-line "why now" and lists candidate file names so they can be split into separate planning files later.

---

## Findings that motivate Tier 0 (latent bugs / gaps in current code)

While surveying, three issues stood out that any user will hit immediately:

1. **The objective is never assembled.** Every component implements `setup_objective(model)`, but `Network.create_model()` calls `setup_variables_for_model` and `setup_constraints` only — `setup_objective` is dead code. Today, tests manually write `n.model.minimize = (gen.p * gen.marginal_cost).sum()`. Anyone using the public API as documented gets an unconstrained-cost LP. See [network.py:95-122](src/qwenaplan/network.py#L95-L122) vs. [components.py:252-256](src/qwenaplan/components.py#L252-L256).
2. **There is no `Load` component, and `bus.p_net` is a free variable acting as a hidden slack.** The PyPSA importer drops Load components with a comment claiming they are "handled via bus p_net", but nothing actually injects load. KCL therefore solves with zero demand by default. See [importers.py:8](src/qwenaplan/importers.py#L8) and [physics.py:11-39](src/qwenaplan/physics.py#L11-L39). **Decision:** remove `p_net` entirely; demand is a `Load`; if the user wants a slack, they add a high-marginal-cost generator explicitly.
3. **There is no `optimize()` method and no exporter.** Users must reach into `n.model` directly and there is no way to round-trip a network back out. The importer is one-way only.

---

## Tier 0 — Foundation fixes (do these first)

> *Why now:* without these, the documented workflow doesn't actually produce a meaningful optimization. Each item is a single-file or near-single-file change.

- **`plan_01_objective_assembly.md`** — Wire `setup_objective` into `Network.create_model()`. Sum each component's contribution into `model.minimize`. Decide the units convention now (per-MWh × snapshot duration; see plan_05). Files: [network.py](src/qwenaplan/network.py), [components.py](src/qwenaplan/components.py).
- **`plan_02_load_component_remove_pnet.md`** — Introduce `Load` (`name`, `bus`, `p_set` as scalar or `pl.Series`). **Remove `bus.p_net` and `bus.theta` slack semantics**: `theta` stays (KVL needs it) but there is no longer any free injection variable on a bus. KCL becomes `Σ injections − Σ withdrawals = 0` with no slack. Users wanting an unmet-load slack add a high-marginal-cost generator (e.g. €10 000/MWh) explicitly — this is on the user, by design. Update PyPSA importer to actually import Loads (including time-varying `p_set`). Possibly derive `Load` from a thin internal base shared with `Generator` (sign-flipped) — judge once both are in front of you. Files: [components.py](src/qwenaplan/components.py), [network.py](src/qwenaplan/network.py), [physics.py](src/qwenaplan/physics.py), [importers.py](src/qwenaplan/importers.py), [__init__.py](src/qwenaplan/__init__.py), new test.
- **`plan_03_network_optimize.md`** — Add `Network.optimize(solver=...)` that calls `model.optimize()` and surfaces termination status. Add `Network.objective_value` property. Currently every test does this by hand. File: [network.py](src/qwenaplan/network.py).
- **`plan_04_results_accessors.md`** — Add `.result` / tidy accessors on each component (`gen.p_t`, `storage.soc_t`, `line.p_t`) returning Polars frames. Centralise the `variable.solution["solution"].to_list()` pattern that's currently duplicated across tests. Files: [base.py](src/qwenaplan/base.py), [components.py](src/qwenaplan/components.py).
- **`plan_05_snapshot_duration_and_weighting.md`** — Add **two** orthogonal time concepts:
  - `snapshot_duration` (hours): physical length of each snapshot. Storage SOC balance becomes `soc(t+1) = soc(t) + (p_in × eff_in − p_out / eff_out) × duration(t) + influx × duration(t)`. This is what makes "this snapshot represents a 6 h overnight block" work physically (a battery with 30 MW charge power genuinely puts 180 MWh into the reservoir).
  - `snapshot_weighting` (dimensionless): how many *occurrences* of this snapshot we represent (e.g. 365 for a "typical day" in a 365-day year). Multiplies cost contributions in the objective.
  - Default both to 1.0 per snapshot for backward compatibility. PyPSA conflates these; we should keep them distinct because it makes representative-period modelling cleaner. Files: [network.py](src/qwenaplan/network.py), [components.py](src/qwenaplan/components.py) (objective + storage SOC).

---

## Tier 1 — I/O round-trip & ergonomics

> *Why now:* without an exporter, qwenaplan is write-only. There are real choices to make here — see the export design note below before splitting into chunks.

### Export design — open questions to resolve in `plan_10_export_design.md`

This deserves its own design doc before any code; the choices are entangled:

1. **What gets saved?**
   - (a) Just the *topology + parameters* (re-buildable network, no constraints, no model).
   - (b) The whole pyoframe model (constraints, custom expressions, partial fixings like "this generator has fixed maintenance in 2026").
   - (c) Both, separately: a "data" layer and an "extras" layer.

   You raised the case of preserving custom in-place constraints (e.g. fixed maintenance schedules). That nudges us toward (b)/(c). But once a `pf.Model` exists, picklability is up to pyoframe — needs verification before committing.

2. **How is it serialised?**
   - **xlsx (one workbook, one sheet per component type)** — your preference. Pros: human-readable, opens in Excel, multi-table-per-file (no folder bloat), good for stakeholder hand-off. Cons: slow to write at large network size (8760 × 9k buses × time-series columns gets painful), no streaming, dependency on `xlsxwriter` / `openpyxl`.
   - **CSV folder (PyPSA-style)** — Pros: fast, streams, easy to diff in git, Polars-native. Cons: file bloat, fragile if user moves files around.
   - **Pickle (dill) the whole network object** — Pros: round-trips *everything* including custom constraints, in-progress model state. Cons: opaque, brittle across versions, won't survive a refactor of internal classes.

   **My pushback on "no benefit in csv":** there is one — diffability and streaming for large networks. But for the scale you're targeting and the stakeholder-handoff use case, **xlsx is the right primary format**. I'd recommend **xlsx as the headline exporter, plus dill-pickle for "save the full live network including constraints"**, and skip CSV unless someone asks. That keeps the format count to two with clear, non-overlapping use cases.

3. **When does export happen?** Pre-`create_model()` (data-only) or post (data + model)? Probably both must be supported, since the live-constraints use case needs post-build export.

### Concrete chunks once design is settled

- **`plan_10_export_design.md`** — The doc above. Pick: xlsx primary, dill secondary, csv deferred. Verify pyoframe model picklability.
- **`plan_11_xlsx_exporter.md`** — `network.to_xlsx(path)`: one sheet per component type plus `snapshots` and `meta` sheets. Time-series go in dedicated sheets keyed by component name + snapshot index. Use `polars.write_excel`.
- **`plan_12_xlsx_importer.md`** — Round-trip companion. Reuse validation logic from `PyPSAImporter`. Ideally extract the per-component validation into shared helpers so PyPSA, xlsx, and any future importer share it.
- **`plan_13_pickle_save_load.md`** — `network.save(path)` / `Network.load(path)` using `dill`. Document what gets preserved (custom constraints, partial fixings) and what doesn't (the solver instance). Add a version-stamp and refuse cross-version loads with a clear error.
- **`plan_14_network_summary.md`** — Better `Network.__repr__`, plus `network.describe()` returning a Polars DataFrame: counts, total capacity by carrier, total demand, snapshot range, total horizon (sum of `snapshot_duration × snapshot_weighting`).
- **`plan_15_logging.md`** — Replace `print()` in [network.py:100,122](src/qwenaplan/network.py#L100) with the standard `logging` module; opt-in verbosity flag.
- **`plan_16_consistency_checks.md`** — `network.consistency_check()` before `create_model()`: orphan buses, disconnected sub-networks (warn), generators without `p_nom`, snapshots not monotonic, storage `initial_soc` outside bounds, loads at non-existent buses.

---

## Tier 2 — Statistics: per-unit, collected centrally

> *Why now:* once results exist (Tier 0/4), users immediately want capacity factors, curtailment, cost breakdowns. Designed object-oriented so each component owns its own stats logic.

**Architectural pattern:** each component class gains a thin `stats()` method (or set of methods) returning a per-unit Polars row. A central `network.statistics.<aggregator>` collects rows across components. Implementation lives in a separate file (`src/qwenaplan/statistics/` package) — either as **mixins** that the component classes pick up, or as **registered free functions keyed by component class**. Mixin keeps the call site object-oriented (`gen.capacity_factor()`); free-functions keep `components.py` lean. **Recommendation:** mixin per component type, mixin file co-located in the statistics package, applied in `__init__.py`. This keeps `components.py` focused on physics/optimization while `gen.capacity_factor()` reads naturally to the user.

- **`plan_20_statistics_package_skeleton.md`** — New `src/qwenaplan/statistics/` package. `__init__.py` applies mixins. Define the one-row-per-unit return shape (`name`, `carrier`, `value`, plus stat-specific columns). Files: new package.
- **`plan_21_per_unit_capacity_factor.md`** — `Generator.capacity_factor()`, `StorageUnit.capacity_factor()`, etc. Use `snapshot_duration × snapshot_weighting` for honest annualisation.
- **`plan_22_per_unit_curtailment.md`** — `Generator.curtailment()`: compares `p_t` against `p_max_pu × p_nom × duration × weighting`. Carrier-aware so you can later sum by carrier.
- **`plan_23_per_unit_costs.md`** — `Generator.opex()`, `StorageUnit.opex()` etc.: marginal-cost contribution to objective, time-weighted. Sets up the per-unit hooks that Tier 5's `capex` will plug into.
- **`plan_24_aggregator.md`** — `network.statistics.energy_balance()`, `.system_cost_breakdown(by="carrier")`, `.summary()`. Just iterates components, calls per-unit stats, concatenates Polars frames. Slicing by carrier becomes one-liners.
- **`plan_25_dual_values_lmp.md`** — Expose duals of KCL constraints as **nodal prices / LMPs** on `Bus` (`bus.marginal_price_t`). pyoframe supports duals; tidy accessor is enough.

---

## Tier 3 — Carrier as a real component (slicing)

> *Why now:* `carrier` is currently a metadata string. Promoting it lets every statistics aggregation slice "by carrier" trivially, and is the natural anchor for emissions/CO₂ accounting later, *without* the full PyPSA `GlobalConstraint` machinery (which we are deliberately not replicating).

- **`plan_30_carrier_component.md`** — Add `Carrier` (`name`, `nice_name`, `colour`, `co2_emissions` g/kWh, others as needed). Backwards-compatible: any string `carrier` parameter on a component auto-creates a `Carrier(name=string)` with defaults if not already registered. `Network.carriers` dict like the others.
- **`plan_31_carrier_slicing_helpers.md`** — `network.generators_by_carrier("solar")`, `network.statistics.energy_balance(by="carrier")`. No formal `GlobalConstraint` API — if the user wants a CO₂ cap, they write a one-line custom constraint summing over `network.generators_by_carrier(...)`.
- **`plan_32_custom_constraints_api.md`** — Document and stabilise the pyoframe escape hatch: `network.add_custom_constraint(name, expr)` so users can express bespoke constraints (CO₂ caps, reserve margins, must-run blocks) without subclassing. **This replaces the global-constraints replication idea entirely.**

---

## Tier 4 — Capacity expansion planning (single-period CEP)

> *Why now:* this is the feature that makes a tool a "planning tool" rather than a "dispatch tool". The PyPSA importer already detects and rejects `p_nom_extendable`, etc. — clear hook for adding it.

- **`plan_40_extendable_generators.md`** — `p_nom_extendable: bool`, `p_nom_min`, `p_nom_max`, `capital_cost`. Replace fixed `p_nom` with a pyoframe variable; add `capital_cost × p_nom_var` to objective via `setup_objective`.
- **`plan_41_extendable_lines.md`** — Same for `ACLine.s_nom_extendable` and `Link.p_nom_extendable`. KVL stays linear because impedance is fixed and only thermal capacity grows (PyPSA convention).
- **`plan_42_extendable_storage.md`** — Same for `StorageUnit`/`Battery`/`PumpedHydroStorage`: extendable `e_nom`, `p_nom_in`, `p_nom_out`. PHS has both energy and power capacities — typically a fixed turbine-to-reservoir ratio.
- **`plan_43_pypsa_importer_cep.md`** — Once 4.0–4.2 land, drop the corresponding entries from `INVESTMENT_ATTRS` in [importers.py](src/qwenaplan/importers.py#L48-L66) and import them properly.
- **`plan_44_capex_in_statistics.md`** — Plug capex into the per-unit stats from Tier 2 (`Generator.capex()`).

---

## Tier 5 — Unit commitment (committable, MILP)

> *Why now:* you flagged this explicitly — needed for some studies even though we won't go all the way to min up/down time.

- **`plan_50_committable_generators.md`** — Add `committable: bool` and a binary `status_t` variable per generator. Constraints: `p_min_pu × p_nom × status ≤ p ≤ p_max_pu × p_nom × status`. Optional `start_up_cost`, `shut_down_cost`. **Do not add `min_up_time` / `min_down_time`** — too expensive, too little benefit (per your call).
- **`plan_51_committable_storage_dispatch.md`** — Optional binary "is discharging" flag on storage to forbid simultaneous charge+discharge for cases where it sneaks in. Mark as opt-in: most LPs don't need it.
- **`plan_52_milp_solver_handling.md`** — Make sure the solver path handles MILP termination statuses properly in `Network.optimize()`. Document the perf cliff.
- **`plan_53_uc_tests.md`** — Numerical tests: a generator with `p_min_pu=0.4, committable=True` must either be off or ≥40% of `p_nom`.

---

## Tier 6 — Multi-year investment (DEFERRED, integrate with Benders project)

> *Why now:* not now. Listed only so future-you knows we considered it. Aggregate with the separate Benders decomposition project later — it's the natural decomposition: master problem chooses investments per year, sub-problems are per-year qwenaplan dispatch problems. The decomposition lives outside qwenaplan; qwenaplan just needs to be parameter-callable as a single-year solver.

- **`plan_60_investment_period_loop_hooks.md`** *(future)* — Make sure `Network.set_snapshots()`, capex calculation, and result extraction are clean enough that an external Benders driver can re-instantiate / parameterise networks per period without monkey-patching internals.

---

## Tier 7 — Test suite review

> *Why now:* current tests are mostly smoke tests that verify variables exist, not that numerical answers are correct. Each tier above is safer with these in place. Can run in parallel with most of tiers 1–5.

- **`plan_70_test_audit.md`** — Read every test, mark each as: smoke / numeric / redundant / actually-tests-something. Output a kill-list.
- **`plan_71_numerical_dispatch_tests.md`** — Replace smoke tests with analytical-solution checks: "if cheap gen costs 10 and expensive 50, expensive must be at zero until cheap is at p_nom". Parametrize over scenarios.
- **`plan_72_infeasibility_tests.md`** — Demand > capacity, ramping conflicts, soc_min > initial_soc; assert correct termination status and surface clear error messages.
- **`plan_73_io_roundtrip_tests.md`** — Build → export (xlsx) → import → re-build → solve → identical objective.
- **`plan_74_pypsa_importer_tests.md`** — Synthetic small PyPSA networks fixture; importer matrix (strict vs. lenient) for every supported attribute.
- **`plan_75_large_network_perf.md`** — A 200-bus / 8760-snapshot stress test marked `@pytest.mark.slow`.

---

## Tier 8 — Power-flow & topology utilities (lower priority)

- **`plan_80_subnetwork_detection.md`** — `network.determine_topology()` returns connected components; flag islanded buses; `bus.sub_network` attribute.
- **`plan_81_lossy_lines.md`** — Optional piecewise-linear loss model on `ACLine`. Keep DC default; behind a flag.

---

## Explicitly **out of scope** (what we will NOT add)

Confirming in writing so the project stays focused:

- **Full nonlinear AC OPF / Newton-Raphson power flow** — needs an NLP solver, wrong abstraction for planning.
- **Plotting / GIS visualisation** — defer to a later iteration; no work now. Users plot Polars frames with their tool of choice in the meantime.
- **`min_up_time` / `min_down_time` / linearised UC** — too expensive, too little benefit. We do `committable` (Tier 5) but stop there.
- **PyPSA `GlobalConstraint` replication** — replaced by Carrier slicing + custom constraint API (Tier 3).
- **Frequency / inertia / ROCOF stability** — AC-domain, separate tooling.
- **HDF5 I/O** — PyPSA itself deprecated it.
- **PyPSA-Eur build pipeline** — qwenaplan should *consume* a PyPSA-Eur network via the importer; replicating the data pipeline is way out of scope.
- **CSV export** — xlsx + pickle cover the use cases; revisit only if a concrete user need appears.
- **N-1 / SCLOPF** — not in scope for now.
- **Legacy pyomo / linopy backend support** — pyoframe is the chosen abstraction.

---

## Suggested execution order (for a weaker model)

1. **Tier 0 in order** — small, well-bounded, unblocks everything.
2. **Tier 7 plans `plan_70` and `plan_71`** early so subsequent tiers have a numerical-truth safety net.
3. **Tier 1** I/O design doc (`plan_10`) before any other tier-1 code.
4. **Tier 2** (statistics) — small, parallelisable.
5. **Tier 3** (carrier + custom constraints).
6. **Tier 4** (CEP) — biggest single jump in value.
7. **Tier 5** (committable / MILP).
8. **Tier 8** opportunistic.
9. **Tier 6** only when the Benders project is ready to integrate.

---

## Verification (per chunk)

Each individual chunk plan should include:
- Unit tests against a known-correct numerical answer (not just smoke tests).
- A short snippet demonstrating the new feature end-to-end against the `test_network` / `multi_bus_network` fixtures in [conftest.py](tests/conftest.py).
- Update of [README.md](README.md) Quick Start where the public API changes (e.g. once `Load`, `Network.optimize()`, `snapshot_duration` exist).

For the whole roadmap: a smoke test that imports a small reference PyPSA network, exports it to xlsx, re-imports it, solves it, and asserts the objective matches PyPSA within 1e-6.
