import polars as pl
import pyoframe as pf
from typing import Dict, List, Any
from .components import (
    Bus,
    Generator,
    Load,
    ACLine,
    Link,
    StorageUnit,
    PumpedHydroStorage,
    Battery,
)
from .views import View, _build_bus_view


# Registry-attribute -> default view name. Keep aligned with
# ``_COMPONENT_REGISTRY`` below; the auto-populator iterates this list
# to mint a view per registry. Excludes ``buses`` (each bus gets its
# own view, keyed by bus name) so we don't shadow user expectations.
_DEFAULT_REGISTRY_VIEWS: list[tuple[str, str]] = [
    ("generators", "generators"),
    ("loads", "loads"),
    ("storage_units", "storage_units"),
    ("batteries", "batteries"),
    ("pumped_hydro", "pumped_hydro"),
    ("lines", "lines"),
    ("links", "links"),
]


class Network:
    def __init__(self):
        self.buses: Dict[str, Bus] = {}
        self.lines: Dict[str, ACLine] = {}
        self.links: Dict[str, Link] = {}
        self.generators: Dict[str, Generator] = {}
        self.loads: Dict[str, Load] = {}
        self.storage_units: Dict[str, StorageUnit] = {}
        self.pumped_hydro: Dict[str, PumpedHydroStorage] = {}
        self.batteries: Dict[str, Battery] = {}

        # Named subsets of components with aggregated var/sol surfaces.
        # Auto-populated at ``create_model()`` (one entry per registry +
        # one per bus); users may also assign their own.
        self.views: Dict[str, View] = {}

        self.snapshots = None
        # Per-snapshot timing (filled by set_snapshots/set_snapshot_durations):
        # - snapshot_duration[t]: physical length of snapshot t (hours).
        #   Multiplies storage SOC dynamics (energy = power × duration) and
        #   the objective (cost = price × power × duration).
        # - snapshot_weighting[t]: how many physical occurrences of t the
        #   model represents (e.g. 365 for a "typical day" in a yearly
        #   model). Multiplies the objective only — SOC dynamics see the
        #   single physical occurrence.
        self.snapshot_duration: pl.Series = None
        self.snapshot_weighting: pl.Series = None
        self.model = None
        self._is_locked = False

    # Class -> registry-dict-name. Used by both add() (class or string-name
    # accepted) and the rest of the network for iteration.
    _COMPONENT_REGISTRY = {
        Bus: "buses",
        ACLine: "lines",
        Link: "links",
        Generator: "generators",
        Load: "loads",
        StorageUnit: "storage_units",
        PumpedHydroStorage: "pumped_hydro",
        Battery: "batteries",
    }

    def add(self, cls, name: str, **kwargs):
        """Create a component and register it on the network.

        ``cls`` may be a component class (``Generator``) or its string name
        (``"Generator"``) — the latter mirrors PyPSA's API and is what the
        PyPSA importer uses.
        """
        if self._is_locked:
            raise RuntimeError("Network is locked after create_model().")

        # Accept string class names (e.g. from the PyPSA importer).
        if isinstance(cls, str):
            by_name = {c.__name__: c for c in self._COMPONENT_REGISTRY}
            if cls not in by_name:
                raise ValueError(f"Unsupported component class name {cls!r}")
            cls = by_name[cls]

        if cls not in self._COMPONENT_REGISTRY:
            raise ValueError(f"Unsupported component class {cls}")

        registry = getattr(self, self._COMPONENT_REGISTRY[cls])
        obj = cls(name, self, **kwargs)
        registry[name] = obj
        return obj

    def set_snapshots(
        self,
        snapshots: pl.Series,
        duration: float | pl.Series = 1.0,
        weighting: float | pl.Series = 1.0,
    ):
        """Define the time axis and trigger variable creation.

        Parameters
        ----------
        snapshots : pl.Series
            Snapshot index. The series ``.name`` becomes the time-dimension
            name used throughout pyoframe expressions.
        duration : float | pl.Series, default 1.0
            Hours represented by each snapshot. Scalar broadcasts. Used by
            storage SOC dynamics (energy gained = power × duration) and the
            objective (cost = marginal_cost × p × duration × weighting).
        weighting : float | pl.Series, default 1.0
            Occurrence multiplier (e.g. 365 for a representative day in a
            yearly model). Multiplies objective contributions only, never
            the SOC equation.
        """
        if self._is_locked:
            raise RuntimeError("Network is locked.")

        self.snapshots = snapshots
        n = len(snapshots)
        self.snapshot_duration = (
            duration if isinstance(duration, pl.Series)
            else pl.Series("snapshot_duration", [float(duration)] * n)
        )
        self.snapshot_weighting = (
            weighting if isinstance(weighting, pl.Series)
            else pl.Series("snapshot_weighting", [float(weighting)] * n)
        )

        # Trigger all components to initialize their pyoframe variables
        all_components = {
            **self.buses,
            **self.lines,
            **self.links,
            **self.generators,
            **self.loads,
            **self.storage_units,
            **self.pumped_hydro,
            **self.batteries,
        }.values()
        for comp in all_components:
            comp.setup_variables()

    def _snapshot_duration_param(self) -> pf.Param:
        """Build a pyoframe Param of per-snapshot durations (hours)."""
        df = self.snapshots.to_frame().with_columns(
            self.snapshot_duration.alias("duration")
        )
        return pf.Param(df)

    def _snapshot_weighting_param(self) -> pf.Param:
        """Build a pyoframe Param of per-snapshot occurrence counts."""
        df = self.snapshots.to_frame().with_columns(
            self.snapshot_weighting.alias("weighting")
        )
        return pf.Param(df)

    def _objective_cost_weight_param(self) -> pf.Param:
        """Build a Param of duration × weighting per snapshot.

        Components that contribute to the objective (e.g. Generator marginal
        cost) multiply their per-snapshot expression by this so a single
        ``.sum()`` at the network level produces honest annualised cost.
        """
        df = self.snapshots.to_frame().with_columns(
            (self.snapshot_duration * self.snapshot_weighting).alias("cost_weight")
        )
        return pf.Param(df)

    def get_connected_power_elements(self, bus: "Bus"):
        """Returns all generators/loads/storage connected to the given bus."""
        elements = []
        # Generators
        elements.extend([gen for gen in self.generators.values() if gen.bus == bus])
        # Loads
        elements.extend([ld for ld in self.loads.values() if ld.bus == bus])
        # Storage units
        elements.extend([su for su in self.storage_units.values() if su.bus == bus])
        # Pumped hydro
        elements.extend([phs for phs in self.pumped_hydro.values() if phs.bus == bus])
        # Batteries
        elements.extend([bat for bat in self.batteries.values() if bat.bus == bus])
        return elements

    def get_connected_lines(self, bus: "Bus"):
        """Returns all lines (ACLine or Link) connected to the given bus."""
        connected = []
        # Check AC Lines
        for line in self.lines.values():
            if line.from_bus == bus or line.to_bus == bus:
                connected.append(line)
        # Check Links
        for link in self.links.values():
            if link.from_bus == bus or link.to_bus == bus:
                connected.append(link)
        return connected

    def create_model(self, solver: str = "highs"):
        """Finalize topology, generate constraints, and build the objective.

        Parameters
        ----------
        solver : str, default "highs"
            Solver name passed to ``pyoframe.Model``. Pyoframe requires this
            explicitly; we default to HiGHS (free, bundled via highsbox).

        Notes
        -----
        After this call:
        - ``self.model`` is a ``pyoframe.Model`` with all variables and constraints.
        - ``self.model.minimize`` is the sum of every component's
          ``setup_objective`` contribution (e.g. generator marginal costs).
          Pyoframe forbids reassigning ``minimize``; use ``+=`` / ``-=`` to add
          extra terms (custom constraints can still be attached freely).
        - The network is locked: no further components may be added.
        """
        if self.snapshots is None:
            raise RuntimeError("Must call set_snapshots() before create_model().")

        print("Building optimization model...")
        # 1. Initialize pyoframe model
        self.model = pf.Model(solver=solver)

        # 2. Re-trigger variable setup to add them to the model
        all_components = list({
            **self.buses,
            **self.lines,
            **self.links,
            **self.generators,
            **self.storage_units,
            **self.pumped_hydro,
            **self.batteries,
        }.values())
        for comp in all_components:
            comp.setup_variables_for_model(self.model)

        # 3. Add constraints to the model
        for comp in all_components:
            comp.setup_constraints(self.model)

        # 4. Assemble the objective by summing per-component contributions.
        #    Components that contribute (e.g. Generator) call
        #    self._add_to_objective(...) inside setup_objective; non-contributing
        #    components do nothing. We finalize at the end.
        self._objective_terms: List[Any] = []
        for comp in all_components:
            comp.setup_objective(self)
        if self._objective_terms:
            total = self._objective_terms[0]
            for term in self._objective_terms[1:]:
                total = total + term
            self.model.minimize = total.sum()
        # If no component contributes, leave model.minimize unset; pyoframe
        # treats that as a feasibility problem.

        self._is_locked = True
        # Populate default views now that the registries are stable. We
        # do this after locking so a view's component list is guaranteed
        # not to grow under it; users adding custom views afterwards is
        # fine (they overwrite or add new keys on the plain dict).
        self._populate_default_views()
        print("Model created and network locked.")

    def _populate_default_views(self):
        """Build the auto-populated ``self.views`` entries.

        One view per non-empty registry (``"generators"``, ``"loads"``,
        ``"lines"``, etc.) plus one per bus (keyed by bus name, with the
        bus-injection sign convention). Empty registries get an empty
        view so ``n.views["loads"]`` is always present — the sol layer
        returns a snapshot-indexed zero column for empty cases.

        Bus / registry-name collisions raise loudly. None of the
        registry names (``"generators"`` etc.) are valid bus names in
        practice, but the explicit check is cheap and the alternative
        is silent shadowing of the KCL identity.
        """
        # 1. Registry views. Pass ``network=self`` so empty registries
        # (e.g. a network with no batteries) still have a network handle
        # for the sol layer to read the snapshot dim from.
        for registry_attr, view_name in _DEFAULT_REGISTRY_VIEWS:
            members = list(getattr(self, registry_attr).values())
            self.views[view_name] = View(view_name, members, network=self)

        # 2. Bus views.
        registry_names = {name for _, name in _DEFAULT_REGISTRY_VIEWS}
        for bus_name, bus in self.buses.items():
            if bus_name in registry_names:
                raise ValueError(
                    f"Bus name {bus_name!r} collides with a default registry "
                    f"view of the same name. Rename the bus to avoid shadowing "
                    f"the n.views[{bus_name!r}] aggregation."
                )
            self.views[bus_name] = _build_bus_view(self, bus)

    def _add_to_objective(self, expr: Any):
        """Components call this from their ``setup_objective`` to contribute.

        Receives a pyoframe expression indexed by snapshots (we ``.sum()``
        once at the end so each contribution does not need to know about
        snapshot weighting / duration — that lives on the network).
        """
        self._objective_terms.append(expr)

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def optimize(self):
        """Solve the model. Must be called after ``create_model()``.

        Returns the pyoframe termination status. Raises if the model has not
        been built yet so users get a clear error instead of an AttributeError.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call create_model() before optimize()."
            )
        self.model.optimize()
        return self.model.attr.TerminationStatus

    @property
    def objective_value(self) -> float:
        """Solved objective value.

        Returns ``None`` if the model hasn't been built. Returns ``0.0`` if
        the model has no objective at all (no component contributed) — that
        case is degenerate but valid (pure feasibility problem).
        """
        if self.model is None:
            return None
        try:
            return self.model.objective.value
        except ValueError:
            # pyoframe raises ValueError when no objective is set.
            return 0.0

    def __repr__(self):
        return (
            f"<Network(Buses={len(self.buses)}, Lines={len(self.lines)}, "
            f"Gens={len(self.generators)}, Loads={len(self.loads)}, "
            f"Storage={len(self.storage_units)}, PHS={len(self.pumped_hydro)}, "
            f"Batteries={len(self.batteries)})>"
        )
