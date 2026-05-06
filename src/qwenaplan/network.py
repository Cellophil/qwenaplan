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

        self.snapshots = None
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

    def set_snapshots(self, snapshots: pl.Series):
        """Define the time axis and trigger variable creation."""
        if self._is_locked:
            raise RuntimeError("Network is locked.")

        self.snapshots = snapshots
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
        print("Model created and network locked.")

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
        """Solved objective value, or ``None`` if not solved yet."""
        if self.model is None:
            return None
        return self.model.objective.value

    def __repr__(self):
        return (
            f"<Network(Buses={len(self.buses)}, Lines={len(self.lines)}, "
            f"Gens={len(self.generators)}, Loads={len(self.loads)}, "
            f"Storage={len(self.storage_units)}, PHS={len(self.pumped_hydro)}, "
            f"Batteries={len(self.batteries)})>"
        )
