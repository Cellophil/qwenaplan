import polars as pl
import pyoframe as pf
from typing import Dict, List, Any
from .components import Bus, Generator, ACLine, Link, StorageUnit, PumpedHydroStorage, Battery


class Network:
    def __init__(self):
        self.buses: Dict[str, Bus] = {}
        self.lines: Dict[str, ACLine] = {}
        self.links: Dict[str, Link] = {}
        self.generators: Dict[str, Generator] = {}
        self.storage_units: Dict[str, StorageUnit] = {}
        self.pumped_hydro: Dict[str, PumpedHydroStorage] = {}
        self.batteries: Dict[str, Battery] = {}

        self.snapshots = None
        self.model = None
        self._is_locked = False

    def add(self, cls, name: str, **kwargs):
        if self._is_locked:
            raise RuntimeError("Network is locked after create_model().")

        if cls == Bus:
            obj = Bus(name, self, **kwargs)
            self.buses[name] = obj
        elif cls == ACLine:
            obj = ACLine(name, self, **kwargs)
            self.lines[name] = obj
        elif cls == Link:
            obj = Link(name, self, **kwargs)
            self.links[name] = obj
        elif cls == Generator:
            obj = Generator(name, self, **kwargs)
            self.generators[name] = obj
        elif cls == StorageUnit:
            obj = StorageUnit(name, self, **kwargs)
            self.storage_units[name] = obj
        elif cls == PumpedHydroStorage:
            obj = PumpedHydroStorage(name, self, **kwargs)
            self.pumped_hydro[name] = obj
        elif cls == Battery:
            obj = Battery(name, self, **kwargs)
            self.batteries[name] = obj
        else:
            raise ValueError(f"Unsupported component class {cls}")
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

    def create_model(self):
        """Finalize topology and generate constraints."""
        if self.snapshots is None:
            raise RuntimeError("Must call set_snapshots() before create_model().")

        print("Building optimization model...")
        # 1. Initialize pyoframe model
        self.model = pf.Model()

        # 2. Re-trigger variable setup to add them to the model
        all_components = {
            **self.buses,
            **self.lines,
            **self.links,
            **self.generators,
            **self.storage_units,
            **self.pumped_hydro,
            **self.batteries,
        }.values()
        for comp in all_components:
            comp.setup_variables_for_model(self.model)

        # 3. Add constraints to the model
        for comp in all_components:
            comp.setup_constraints(self.model)

        self._is_locked = True
        print("Model created and network locked.")

    def __repr__(self):
        return f"<Network(Buses={len(self.buses)}, Lines={len(self.lines)}, Gens={len(self.generators)}, Storage={len(self.storage_units)}, PHS={len(self.pumped_hydro)}, Batteries={len(self.batteries)})>"
