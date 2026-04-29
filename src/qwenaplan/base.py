from abc import ABC, abstractmethod
from typing import Any, Optional


class Component(ABC):
    """Base class for all network elements."""

    def __init__(self, name: str, network: "Network"):
        self.name = name
        self.network = network
        # The internal reference to the Polars row/index
        self._data_idx: Optional[int] = None

    @abstractmethod
    def setup_variables(self):
        """Register variables in pyoframe when snapshots are set."""
        pass

    @abstractmethod
    def setup_constraints(self, model: Any):
        """Add constraints to the optimization model."""
        pass

    @abstractmethod
    def setup_objective(self, model: Any):
        """Add contribution to the objective function.
        
        Override in subclasses that contribute to the objective (e.g., generators with marginal costs).
        Branch components should implement this with pass (no direct objective contribution).
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"


class PowerElement(Component):
    """Elements that inject or withdraw power from a bus."""

    def __init__(self, name: str, network: "Network", bus):
        super().__init__(name, network)
        # Import Bus locally to avoid circular import
        from .components import Bus

        if not isinstance(bus, Bus):
            raise TypeError(f"Expected Bus object, got {type(bus)}")
        self.bus = bus


class BranchElement(Component):
    """Elements that connect two buses."""

    def __init__(self, name: str, network: "Network", from_bus, to_bus):
        super().__init__(name, network)
        # Import Bus locally to avoid circular import
        from .components import Bus

        if not (isinstance(from_bus, Bus) and isinstance(to_bus, Bus)):
            raise TypeError("Both from_bus and to_bus must be Bus objects")
        self.from_bus = from_bus
        self.to_bus = to_bus
