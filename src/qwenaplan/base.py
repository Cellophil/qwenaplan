from abc import ABC, abstractmethod
from typing import Any, Optional


def _solution_as(variable, column_name):
    """Return ``variable.solution`` with the ``solution`` column renamed.

    Centralises the pattern ``var.solution["solution"].to_list()`` used all
    over the test suite. Returns the full Polars DataFrame (snapshot index +
    value column) so users can join, filter, or aggregate as needed.

    Raises a clear error if the model has not been solved yet (pyoframe
    raises something less helpful in that case).
    """
    try:
        df = variable.solution
    except Exception as e:
        raise RuntimeError(
            f"Cannot read solution for {column_name!r}: model not solved "
            f"(or pyoframe internal error: {e!s})."
        ) from e
    return df.rename({"solution": column_name})


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
    def setup_objective(self, network: "Network"):
        """Add contribution to the network objective.

        Components that contribute (e.g. Generator with marginal_cost) should
        call ``network._add_to_objective(expr)`` with a pyoframe expression
        indexed by snapshots. The network sums and ``.sum()``s all
        contributions when assembling ``model.minimize``.

        Components without direct objective contribution should ``pass``.
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
