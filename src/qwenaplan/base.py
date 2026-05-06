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


class _VarContainer:
    """Attribute bag for a component's pyoframe variables / expressions.

    Holds named entries (``self.<name>_t``) that are pyoframe ``Variable`` or
    expression objects. Subclasses can add ``_pu_t`` view properties that
    return expressions derived from sibling entries.

    The bag stores its values in ``__dict__``; ``__repr__`` lists every
    plain (non-property) entry so a user typing ``gen.var`` at a REPL can
    see what's available.
    """

    __slots__ = ("__dict__", "_owner")

    def __init__(self, owner=None):
        # owner: the Component (or composite) this container is attached to.
        # Stored under a fixed name so subclasses can read sibling state
        # (component parameters like p_nom) when computing _pu_t views.
        object.__setattr__(self, "_owner", owner)

    def __repr__(self) -> str:
        items = sorted(k for k in self.__dict__ if not k.startswith("_"))
        # Include any computed _pu_t view properties exposed by subclasses.
        for cls in type(self).__mro__:
            for name, attr in cls.__dict__.items():
                if isinstance(attr, property) and name.endswith("_pu_t"):
                    items.append(f"{name} (view)")
        owner_name = getattr(self._owner, "name", "?")
        return f"<{type(self).__name__}({owner_name}): {', '.join(items)}>"


class _SolContainer:
    """Lazy accessor for solved values.

    For every pyoframe variable that lives at ``component.var.<name>_t``,
    accessing ``component.sol.<name>_t`` returns a Polars DataFrame keyed by
    snapshot with the variable's value column renamed from ``solution`` to
    the variable's friendly name.

    Resolution is dynamic (computed on each access) so re-solving the model
    cannot leave a stale snapshot behind. Subclasses may override or extend
    with derived views (``_pu_t``).
    """

    __slots__ = ("_owner",)

    def __init__(self, owner):
        self._owner = owner

    def __getattr__(self, name: str):
        # Only handle the _t suffix here; bare attribute lookups fall through
        # to AttributeError so typos still raise.
        if not name.endswith("_t"):
            raise AttributeError(name)
        var = getattr(self._owner.var, name, None)
        if var is None:
            raise AttributeError(
                f"{type(self._owner).__name__} has no solution attribute {name!r}; "
                f"available: {sorted(k for k in self._owner.var.__dict__ if k.endswith('_t'))}"
            )
        # Friendly column name strips the trailing "_t" so consumers index by
        # the physical name (``df['p']``, ``df['soc']``).
        col = name[:-2] if name.endswith("_t") else name
        return _solution_as(var, col)

    def __repr__(self) -> str:
        items = sorted(k for k in self._owner.var.__dict__ if k.endswith("_t"))
        for cls in type(self).__mro__:
            for name, attr in cls.__dict__.items():
                if isinstance(attr, property) and name.endswith("_pu_t"):
                    items.append(f"{name} (view)")
        return f"<{type(self).__name__}({self._owner.name}): {', '.join(items)}>"


class Component(ABC):
    """Base class for all network elements."""

    def __init__(self, name: str, network: "Network"):
        self.name = name
        self.network = network
        # The internal reference to the Polars row/index
        self._data_idx: Optional[int] = None
        # var / sol containers. Concrete subclasses may override the classes
        # they instantiate (see Generator, _StorageBase, etc.) to expose
        # per-unit view properties.
        self.var = self._var_container_cls()(owner=self)
        self.sol = self._sol_container_cls()(owner=self)

    # Subclasses override these to plug in component-specific containers
    # with _pu_t view properties.
    def _var_container_cls(self):
        return _VarContainer

    def _sol_container_cls(self):
        return _SolContainer

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
