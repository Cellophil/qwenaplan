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


def _t_property_names(cls) -> list[str]:
    """Return the ``_t``-suffixed property names declared anywhere on ``cls``'s MRO.

    Used by both var/sol containers to discover view properties (concrete
    ``_t`` views like ``_BatterySol.p_t`` *and* derived ``_pu_t`` views like
    ``_GeneratorSol.p_pu_t``). Public properties only — leading underscore
    skipped.
    """
    seen: set[str] = set()
    for klass in cls.__mro__:
        for name, attr in klass.__dict__.items():
            if isinstance(attr, property) and name.endswith("_t") and not name.startswith("_"):
                seen.add(name)
    return sorted(seen)


class _VarContainer:
    """Attribute bag for a component's pyoframe variables / expressions.

    Holds named entries (``self.<name>_t``) that are pyoframe ``Variable`` or
    expression objects. Subclasses can add ``_pu_t`` view properties that
    return expressions derived from sibling entries.

    The bag stores its values in ``__dict__``; ``__repr__`` lists every
    plain (non-property) entry so a user typing ``gen.var`` at a REPL can
    see what's available.

    Bracket access (``var["p_t"]``) and iteration (``list(var)``,
    ``"p_t" in var``, ``var.keys()``) are supported alongside dot-access
    so users can build keys programmatically and tab-discover what's
    available.
    """

    __slots__ = ("__dict__", "_owner")

    def __init__(self, owner=None):
        # owner: the Component (or composite) this container is attached to.
        # Stored under a fixed name so subclasses can read sibling state
        # (component parameters like p_nom) when computing _pu_t views.
        object.__setattr__(self, "_owner", owner)

    def _names(self) -> list[str]:
        """Sorted list of ``_t``-suffixed entries available on this container.

        Combines concrete instance entries (pyoframe variables stored in
        ``__dict__``) with view properties defined on the class (``p_pu_t``
        on subclasses, plus property-backed ``_t`` entries for composites
        like ``_BatteryVar.p_t``).
        """
        names: set[str] = {
            k for k in self.__dict__
            if k.endswith("_t") and not k.startswith("_")
        }
        names.update(_t_property_names(type(self)))
        return sorted(names)

    # --- Mapping-ish surface ---------------------------------------------
    # We don't subclass collections.abc.Mapping because the values returned
    # are not a uniform type (Variable / pyoframe expression / DataFrame on
    # the sol side) and forcing __len__/items() to materialise everything
    # would defeat lazy resolution. The four dunders below cover the actual
    # use cases.

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(
                f"{type(self).__name__} keys are strings, got {type(key).__name__}"
            )
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(str(e)) from None

    def __iter__(self):
        return iter(self._names())

    def __contains__(self, key) -> bool:
        return isinstance(key, str) and key in self._names()

    def keys(self) -> list[str]:
        return self._names()

    def __repr__(self) -> str:
        plain = sorted(k for k in self.__dict__ if not k.startswith("_"))
        views = [f"{n} (view)" for n in _t_property_names(type(self))]
        owner_name = getattr(self._owner, "name", "?")
        return f"<{type(self).__name__}({owner_name}): {', '.join(plain + views)}>"


class _SolContainer:
    """Lazy accessor for solved values.

    For every pyoframe variable that lives at ``component.var.<name>_t``,
    accessing ``component.sol.<name>_t`` returns a Polars DataFrame keyed by
    snapshot with the variable's value column renamed from ``solution`` to
    the variable's friendly name.

    Resolution is dynamic (computed on each access) so re-solving the model
    cannot leave a stale snapshot behind. Subclasses may override or extend
    with derived views (``_pu_t``).

    Bracket access (``sol["p_t"]``) and iteration (``list(sol)``,
    ``"p_t" in sol``, ``sol.keys()``) are supported alongside dot-access
    so users can build keys programmatically.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner):
        self._owner = owner

    def __getattr__(self, name: str):
        # Only handle the _t suffix here; bare attribute lookups fall through
        # to AttributeError so typos still raise.
        if not name.endswith("_t"):
            raise AttributeError(name)
        var = getattr(self._owner, "var", None)
        if var is None:
            raise AttributeError(
                f"{type(self._owner).__name__} has no var container; "
                f"cannot resolve sol.{name}."
            )
        target = getattr(var, name, None)
        if target is None:
            available = self._names()
            raise AttributeError(
                f"{type(self._owner).__name__} has no solution attribute {name!r}; "
                f"available: {available}"
            )
        # Friendly column name strips the trailing "_t" so consumers index by
        # the physical name (``df['p']``, ``df['soc']``).
        col = name[:-2] if name.endswith("_t") else name
        return _solution_as(target, col)

    def _names(self) -> list[str]:
        """Sorted list of ``_t``-suffixed entries available on this container.

        Two sources, unioned:
        - Property-backed entries on this sol class (or its subclasses) —
          covers ``_GeneratorSol.p_pu_t`` and the composite ``_BatterySol.p_t``
          / ``_BatterySol.p_dispatch_t`` shape.
        - Whatever resolves through the owner's ``var`` container (concrete
          pyoframe variables in ``var.__dict__`` plus any of *its*
          property-backed views). Skipped silently if the owner has no
          ``var`` (e.g. ``Load`` deletes its ``var`` since loads have no
          decision variables).
        """
        names: set[str] = set(_t_property_names(type(self)))
        var = getattr(self._owner, "var", None)
        if var is not None:
            try:
                names.update(var._names())
            except AttributeError:
                # Defensive: var doesn't follow the _names() protocol. Fall
                # back to whatever's directly inspectable.
                names.update(
                    k for k in getattr(var, "__dict__", {})
                    if k.endswith("_t") and not k.startswith("_")
                )
                names.update(_t_property_names(type(var)))
        return sorted(names)

    # --- Mapping-ish surface (mirrors _VarContainer; see comment there). --

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(
                f"{type(self).__name__} keys are strings, got {type(key).__name__}"
            )
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(str(e)) from None

    def __iter__(self):
        return iter(self._names())

    def __contains__(self, key) -> bool:
        return isinstance(key, str) and key in self._names()

    def keys(self) -> list[str]:
        return self._names()

    def __repr__(self) -> str:
        names = self._names()
        # Tag the property-backed ones as views to match the existing repr style.
        view_set = set(_t_property_names(type(self)))
        items = [f"{n} (view)" if n in view_set else n for n in names]
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

    # ---- Bus-injection abstraction ---------------------------------------
    # Every component answers "what do I contribute at this bus?" via the
    # same call. Power elements ignore the argument (they only touch one
    # bus); branches override below to sign their flow by which end ``bus``
    # is. KCL ([physics.py]) calls into this so the from_bus / to_bus
    # if-statement lives exactly once, and the views layer
    # ([views.py]) reuses the same call so the sol-side bus view shares
    # the sign convention with the symbolic side. No magic — just an
    # interface that lets the two sides of the codebase stay consistent.

    def injection_sign_at(self, bus) -> int:
        """Numeric sign of this component's net injection at ``bus``.

        Default: ``+1``. Branches override to return ``+1`` at ``to_bus``
        and ``-1`` at ``from_bus``. Used by the views layer to sign the
        symbolic contribution to ``view.var.p_t_sum`` (where pyoframe
        composes the signed expression cleanly).

        Raises ``ValueError`` if the component does not touch ``bus`` —
        only meaningful on the branch override.
        """
        return +1

    def sol_sign_at(self, bus) -> int:
        """Numeric sign to apply on ``sol.p_t`` for a *bus view* DataFrame.

        Mirrors :meth:`injection_sign_at`, but corrects for components
        whose ``sol.p_t`` is in a different convention than their bus
        injection. The default is ``+1`` because most ``sol.p_t`` frames
        are already in injection convention (Generator output is
        positive injection; storage's ``sol.p_t`` is already
        ``p_out − p_in``, the signed net). Two overrides:

        - :class:`Load` — ``sol.p_t`` is the *demand* (positive value);
          the bus injection is the negation, so this returns ``-1``.
        - :class:`BranchElement` — same orientation rule as
          ``injection_sign_at`` (``+1`` at ``to_bus``, ``-1`` at ``from_bus``).

        Used by :class:`_ViewSol` to ensure rows of
        ``n.views[bus].sol.p_t`` sum to zero per snapshot. Without this
        the var-side and sol-side bus views would disagree on Load's
        sign, breaking the KCL-read-off-the-data contract.
        """
        return +1

    def injection_at(self, bus):
        """Pyoframe expression for this component's net injection at ``bus``.

        Power-element default: returns ``get_p_net()`` if defined (Load,
        storage, composites — already used by KCL), else falls back to
        ``var.p_t`` (plain Generator). The ``bus`` argument is ignored —
        power elements only touch one bus, and ``injection_sign_at`` is
        a no-op for them.

        Branch elements override (see :class:`BranchElement`) to apply
        the sign returned by ``injection_sign_at(bus)`` so a positive
        ``var.p_t`` (flow from→to) becomes a positive contribution at
        the ``to_bus`` and a negative one at the ``from_bus``.
        """
        sign = self.injection_sign_at(bus)
        if hasattr(self, "get_p_net"):
            base = self.get_p_net()
        else:
            base = self.var.p_t
        return base if sign == 1 else sign * base

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

    # The from_bus / to_bus distinction is arbitrary at modelling time
    # (PyPSA's bus0 / bus1 inheritance). Consumers asking "what does this
    # branch do at *my* bus?" go through ``injection_at(bus)`` and never
    # have to know the orientation themselves. ``var.p_t`` is signed
    # ``from_bus → to_bus``, so the to_bus sees ``+p_t`` and the from_bus
    # sees ``-p_t``.

    def injection_sign_at(self, bus) -> int:
        if bus is self.to_bus:
            return +1
        if bus is self.from_bus:
            return -1
        raise ValueError(
            f"{type(self).__name__} {self.name!r} does not touch bus "
            f"{bus.name!r}; touches {self.from_bus.name!r} and {self.to_bus.name!r}."
        )

    # Branches use the same sign rule on the sol side as on the var
    # side: ``sol.p_t`` is already signed ``from_bus → to_bus``, so
    # negating at the from_bus is the right correction. (Power
    # elements and storage default to ``+1`` because their ``sol.p_t``
    # is already in injection convention; Load overrides to ``-1``.)
    def sol_sign_at(self, bus) -> int:
        return self.injection_sign_at(bus)
