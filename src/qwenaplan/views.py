"""Named subsets of network components with aggregated ``var`` / ``sol``.

A :class:`View` is a thin handle wrapping ``(name, components, network)``.
It exposes ``view.var`` / ``view.sol`` shaped like a single component's
containers, but the values aggregate across the members. Two layers,
deliberately asymmetric:

- **``view.var``** — only ``_sum`` is meaningful. A Python list of pyoframe
  expressions doesn't compose into a constraint; only ``Σ_member <expr>``
  does. So ``view.var.p_t_sum`` resolves and ``view.var.p_t`` raises.
- **``view.sol``** — both shapes are useful. ``view.sol.p_t`` is a *wide*
  Polars DataFrame (one column per member, named after the member); it's
  what you want for stack charts and per-component inspection.
  ``view.sol.p_t_sum`` is a single-column long DataFrame, the per-snapshot
  total, interchangeable with a single component's ``sol.p_t``.

Two construction modes:

- **Free view** (``bus=None``): aggregates ``member.get_p_net()`` for the
  ``p_t`` path (so Loads enter as ``pf.Param``, storage as
  ``p_out − p_in``, branches as bare ``var.p_t``); other ``_t`` names go
  through ``member.var.<name_t>`` directly. This is what the
  registry-keyed default views (``"generators"``, ``"loads"``, …) use.
- **Bus view** (``bus=<Bus>``): every member's ``p_t`` contribution is
  asked of :meth:`Component.injection_at`, which signs branches by
  ``from_bus`` / ``to_bus``. Rows of ``view.sol.p_t`` then sum to zero per
  snapshot — KCL read off the data.

Both ``injection_at`` and the parallel numeric :meth:`Component.injection_sign_at`
live on :class:`Component` (see [base.py]), with branch overrides on
:class:`BranchElement`. The view layer never touches ``from_bus`` /
``to_bus`` directly — that knowledge stays exactly one place.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import polars as pl

if TYPE_CHECKING:
    from .components import Bus
    from .network import Network


class View:
    """A named subset of network components.

    Parameters
    ----------
    name : str
        Identifier used for repr / dict key in ``Network.views``.
    components : list
        Member components. Must all belong to the same ``Network``. ``Bus``
        objects are not allowed (no useful aggregation semantics —
        registry views like ``"generators"`` cover the meaningful cases,
        and bus views are auto-built keyed by bus name).
    bus : Bus, optional
        If supplied, the view is in *bus mode*: ``var.p_t_sum`` /
        ``sol.p_t`` aggregate via :meth:`Component.injection_at`, signing
        branches correctly. Auto-populated bus views pass this; user-built
        custom views typically don't.
    network : Network, optional
        Network the view belongs to. Normally inferred from the
        components (they all carry ``.network``); only needed when the
        component list is empty (e.g. an auto-populated ``"batteries"``
        view on a network with no batteries) so the sol layer can still
        reach the snapshot dimension.
    """

    __slots__ = ("name", "components", "bus", "network", "var", "sol")

    def __init__(
        self,
        name: str,
        components: list,
        *,
        bus: Optional["Bus"] = None,
        network: Optional["Network"] = None,
    ):
        # Local import keeps views.py importable before components.py
        # finishes (init.py wires both).
        from .components import Bus

        self.name = name

        bus_members = [c for c in components if isinstance(c, Bus)]
        if bus_members:
            offenders = ", ".join(b.name for b in bus_members)
            raise TypeError(
                f"View {name!r}: Bus objects cannot be view members "
                f"({offenders}). Buses don't have a meaningful aggregation "
                f"semantics; use the bus's auto-populated view via "
                f"n.views[bus.name] for KCL identities, or build a custom "
                f"View over the components attached to those buses."
            )

        self.components = list(components)

        nets = {c.network for c in self.components}
        if len(nets) > 1:
            raise ValueError(
                f"View {name!r} mixes components from different networks; "
                f"each View must live in exactly one Network."
            )
        inferred = next(iter(nets)) if nets else None
        if network is not None and inferred is not None and network is not inferred:
            raise ValueError(
                f"View {name!r}: explicit network argument disagrees with "
                f"the network inferred from members."
            )
        self.network = inferred if inferred is not None else network
        self.bus = bus

        self.var = _ViewVar(owner=self)
        self.sol = _ViewSol(owner=self)

    def __repr__(self) -> str:
        kind = f"bus={self.bus.name!r}" if self.bus is not None else "free"
        names = [c.name for c in self.components]
        return f"<View(name={self.name!r}, {kind}, members={names})>"


# ---------------------------------------------------------------------------
# var side — only ``_sum`` is meaningful
# ---------------------------------------------------------------------------

class _ViewVar:
    """Symbolic side of a view.

    Only summed expressions resolve. ``view.var.<name>_t_sum`` returns
    the pyoframe expression ``Σ_member contribution(member, name_t)``,
    where ``contribution`` is:

    - bus view, ``p_t``: ``member.injection_at(view.bus)`` — signs
      branches automatically.
    - free view, ``p_t``: ``member.get_p_net()`` if defined (Load,
      storage, composites), else ``member.var.p_t``. Loads contribute
      as a ``pf.Param`` — they show up as the constant in the
      expression, exactly as the user-built mixed-view case asks.
    - any other ``<name>_t``: direct attribute lookup on
      ``member.var``. The bus argument is irrelevant for these (signs
      are only meaningful for ``p_t``).
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: View):
        self._owner = owner

    def __getattr__(self, attr: str):
        # Internal attributes (``_owner`` is in __slots__) bypass this;
        # everything else routes through the suffix protocol.
        if not attr.endswith("_t_sum"):
            raise AttributeError(
                f"View var only exposes summed expressions (suffix _t_sum); "
                f"got {attr!r}. For per-member symbolic access, iterate "
                f"view.components and read each member's var directly."
            )
        name_t = attr[: -len("_sum")]   # ``"p_t_sum"`` → ``"p_t"``
        return self._aggregate(name_t)

    def _aggregate(self, name_t: str):
        view = self._owner
        terms: list = []
        missing: list[str] = []
        for c in view.components:
            term = self._member_term(c, name_t)
            if term is None:
                missing.append(c.name)
            else:
                terms.append(term)

        if missing:
            raise AttributeError(
                f"View {view.name!r}: cannot sum {name_t!r} — "
                f"missing on members: {missing}"
            )
        if not terms:
            # Pyoframe can't synthesise an indexed-zero without a
            # representative term, and an unindexed scalar zero would
            # be wrong (the consumer expects a snapshot-indexed expr).
            raise AttributeError(
                f"View {view.name!r} is empty — no terms to sum for {name_t!r}. "
                f"Add at least one member, or use sol.{name_t}_sum which "
                f"can return an indexed zero column."
            )

        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total

    def _member_term(self, component, name_t: str):
        view = self._owner
        # Bus mode + p_t: the component knows its sign at our bus.
        # Power elements ignore the bus arg; branches sign by
        # from_bus / to_bus. Loads return a pf.Param. (See
        # Component.injection_at / BranchElement.injection_sign_at.)
        if view.bus is not None and name_t == "p_t":
            try:
                return component.injection_at(view.bus)
            except (ValueError, AttributeError):
                # Component does not touch this bus, or has no var/p_t
                # at all. Treat as missing → the caller raises listing
                # the offender.
                return None

        # Free view, p_t: prefer ``get_p_net`` (Loads → Param, storage
        # → signed expr, composites → signed expr) so user-built mixed
        # views resolve loads as constants automatically.
        if name_t == "p_t" and hasattr(component, "get_p_net"):
            return component.get_p_net()

        # Anything else (or plain Generator with no get_p_net): direct
        # attribute lookup on ``var``. ``Load`` has no ``var`` and no
        # ``_t`` other than ``p_t`` (which is handled above), so the
        # ``getattr(None, ...)`` path falls through to None → missing.
        var = getattr(component, "var", None)
        if var is None:
            return None
        return getattr(var, name_t, None)


# ---------------------------------------------------------------------------
# sol side — wide DataFrame *and* sum
# ---------------------------------------------------------------------------

class _ViewSol:
    """Solved side of a view.

    Two shapes per ``_t`` name:

    - ``view.sol.<name>_t`` — wide Polars DataFrame, columns
      ``[snapshot_dim, *member_names]``. Each member's value column is
      renamed to that member's ``name`` (signed for bus views), then
      chain-joined on the snapshot dim. Column order = construction
      order (registry insertion order for auto-populated views).

    - ``view.sol.<name>_t_sum`` — long Polars DataFrame, columns
      ``[snapshot_dim, p]``. Sign-applied sum across members per
      snapshot. The value column is named after the physical name
      (stripping ``_t``) so the result is interchangeable with a
      single-component ``sol.<name>_t``.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: View):
        self._owner = owner

    def __getattr__(self, attr: str):
        if attr.endswith("_t_sum"):
            return self._aggregate_sum(attr[: -len("_sum")])
        if attr.endswith("_t"):
            return self._aggregate_wide(attr)
        raise AttributeError(attr)

    # ---- helpers --------------------------------------------------------

    def _snapshot_col(self) -> str:
        """Name of the snapshot dimension (e.g. ``"hour"``).

        Raises if the network has no snapshots yet — the sol layer is
        only meaningful post ``set_snapshots``, but failing here gives a
        clearer error than the join-time KeyError.
        """
        view = self._owner
        if view.network is None or view.network.snapshots is None:
            raise RuntimeError(
                f"View {view.name!r}: cannot resolve sol — network has no "
                f"snapshots yet (call n.set_snapshots(...) first)."
            )
        return view.network.snapshots.name

    def _member_frame(self, component, name_t: str) -> Optional[pl.DataFrame]:
        """Fetch ``component.sol.<name_t>``; return None if unavailable."""
        sol = getattr(component, "sol", None)
        if sol is None:
            return None
        try:
            return getattr(sol, name_t)
        except AttributeError:
            return None

    def _value_col(self, df: pl.DataFrame, snap: str) -> str:
        """The single non-snapshot column on a member sol frame."""
        non_snap = [c for c in df.columns if c != snap]
        if len(non_snap) != 1:
            # Defensive: sol frames are always (snap, value). If a
            # subclass ever returns a multi-column frame, refuse to
            # guess.
            raise RuntimeError(
                f"Expected one value column on member sol frame, got "
                f"{non_snap!r}; columns = {df.columns}"
            )
        return non_snap[0]

    # ---- wide form ------------------------------------------------------

    def _aggregate_wide(self, name_t: str) -> pl.DataFrame:
        view = self._owner
        snap = self._snapshot_col()

        # Empty view → snapshot-indexed frame with no value columns.
        # Preserves the dim so downstream joins / slicing still work.
        if not view.components:
            return view.network.snapshots.to_frame()

        missing: list[str] = []
        per_member: list[pl.DataFrame] = []
        for c in view.components:
            df = self._member_frame(c, name_t)
            if df is None:
                missing.append(c.name)
                continue

            value_col = self._value_col(df, snap)
            # Use ``sol_sign_at`` (not ``injection_sign_at``) because
            # ``sol.p_t`` semantics differ from injection convention for
            # Loads (sol = +p_set, injection = -p_set). For branches and
            # power elements the two are equal; for Loads sol_sign_at
            # returns -1.
            sign = c.sol_sign_at(view.bus) if view.bus is not None else +1
            renamed = df.rename({value_col: c.name}).select([snap, c.name])
            if sign != 1:
                renamed = renamed.with_columns(
                    (pl.col(c.name) * sign).alias(c.name)
                )
            per_member.append(renamed)

        if missing:
            raise AttributeError(
                f"View {view.name!r}: cannot build wide {name_t!r} — "
                f"missing on members: {missing}"
            )

        # Chain join on the snapshot dim. polars' join is fast even for
        # many members; we don't bother with concat-and-pivot.
        out = per_member[0]
        for frame in per_member[1:]:
            out = out.join(frame, on=snap, how="inner")
        return out

    # ---- sum form -------------------------------------------------------

    def _aggregate_sum(self, name_t: str) -> pl.DataFrame:
        view = self._owner
        snap = self._snapshot_col()
        # Physical column name: strip the trailing ``_t`` so the result
        # is shaped like a single component's sol (``df['p']``,
        # ``df['soc']``, …).
        out_col = name_t[:-2] if name_t.endswith("_t") else name_t

        # Empty view → snapshot-indexed frame with a zero column. Honest:
        # the symbolic side raises here (no representative term), but
        # numerically zero is the right answer for "sum of nothing."
        if not view.components:
            n = len(view.network.snapshots)
            return view.network.snapshots.to_frame().with_columns(
                pl.Series(out_col, [0.0] * n)
            )

        # Reuse the wide-form (it already applies signs and validates
        # member coverage). Then sum across the value columns.
        wide = self._aggregate_wide(name_t)
        value_cols = [c for c in wide.columns if c != snap]
        if not value_cols:
            n = len(view.network.snapshots)
            return view.network.snapshots.to_frame().with_columns(
                pl.Series(out_col, [0.0] * n)
            )
        return wide.select([
            pl.col(snap),
            pl.sum_horizontal(value_cols).alias(out_col),
        ])


# ---------------------------------------------------------------------------
# Default view builder
# ---------------------------------------------------------------------------

def _build_bus_view(network: "Network", bus: "Bus") -> View:
    """Auto-populated bus view: every connected component, signed by bus.

    Power elements come through :meth:`Network.get_connected_power_elements`
    and lines/links through :meth:`Network.get_connected_lines`. The
    sign convention lives on the components (via ``injection_at`` /
    ``injection_sign_at``); we just hand them the bus and let them
    sort themselves out.
    """
    members = list(network.get_connected_power_elements(bus))
    members.extend(network.get_connected_lines(bus))
    return View(bus.name, members, bus=bus)
