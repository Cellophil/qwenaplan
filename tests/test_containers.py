"""Tests for the mapping-ish surface on ``var`` / ``sol`` containers.

Both ``_VarContainer`` and ``_SolContainer`` accept dot-access (``gen.sol.p_t``)
*and* bracket-access (``gen.sol["p_t"]``), iterate their available ``_t`` keys
(``list(gen.sol)``), support ``in`` (``"p_t" in gen.sol``), and expose
``keys()``. The tests below cover:

- Equivalence of dot- and bracket-access on every container shape we ship:
  the simple-attribute case (``Generator``), the property-backed view case
  (``_GeneratorSol.p_pu_t``), the missing-``var`` case (``Load``), and the
  composite-with-property-_t case (``Battery``, where every ``_t`` entry
  is a window onto an inner storage / generator).
- Error shapes: ``KeyError`` for unknown string keys, ``TypeError`` for
  non-string keys.
- Iteration is deterministic and sorted.

Numerical correctness of the underlying values lives in ``test_pu_views.py``
and the per-component tests; here we only assert that the new accessors
return the same object as the existing dot-access does.
"""
import polars as pl
import pyoptinterface as poi
import pytest

import qwenaplan as qp


# ---------------------------------------------------------------------------
# Generator — the canonical "concrete _t in var.__dict__ + _pu_t property" shape
# ---------------------------------------------------------------------------

class TestGeneratorContainers:
    """Generator's ``var.p_t`` is a concrete pyoframe Variable (lives in
    ``__dict__``); its ``var.p_pu_t`` / ``sol.p_pu_t`` are MRO properties.
    Both must surface through the mapping API."""

    def _solved_gen(self, snapshots):
        n = qp.Network()
        bus = n.add(qp.Bus, "B")
        gen = n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=30.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        return n, gen

    def test_sol_bracket_equals_dot_for_concrete_t(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        # Same DataFrame contents whichever access form we use. The objects
        # may not be identical (each access re-evaluates), but the values
        # must match.
        assert gen.sol["p_t"].equals(gen.sol.p_t)

    def test_sol_bracket_equals_dot_for_pu_view(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        assert gen.sol["p_pu_t"].equals(gen.sol.p_pu_t)

    def test_var_bracket_returns_same_variable(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        # var.p_t is stored in __dict__; bracket and dot return the same obj.
        assert gen.var["p_t"] is gen.var.p_t

    def test_iter_lists_concrete_and_view_keys(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        keys = list(gen.sol)
        # p_t is concrete on var; p_pu_t is a property on _GeneratorSol.
        # The mapping surface should expose both.
        assert "p_t" in keys
        assert "p_pu_t" in keys
        # Sorted, deterministic.
        assert keys == sorted(keys)

    def test_keys_method_matches_iter(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        assert gen.sol.keys() == list(gen.sol)
        assert gen.var.keys() == list(gen.var)

    def test_contains_string_keys(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        assert "p_t" in gen.sol
        assert "p_pu_t" in gen.sol
        assert "bogus" not in gen.sol
        # Non-string in ``in`` returns False (does not raise).
        assert 42 not in gen.sol

    def test_unknown_string_key_raises_keyerror(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        with pytest.raises(KeyError):
            _ = gen.sol["bogus"]
        with pytest.raises(KeyError):
            _ = gen.var["bogus"]

    def test_non_string_key_raises_typeerror(self, snapshots):
        _, gen = self._solved_gen(snapshots)
        with pytest.raises(TypeError, match="strings"):
            _ = gen.sol[42]
        with pytest.raises(TypeError, match="strings"):
            _ = gen.var[("p_t",)]


# ---------------------------------------------------------------------------
# Load — the "no var container" edge case
# ---------------------------------------------------------------------------

class TestLoadContainers:
    """Loads have no decision variables; their ``var`` attribute is deleted
    in ``Load.__init__``. ``sol.p_t`` is a property returning the parameter
    profile. The mapping surface must work despite the missing ``var``."""

    def test_sol_bracket_returns_param_dataframe(self, network, snapshots):
        bus = network.add(qp.Bus, "B")
        ld = network.add(qp.Load, "L", bus=bus, p_set=42.0)
        network.set_snapshots(snapshots)
        # Available before optimize() since p_set is a parameter.
        assert ld.sol["p_t"].equals(ld.sol.p_t)
        assert ld.sol["p_t"]["p"].to_list() == [42.0] * len(snapshots)

    def test_iter_returns_only_p_t(self, network, snapshots):
        bus = network.add(qp.Bus, "B")
        ld = network.add(qp.Load, "L", bus=bus, p_set=10.0)
        network.set_snapshots(snapshots)
        # Load defines just p_t on _LoadSol; nothing else should leak in.
        assert list(ld.sol) == ["p_t"]
        assert "p_t" in ld.sol
        assert "p_pu_t" not in ld.sol

    def test_unknown_key_still_keyerror_without_var(self, network, snapshots):
        bus = network.add(qp.Bus, "B")
        ld = network.add(qp.Load, "L", bus=bus, p_set=10.0)
        network.set_snapshots(snapshots)
        with pytest.raises(KeyError):
            _ = ld.sol["bogus_t"]


# ---------------------------------------------------------------------------
# Storage — has both concrete _t (var.__dict__) and _pu_t views on subclass
# ---------------------------------------------------------------------------

class TestStorageContainers:
    """``_StorageBase`` keeps ``soc_t``/``p_in_t``/``p_out_t`` as concrete
    pyoframe Variables in ``var.__dict__``, and ``_StorageBaseVar`` /
    ``_StorageBaseSol`` add ``soc_pu_t`` / ``p_pu_t`` as MRO properties."""

    def test_iter_includes_concrete_and_pu_views(self, storage_test_network, snapshots):
        n, _, _, storage, _ = storage_test_network
        n.set_snapshots(snapshots)
        keys = list(storage.sol)
        for expected in ("soc_t", "p_in_t", "p_out_t", "soc_pu_t", "p_pu_t"):
            assert expected in keys, (expected, keys)
        # Same on var side.
        var_keys = list(storage.var)
        for expected in ("soc_t", "p_in_t", "p_out_t", "soc_pu_t", "p_pu_t"):
            assert expected in var_keys, (expected, var_keys)

    def test_bracket_access_after_solve(self, storage_test_network, snapshots):
        n, _, _, storage, _ = storage_test_network
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        # All five entries reachable both ways and equal.
        for k in ("soc_t", "p_in_t", "p_out_t", "soc_pu_t", "p_pu_t"):
            assert storage.sol[k].equals(getattr(storage.sol, k))


# ---------------------------------------------------------------------------
# Battery composite — every _t entry is a property (no __dict__ entries at all)
# ---------------------------------------------------------------------------

class TestBatteryContainers:
    """Battery's ``_BatteryVar`` / ``_BatterySol`` expose all ``_t`` entries
    as properties (windows onto the inner storage). There is no ``__dict__``
    contribution; the mapping surface must rely entirely on the MRO walk."""

    def test_iter_includes_all_battery_views(self, battery_test_network, snapshots):
        n, _, _, battery, _ = battery_test_network
        n.set_snapshots(snapshots)
        keys = set(battery.sol)
        # Every documented Battery view should be discoverable.
        for expected in (
            "soc_t", "p_in_t", "p_out_t", "p_t",
            "p_store_t", "p_dispatch_t",
            "soc_pu_t", "p_pu_t",
        ):
            assert expected in keys, (expected, keys)

    def test_bracket_access_for_property_only_t(self, battery_test_network, snapshots):
        n, _, _, battery, _ = battery_test_network
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        # p_t is computed (p_out − p_in); p_dispatch_t / p_store_t are aliases
        # with a friendlier column name. All must be reachable via bracket.
        for k in ("p_t", "p_in_t", "p_out_t", "p_store_t", "p_dispatch_t",
                  "soc_t", "soc_pu_t", "p_pu_t"):
            assert battery.sol[k].equals(getattr(battery.sol, k))

    def test_var_bracket_for_property_only_t(self, battery_test_network, snapshots):
        n, _, _, battery, _ = battery_test_network
        n.set_snapshots(snapshots)
        # var-side property windows return pyoframe Variables / expressions.
        # We only check identity-of-result, not numerical content.
        for k in ("soc_t", "p_in_t", "p_out_t"):
            assert battery.var[k] is getattr(battery.var, k)
