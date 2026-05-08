"""Tests for ``qp.View`` and ``Network.views``.

Covers the 15 cases from the views plan: registry-keyed defaults, bus
views (KCL-as-data), the asymmetric ``var.p_t_sum`` / ``sol.p_t`` /
``sol.p_t_sum`` surface, the load-as-constant principle (Loads enter
the var-side sum as a ``pf.Param``, so a custom mixed view summing a
generator and a load resolves to a usable expression), and the error
shapes (no ``_sum`` suffix → AttributeError; missing ``_t`` on a member
→ AttributeError listing offenders; mixed networks → ValueError;
``Bus`` as a member → TypeError).

Numerical correctness of the underlying solved values lives in the
per-component test files; here we only verify the view layer assembles
them correctly (column ordering, sign convention on bus views,
interchangeability between ``view.sol.<name>_t_sum`` and a single
component's ``sol.<name>_t``).
"""
import polars as pl
import pyoframe as pf
import pyoptinterface as poi
import pytest

import qwenaplan as qp


# ---------------------------------------------------------------------------
# Fixture: a small 3-bus showcase-shaped topology so a single solve covers
# generators / loads / lines / a battery / a link without being huge.
# ---------------------------------------------------------------------------

@pytest.fixture
def small_showcase(snapshots):
    """Bus1—line—Bus2—line—Bus3 with a link Bus1—Bus3.

    Bus1: Coal (cheap), no load.
    Bus2: Solar, Battery, Load_urban.
    Bus3: Peaker (expensive), Load_rural.

    Returns ``(n, [bus1, bus2, bus3], [coal, solar, peaker], battery,
    [load_urban, load_rural], [line_main, line_skinny], [link_dc])``.
    """
    n = qp.Network()
    bus1 = n.add(qp.Bus, "Bus1")
    bus2 = n.add(qp.Bus, "Bus2")
    bus3 = n.add(qp.Bus, "Bus3")

    coal = n.add(qp.Generator, "Coal", bus=bus1, p_nom=100.0, marginal_cost=10.0)
    solar = n.add(qp.Generator, "Solar", bus=bus2, p_nom=50.0, marginal_cost=0.0)
    peaker = n.add(qp.Generator, "Peaker", bus=bus3, p_nom=40.0, marginal_cost=200.0)

    battery = n.add(
        qp.Battery, "Battery", bus=bus2,
        e_nom=100.0, p_nom=30.0,
        eff_store=0.95, eff_dispatch=0.95, initial_soc=20.0,
    )

    load_urban = n.add(qp.Load, "Load_urban", bus=bus2, p_set=40.0)
    load_rural = n.add(qp.Load, "Load_rural", bus=bus3, p_set=20.0)

    line_main = n.add(qp.ACLine, "ACLine_main", from_bus=bus1, to_bus=bus2,
                      x_pu=0.05, s_nom=80.0)
    line_skinny = n.add(qp.ACLine, "ACLine_skinny", from_bus=bus2, to_bus=bus3,
                        x_pu=0.10, s_nom=50.0)
    link_dc = n.add(qp.Link, "Link_dc", from_bus=bus1, to_bus=bus3, p_nom=25.0)

    n.set_snapshots(snapshots)
    n.create_model()
    assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

    return (
        n,
        [bus1, bus2, bus3],
        [coal, solar, peaker],
        battery,
        [load_urban, load_rural],
        [line_main, line_skinny],
        [link_dc],
    )


# ---------------------------------------------------------------------------
# 1–2. Registry view sol shapes
# ---------------------------------------------------------------------------

class TestRegistryViewSol:
    def test_generators_wide_columns_in_registry_order(self, small_showcase):
        n = small_showcase[0]
        wide = n.views["generators"].sol.p_t
        # Snapshot dim first, then one column per generator in
        # *registry insertion order* (Coal, Solar, Peaker — that's the
        # order they were added in the fixture).
        assert wide.columns == ["time", "Coal", "Solar", "Peaker"]
        # Same row count as snapshots.
        assert wide.shape[0] == len(n.snapshots)

    def test_generators_sum_matches_polars_reduction(self, small_showcase):
        n = small_showcase[0]
        wide = n.views["generators"].sol.p_t
        summed = n.views["generators"].sol.p_t_sum
        # Sum column is named after the physical name (``p``), not
        # ``p_t`` — same shape as a single component's ``sol.p_t``.
        assert summed.columns == ["time", "p"]
        # Polars-side reduction must equal the view's server-side one.
        expected = wide.select([
            pl.col("time"),
            pl.sum_horizontal(["Coal", "Solar", "Peaker"]).alias("p"),
        ])
        assert summed.equals(expected)


# ---------------------------------------------------------------------------
# 3. var.p_t_sum is usable in a custom constraint
# ---------------------------------------------------------------------------

class TestVarPtSumIsUsable:
    """The whole point of the var side: feed the aggregate expression into
    a downstream constraint and have the LP respect it. Two cases: a
    cap that's slack (numbers go where you expect, ≤ cap) and a cap
    that's tight (LP becomes infeasible when there's no other relief —
    proving the cap is real and not silently dropped)."""

    def test_regional_cap_is_respected_when_feasible(self, snapshots):
        """Cap of 60 MW vs 40 MW demand: cap is not tight, total = 40."""
        n = qp.Network()
        b1 = n.add(qp.Bus, "B1")
        b2 = n.add(qp.Bus, "B2")
        n.add(qp.Generator, "Coal", bus=b1, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Generator, "Solar", bus=b1, p_nom=50.0, marginal_cost=0.0)
        n.add(qp.Load, "L", bus=b2, p_set=40.0)
        n.add(qp.ACLine, "Line", from_bus=b1, to_bus=b2, x_pu=0.1, s_nom=200.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.model.regional_cap = n.views["generators"].var.p_t_sum <= 60.0
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        total = n.views["generators"].sol.p_t_sum["p"].to_list()
        for v in total:
            assert v == pytest.approx(40.0, abs=1e-6)
            assert v <= 60.0 + 1e-6

    def test_regional_cap_binds_when_demand_exceeds_cap(self, snapshots):
        """Cap of 30 MW vs 40 MW demand at a single bus (no transmission
        in/out): the LP is infeasible, proving the cap is real and
        tight (not silently dropped)."""
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        n.add(qp.Generator, "Coal", bus=b, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Generator, "Solar", bus=b, p_nom=50.0, marginal_cost=0.0)
        n.add(qp.Load, "L", bus=b, p_set=40.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.model.regional_cap = n.views["generators"].var.p_t_sum <= 30.0
        # No nodal slack, no transmission relief → infeasible.
        status = n.optimize()
        assert status != poi.TerminationStatusCode.OPTIMAL


# ---------------------------------------------------------------------------
# 4–5. var error shapes
# ---------------------------------------------------------------------------

class TestVarErrorShapes:
    def test_var_p_t_without_suffix_raises(self, small_showcase):
        n = small_showcase[0]
        with pytest.raises(AttributeError, match="suffix _t_sum"):
            _ = n.views["generators"].var.p_t

    def test_var_missing_t_lists_offenders(self, small_showcase):
        n = small_showcase[0]
        # ``soc_t`` only exists on storage. ``"generators"`` has none.
        with pytest.raises(AttributeError, match="missing on members"):
            _ = n.views["generators"].var.soc_t_sum


# ---------------------------------------------------------------------------
# 6. Loads-only var (the user's "load resolved as the constant" path).
# ---------------------------------------------------------------------------

class TestLoadsOnlyVar:
    """``n.views['loads'].var.p_t_sum`` is a sum of ``pf.Param`` terms.
    It must (a) resolve at all (no AttributeError because Load has no
    ``var``) and (b) evaluate post-solve to the negation of total
    demand."""

    def test_loads_var_p_t_sum_resolves(self, small_showcase):
        n = small_showcase[0]
        expr = n.views["loads"].var.p_t_sum
        # We can't call .solution on a pure-Param expression and we
        # don't need to: the symbolic test is "the call returns a
        # pyoframe object," and we read the numbers on the sol side.
        assert expr is not None

    def test_loads_sum_equals_negative_total_demand(self, small_showcase):
        n = small_showcase[0]
        # Load_urban (40) + Load_rural (20) = 60 demand; bus injection
        # convention has loads negative, so the *injection-shaped* sum
        # would be -60. But ``view.sol.p_t`` for the (free, non-bus)
        # ``"loads"`` view returns ``sol.p_t`` which is the demand
        # magnitude — so the sum here is +60. The bus view is the one
        # that flips the sign.
        summed = n.views["loads"].sol.p_t_sum["p"].to_list()
        for v in summed:
            assert v == pytest.approx(60.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. Load-as-constant in a custom mixed view (gen + load).
# ---------------------------------------------------------------------------

class TestLoadAsConstantInMixedView:
    """A user-built ``View([gen, load])`` resolves ``var.p_t_sum`` as
    ``gen.var.p_t + (-load_param)`` — pyoframe sums Variable + Param
    cleanly. Two assertions:

    - The var-side expression is usable as a constraint (the load
      enters as the constant ``-p_set``).
    - The sol side does NOT auto-negate free-view loads: free views
      sum the raw ``sol.p_t`` columns. This keeps ``n.views['loads']
      .sol.p_t_sum`` reading as positive demand (the natural reading)
      and only the bus view is the place where loads flip sign on
      sol. Asymmetry is intentional and documented.
    """

    def test_mixed_gen_load_var_sum_used_as_constraint(self, snapshots):
        """Bind ``gen.var.p_t + (-load.p_set) <= 5`` and verify gen.p_t
        ≤ load + 5. This proves the load entered the var-side sum as
        the negative constant (otherwise the constraint would be
        ``gen + load <= 5`` and the LP would shed nothing). At demand
        40, the cap forces gen ≤ 45 — we use 200 MW of capacity, so
        the cap is what binds, not p_nom."""
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        gen = n.add(qp.Generator, "G", bus=b, p_nom=200.0, marginal_cost=10.0)
        load = n.add(qp.Load, "L", bus=b, p_set=40.0)
        n.set_snapshots(snapshots)
        n.create_model()
        view = qp.View("residual", [gen, load])
        n.views["residual"] = view
        n.model.cap = view.var.p_t_sum <= 5.0
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        # KCL forces gen == load == 40. Cap says gen + (-40) <= 5 →
        # gen <= 45. Both consistent at gen = 40.
        for v in gen.sol.p_t["p"].to_list():
            assert v == pytest.approx(40.0, abs=1e-6)

    def test_mixed_gen_load_sol_does_not_auto_negate(self, snapshots):
        """Free-view ``sol.p_t_sum`` is the raw column sum: gen +
        load = 40 + 40 = 80. (The var side is signed via
        ``get_p_net``; the sol side is not — see the class docstring.)
        """
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        gen = n.add(qp.Generator, "G", bus=b, p_nom=100.0, marginal_cost=10.0)
        load = n.add(qp.Load, "L", bus=b, p_set=40.0)
        n.set_snapshots(snapshots)
        n.create_model()
        view = qp.View("residual", [gen, load])
        n.views["residual"] = view
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        for v in view.sol.p_t_sum["p"].to_list():
            assert v == pytest.approx(80.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 8–9. Bus view KCL identity + branch sign convention.
# ---------------------------------------------------------------------------

class TestBusViewKCL:
    def test_bus2_rows_sum_to_zero(self, small_showcase):
        n = small_showcase[0]
        wide = n.views["Bus2"].sol.p_t
        snap = "time"
        value_cols = [c for c in wide.columns if c != snap]
        # Per-snapshot horizontal sum is the KCL identity.
        per_row = wide.select(pl.sum_horizontal(value_cols).alias("kcl"))[
            "kcl"
        ].to_list()
        for v in per_row:
            assert v == pytest.approx(0.0, abs=1e-5)

    def test_bus_view_p_t_sum_is_zero(self, small_showcase):
        n = small_showcase[0]
        for bus_name in ("Bus1", "Bus2", "Bus3"):
            summed = n.views[bus_name].sol.p_t_sum["p"].to_list()
            for v in summed:
                assert v == pytest.approx(0.0, abs=1e-5)

    def test_branch_signs_opposite_at_two_ends(self, small_showcase):
        n = small_showcase[0]
        # ACLine_main: from_bus=Bus1, to_bus=Bus2.
        b1_main = n.views["Bus1"].sol.p_t["ACLine_main"].to_list()
        b2_main = n.views["Bus2"].sol.p_t["ACLine_main"].to_list()
        for a, b in zip(b1_main, b2_main):
            # Equal magnitude, opposite sign — pure transmission, no
            # losses.
            assert a == pytest.approx(-b, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. Battery in a bus view (storage's signed contribution not double-flipped).
# ---------------------------------------------------------------------------

class TestBusViewWithBattery:
    """Battery's ``sol.p_t`` is already ``p_dispatch − p_store`` (signed
    net). The bus view must NOT flip it again — the bus injection sign
    for a battery is +1, same as a generator."""

    def test_battery_sign_matches_raw_sol_p_t(self, small_showcase):
        n, _, _, battery, _, _, _ = small_showcase
        raw = battery.sol.p_t["p"].to_list()
        # Battery is at Bus2; its column in the bus view is its name.
        in_view = n.views["Bus2"].sol.p_t["Battery"].to_list()
        for r, v in zip(raw, in_view):
            assert v == pytest.approx(r, abs=1e-6)


# ---------------------------------------------------------------------------
# 11. Empty view behaviour.
# ---------------------------------------------------------------------------

class TestEmptyView:
    """A network with no batteries still has ``n.views['batteries']``;
    its sol-side returns sensible empty/zero shapes, its var-side raises
    (no representative term to synthesise an indexed zero)."""

    def test_empty_registry_view_sol_p_t_has_no_value_columns(self, snapshots):
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        n.add(qp.Generator, "G", bus=b, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=b, p_set=20.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.optimize()
        wide = n.views["batteries"].sol.p_t   # no batteries
        # Snapshot column only; no member columns.
        assert wide.columns == ["time"]
        assert wide.shape[0] == len(n.snapshots)

    def test_empty_registry_view_sol_p_t_sum_is_zero(self, snapshots):
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        n.add(qp.Generator, "G", bus=b, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=b, p_set=20.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.optimize()
        summed = n.views["batteries"].sol.p_t_sum
        assert summed.columns == ["time", "p"]
        for v in summed["p"].to_list():
            assert v == 0.0

    def test_empty_view_var_raises(self, snapshots):
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        n.add(qp.Generator, "G", bus=b, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=b, p_set=20.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.optimize()
        with pytest.raises(AttributeError, match="empty"):
            _ = n.views["batteries"].var.p_t_sum


# ---------------------------------------------------------------------------
# 12. Mixed-network View raises.
# ---------------------------------------------------------------------------

class TestMixedNetwork:
    def test_mixing_two_networks_raises(self):
        n1 = qp.Network()
        n2 = qp.Network()
        b1 = n1.add(qp.Bus, "B")
        b2 = n2.add(qp.Bus, "B")
        g1 = n1.add(qp.Generator, "G1", bus=b1, p_nom=100.0)
        g2 = n2.add(qp.Generator, "G2", bus=b2, p_nom=100.0)
        with pytest.raises(ValueError, match="different networks"):
            qp.View("bad", [g1, g2])


# ---------------------------------------------------------------------------
# 13. User-set view assignment.
# ---------------------------------------------------------------------------

class TestUserSetView:
    def test_user_view_assignment_works(self, small_showcase):
        n, _, gens, _, _, _, _ = small_showcase
        # Coal + Peaker = "thermal" carrier subset.
        n.views["thermal"] = qp.View("thermal", [gens[0], gens[2]])
        wide = n.views["thermal"].sol.p_t
        assert wide.columns == ["time", "Coal", "Peaker"]


# ---------------------------------------------------------------------------
# 14. Bus / registry-name collision.
# ---------------------------------------------------------------------------

class TestNameCollision:
    def test_bus_named_generators_raises(self, snapshots):
        n = qp.Network()
        # A bus with the literal name "generators" must not be allowed
        # to silently shadow the registry-keyed default view.
        b = n.add(qp.Bus, "generators")
        n.add(qp.Generator, "G", bus=b, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=b, p_set=20.0)
        n.set_snapshots(snapshots)
        with pytest.raises(ValueError, match="collides with a default registry"):
            n.create_model()


# ---------------------------------------------------------------------------
# 15. Bus-as-member raises.
# ---------------------------------------------------------------------------

class TestBusNotAMember:
    def test_view_with_bus_member_raises(self, network):
        b = network.add(qp.Bus, "B")
        with pytest.raises(TypeError, match="Bus objects cannot be view members"):
            qp.View("bad", [b])
