"""Tests for committable Generator: numerical UC, costs, ``status_t`` shape.

Two big things this file covers:

1. **Always-present ``status_t``**: every Generator carries a
   ``var.status_t`` (Param-of-1s when not committable, binary Variable
   when committable). The sol container synthesises a uniform
   ``(snap, status)`` DataFrame either way so downstream views don't
   branch on ``committable``.

2. **Numerical unit commitment**: small dispatch problems where the
   correct answer hinges on the binary status, the start-up / shut-down
   cost, and PyPSA's ``standby_cost`` (qp's ``cost_when_active``).
   The user explicitly asked for these numbers — they're the proof
   that the LP/MILP boundary is wired right.
"""
import polars as pl
import pyoframe as pf
import pyoptinterface as poi
import pytest

import qwenaplan as qp


# ---------------------------------------------------------------------------
# Init / validation
# ---------------------------------------------------------------------------

class TestCommittableInit:
    def test_defaults(self, network):
        b = network.add(qp.Bus, "B")
        g = network.add(qp.Generator, "G", bus=b, p_nom=10.0)
        assert g.committable is False
        assert g.start_up_cost == 0.0
        assert g.shut_down_cost == 0.0
        assert g.cost_when_active == 0.0

    @pytest.mark.parametrize("attr", ["start_up_cost", "shut_down_cost", "cost_when_active"])
    def test_negative_costs_raise(self, network, attr):
        b = network.add(qp.Bus, "B")
        with pytest.raises(ValueError, match=attr):
            network.add(qp.Generator, "G", bus=b, p_nom=10.0, **{attr: -1.0})


# ---------------------------------------------------------------------------
# status_t shape: Param-of-1s vs binary Variable
# ---------------------------------------------------------------------------

class TestStatusTAlwaysPresent:
    def test_var_type(self, snapshots, network):
        b = network.add(qp.Bus, "B")
        # Plain LP generator
        g_lp = network.add(qp.Generator, "G_lp", bus=b, p_nom=10.0)
        # Committable MILP generator
        g_uc = network.add(qp.Generator, "G_uc", bus=b, p_nom=10.0, committable=True)
        network.set_snapshots(snapshots)
        # ``status_t`` exists on both, but is a different pyoframe type.
        from pyoframe import Param, Variable
        # ``pf.Param`` is a function returning an Expression; we accept
        # whichever object pyoframe yields, just not a Variable.
        assert not isinstance(g_lp.var.status_t, Variable)
        assert isinstance(g_uc.var.status_t, Variable)

    def test_status_t_returns_ones_when_not_committable(self, snapshots):
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        g = n.add(qp.Generator, "G", bus=b, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=b, p_set=20.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        # All-1.0 across the snapshot horizon.
        assert g.sol.status_t["status"].to_list() == [1.0] * len(snapshots)

    def test_status_t_returns_binary_when_committable(self, snapshots):
        # Cheap committable but with a high p_min_pu that the load can't
        # absorb → solver must shut it off in some snapshots.
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        cheap = n.add(qp.Generator, "cheap", bus=b, p_nom=100.0, p_min_pu=0.5,
                      marginal_cost=10.0, committable=True)
        n.add(qp.Generator, "exp", bus=b, p_nom=200.0, marginal_cost=50.0)
        n.add(qp.Load, "L", bus=b, p_set=10.0)  # below cheap's min
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        # Cheap can't run: would need ≥50 MW out, only 10 needed.
        assert cheap.sol.status_t["status"].to_list() == [0] * len(snapshots)


# ---------------------------------------------------------------------------
# Dispatch behaviour
# ---------------------------------------------------------------------------

class TestCommittableDispatch:
    """The numerical proof points the user asked for, line by line."""

    def _shared_two_unit_network(self, snapshots, *, p_set, **cheap_kwargs):
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        cheap = n.add(
            qp.Generator, "cheap", bus=b,
            p_nom=100.0, p_min_pu=0.4, marginal_cost=10.0,
            committable=True,
            **cheap_kwargs,
        )
        exp = n.add(qp.Generator, "exp", bus=b, p_nom=200.0, marginal_cost=50.0)
        n.add(qp.Load, "L", bus=b, p_set=p_set)
        n.set_snapshots(snapshots)
        return n, cheap, exp

    def test_low_load_shuts_off_cheap_unit(self, snapshots):
        # 4 snapshots, load=30 each, cheap's min=40. Cheap must stay
        # off — the start-up cost would be wasted relative to running
        # the expensive unit.
        n, cheap, exp = self._shared_two_unit_network(
            snapshots, p_set=30.0, start_up_cost=500.0,
        )
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        assert cheap.sol.status_t["status"].to_list() == [0] * 4
        assert cheap.sol.p_t["p"].to_list() == [0.0] * 4
        assert exp.sol.p_t["p"].to_list() == [30.0] * 4
        # objective = 4 * 30 * 50 = 6000 (no start-up cost paid)
        assert n.objective_value == pytest.approx(6000.0)

    def test_high_load_runs_cheap_at_min_with_one_startup(self, snapshots):
        # load=50 > cheap's min (40). Cheap takes all 4 snapshots; one
        # start-up at t=0 (previously-OFF assumption).
        n, cheap, exp = self._shared_two_unit_network(
            snapshots, p_set=50.0, start_up_cost=500.0,
        )
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        assert cheap.sol.status_t["status"].to_list() == [1] * 4
        assert cheap.sol.p_t["p"].to_list() == [50.0] * 4
        assert exp.sol.p_t["p"].to_list() == [0.0] * 4
        # objective = 4 * 50 * 10 + 1 * 500 = 2500
        assert n.objective_value == pytest.approx(2500.0)
        # The start-up var fires exactly once at t=0.
        assert cheap.sol.start_up_t["start_up"].to_list() == [1.0, 0.0, 0.0, 0.0]

    def test_shut_down_cost_at_load_drop(self):
        # Load profile [80, 80, 0, 0]: cheap is on for two snapshots,
        # off for two. Exactly one shut-down at t=2.
        load_profile = pl.Series("time", [80.0, 80.0, 0.0, 0.0])
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        cheap = n.add(qp.Generator, "cheap", bus=b, p_nom=100.0, p_min_pu=0.4,
                      marginal_cost=10.0, committable=True,
                      shut_down_cost=200.0)
        n.add(qp.Generator, "exp", bus=b, p_nom=200.0, marginal_cost=50.0)
        n.add(qp.Load, "L", bus=b, p_set=load_profile)
        n.set_snapshots(pl.Series("time", [0, 1, 2, 3]))
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        assert cheap.sol.status_t["status"].to_list() == [1, 1, 0, 0]
        assert cheap.sol.shut_down_t["shut_down"].to_list() == [0.0, 0.0, 1.0, 0.0]


class TestCostWhenActive:
    """``cost_when_active`` (PyPSA's ``standby_cost``) is charged every
    snapshot the unit is online — multiplied by ``status_t``."""

    def test_standby_cost_charged_per_active_snapshot(self):
        load_profile = pl.Series("time", [50.0, 50.0, 0.0, 0.0])
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        cheap = n.add(qp.Generator, "cheap", bus=b, p_nom=100.0, p_min_pu=0.4,
                      marginal_cost=10.0, committable=True,
                      cost_when_active=100.0, start_up_cost=0.0)
        n.add(qp.Generator, "exp", bus=b, p_nom=200.0, marginal_cost=50.0)
        n.add(qp.Load, "L", bus=b, p_set=load_profile)
        n.set_snapshots(pl.Series("time", [0, 1, 2, 3]))
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        assert cheap.sol.status_t["status"].to_list() == [1, 1, 0, 0]
        # objective = 2 * 50 * 10  + 2 * 100  = 1200
        #             ↑ marginal     ↑ standby (2 active snapshots)
        assert n.objective_value == pytest.approx(1200.0)

    def test_standby_cost_with_non_committable_charges_constantly(self, snapshots):
        # Edge case the docstring warns about: with committable=False,
        # status is always 1 → cost_when_active fires every snapshot
        # as a flat constant.
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        n.add(qp.Generator, "G", bus=b, p_nom=100.0, marginal_cost=0.0,
              cost_when_active=50.0)
        n.add(qp.Load, "L", bus=b, p_set=10.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        # 4 snapshots * 50 = 200
        assert n.objective_value == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Views: ``views['generators'].sol.status_t`` aggregates uniformly
# ---------------------------------------------------------------------------

class TestCommittableViews:
    def test_views_generators_status_t_wide(self, snapshots):
        """Mixing a non-committable and a committable generator in the
        same network: the wide ``views['generators'].sol.status_t``
        DataFrame should have one column per generator, all aligned on
        the snapshot index."""
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        # Plain LP generator: status_t is the Param-of-1s.
        n.add(qp.Generator, "always_on", bus=b, p_nom=100.0, marginal_cost=20.0)
        # Committable MILP generator: status_t is the binary Variable.
        # Force it OFF by making it expensive + low-load.
        n.add(qp.Generator, "uc", bus=b, p_nom=100.0, p_min_pu=0.5,
              marginal_cost=99.0, committable=True, start_up_cost=1000.0)
        n.add(qp.Load, "L", bus=b, p_set=10.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.optimize()

        wide = n.views["generators"].sol.status_t
        # one snapshot col + one column per generator
        assert {"always_on", "uc"}.issubset(wide.columns)
        assert wide["always_on"].to_list() == [1.0] * 4
        assert wide["uc"].to_list() == [0] * 4

    def test_views_generators_status_t_sum(self, snapshots):
        """``status_t_sum`` is the per-snapshot count of online units."""
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        n.add(qp.Generator, "g1", bus=b, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Generator, "g2", bus=b, p_nom=100.0, marginal_cost=20.0)
        n.add(qp.Load, "L", bus=b, p_set=10.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.optimize()
        # Both non-committable → both online every snapshot → sum = 2.
        long = n.views["generators"].sol.status_t_sum
        assert long["status"].to_list() == [2.0] * 4


# ---------------------------------------------------------------------------
# StorageComposite refuses a committable inner generator
# ---------------------------------------------------------------------------

class TestStorageCompositeRefuses:
    def test_battery_inner_generator_not_committable(self, network):
        bus = network.add(qp.Bus, "B")
        bat = network.add(qp.Battery, "Bat", bus=bus, e_nom=100.0, p_nom=20.0)
        # Inner generator built by the composite — should always be False.
        assert bat._generator.committable is False

    def test_post_hoc_committable_assertion_fires(self, snapshots):
        n = qp.Network()
        b = n.add(qp.Bus, "B")
        bat = n.add(qp.Battery, "Bat", bus=b, e_nom=100.0, p_nom=20.0)
        n.add(qp.Generator, "G", bus=b, p_nom=10.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=b, p_set=5.0)
        # Mutate post-hoc — the assertion in setup_constraints catches it.
        bat._generator.committable = True
        n.set_snapshots(snapshots)
        with pytest.raises(AssertionError, match="must not be committable"):
            n.create_model()


# ---------------------------------------------------------------------------
# Importer: PyPSA standby_cost → qp cost_when_active
# ---------------------------------------------------------------------------

class TestImporter:
    def test_pypsa_standby_cost_imports_as_cost_when_active(self):
        class _Source:
            def __init__(self):
                buses = {"B1": {"v_nom": 1.0}}
                gens = {"G": {
                    "bus": "B1",
                    "p_nom": 100.0,
                    "marginal_cost": 5.0,
                    "committable": True,
                    "start_up_cost": 50.0,
                    "shut_down_cost": 25.0,
                    "standby_cost": 42.0,
                }}
                self.Bus = self.bus = buses
                self.Generator = self.generator = gens
                empty: dict = {}
                self.Load = self.load = empty
                self.Line = self.line = empty
                self.Link = self.link = empty
                self.Transformer = self.transformer = empty
                self.StorageUnit = self.storage_unit = empty
                self.Store = self.store = empty
                self.ShuntImpedance = empty
                self.GlobalConstraint = empty
                self.snapshots = list(range(2))

        importer = qp.PyPSAImporter(_Source(), strict_mode=False)
        target = importer.import_network()
        gen = target.generators["G"]
        assert gen.committable is True
        assert gen.start_up_cost == 50.0
        assert gen.shut_down_cost == 25.0
        # PyPSA's ``standby_cost`` was renamed to qp's ``cost_when_active``.
        assert gen.cost_when_active == 42.0
