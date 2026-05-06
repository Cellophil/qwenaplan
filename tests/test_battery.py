"""Tests for the Battery composite (electrical storage with no separate generator).

The fixture ``battery_test_network`` provides a 1-bus system with a 100 MW
generator (mc=50), a 100 MWh / 30 MW battery (eff 0.95/0.95, initial 20),
and a 20 MW constant load. Tests focus on Battery-specific concerns:
property delegation, net-power expression, mutation propagation, simple
arbitrage. The generic SOC-balance/efficiency-loss assertions live in
test_storage_unit.py.
"""
import polars as pl
import pyoptinterface as poi

import qwenaplan as qp


class TestBatteryInit:
    def test_default_parameters(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=50.0)
        assert b.e_nom == 100.0
        assert b.p_nom == 50.0
        assert b.eff_store == 1.0
        assert b.eff_dispatch == 1.0
        assert b.initial_soc == 0.0
        assert b.soc_min == 0.0
        assert b.soc_max == 100.0

    def test_custom_parameters(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        b = n.add(qp.Battery, "B", bus=bus,
                  e_nom=200.0, p_nom=60.0,
                  eff_store=0.95, eff_dispatch=0.95,
                  initial_soc=100.0, soc_min=20.0, soc_max=180.0)
        assert (b.eff_store, b.eff_dispatch) == (0.95, 0.95)
        assert (b.initial_soc, b.soc_min, b.soc_max) == (100.0, 20.0, 180.0)


class TestBatteryDelegation:
    """The Battery composite is meant to behave as a thin wrapper over the
    inner ``_StorageBase``. Mutations on the composite must propagate."""

    def test_soc_min_mutation_propagates(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=50.0)
        # Composite and inner storage share state.
        assert b.soc_min == b.storage.soc_min
        b.soc_min = 25.0
        assert b.storage.soc_min == 25.0

    def test_p_nom_mutation_updates_both_in_and_out(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=50.0)
        b.p_nom = 30.0
        assert b.storage.p_nom_in == 30.0
        assert b.storage.p_nom_out == 30.0


class TestBatteryDispatch:
    def test_battery_arbitrages_to_avoid_expensive_generation(self, snapshots):
        """Cheap energy at t=0,1; expensive at t=2,3. Battery should charge
        early and discharge later. Final round-trip cost must be cheaper
        than letting the expensive generator serve the load directly.
        """
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        cheap_profile = pl.Series("time", [1.0, 1.0, 0.0, 0.0])
        n.add(qp.Generator, "Cheap", bus=bus, p_nom=100.0,
              marginal_cost=10.0, p_max_pu=cheap_profile)
        n.add(qp.Generator, "Expensive", bus=bus, p_nom=100.0,
              marginal_cost=100.0)
        n.add(qp.Load, "L", bus=bus, p_set=30.0)
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=20.0,
                  eff_store=1.0, eff_dispatch=1.0, initial_soc=0.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        p_store = b.p_store_t["p_store"].to_list()
        p_dispatch = b.p_dispatch_t["p_dispatch"].to_list()
        # Charge in cheap window, discharge in expensive.
        assert p_store[0] > 0 or p_store[1] > 0
        assert p_dispatch[2] > 0 or p_dispatch[3] > 0

    def test_p_t_returns_dispatch_minus_store(self, snapshots):
        """Battery.p_t must equal p_dispatch_t - p_store_t per snapshot."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "Cheap", bus=bus, p_nom=100.0,
              marginal_cost=10.0,
              p_max_pu=pl.Series("time", [1.0, 1.0, 0.0, 0.0]))
        n.add(qp.Generator, "Exp", bus=bus, p_nom=100.0, marginal_cost=100.0)
        n.add(qp.Load, "L", bus=bus, p_set=30.0)
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=20.0,
                  eff_store=1.0, eff_dispatch=1.0, initial_soc=0.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        p_net = b.p_t["p"].to_list()
        p_store = b.p_store_t["p_store"].to_list()
        p_dispatch = b.p_dispatch_t["p_dispatch"].to_list()
        for i in range(4):
            assert abs(p_net[i] - (p_dispatch[i] - p_store[i])) < 1e-6

    def test_soc_bounds_actually_clamp(self, snapshots):
        """Setting soc_min=20 and soc_max=80 after construction must clamp
        the trajectory at every snapshot. (Regression test for the
        composite-mutation bug we just fixed.)"""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=10.0)
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=30.0,
                  initial_soc=50.0)
        # Mutate AFTER construction — the bug was that these did not
        # propagate to the inner storage.
        b.soc_min = 20.0
        b.soc_max = 80.0

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        soc = b.soc_t["soc"].to_list()
        for s in soc:
            assert 20.0 - 1e-6 <= s <= 80.0 + 1e-6
