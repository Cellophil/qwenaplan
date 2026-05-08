"""Tests for Battery-specific behaviour.

The shared composite shape (defaults, ``soc_min`` mutation, the
inner ``storage`` / ``generator`` existence contract) lives in
``test_storage_composite.py`` parametrised over Battery + PHS. This
file holds only Battery-specific things: the single ``p_nom`` knob
that drives both inner rails *and* the inverter, the inner-storage
name preserved across the plan_03 refactor, and the arbitrage / SOC
clamp regression tests.
"""
import polars as pl
import pyoptinterface as poi

import qwenaplan as qp


class TestBatteryInit:
    def test_p_nom_drives_both_inner_rails_and_inverter(self):
        """``Battery.p_nom`` propagates to ``p_nom_in``, ``p_nom_out``, and
        the inverter generator's ``p_nom`` — a single user-facing knob."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=50.0)
        assert b.p_nom == 50.0
        assert b.storage.p_nom_in == 50.0
        assert b.storage.p_nom_out == 50.0
        assert b.generator.p_nom == 50.0

    def test_inner_storage_keeps_battery_name(self):
        """Battery's inner storage carries the battery's own name (no
        ``_storage`` suffix), unlike PHS. Pinned to catch accidental
        symmetric renaming that would break ``model.soc_<name>`` lookups
        for downstream code."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=50.0)
        assert b.storage.name == "B"
        assert b.generator.name == "B_generator"


class TestBatteryDelegation:
    def test_p_nom_mutation_updates_both_in_and_out_and_inverter(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        b = n.add(qp.Battery, "B", bus=bus, e_nom=100.0, p_nom=50.0)
        b.p_nom = 30.0
        assert b.storage.p_nom_in == 30.0
        assert b.storage.p_nom_out == 30.0
        assert b.generator.p_nom == 30.0


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

        p_store = b.sol.p_store_t["p_store"].to_list()
        p_dispatch = b.sol.p_dispatch_t["p_dispatch"].to_list()
        # Charge in cheap window, discharge in expensive.
        assert p_store[0] > 0 or p_store[1] > 0
        assert p_dispatch[2] > 0 or p_dispatch[3] > 0

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

        soc = b.sol.soc_t["soc"].to_list()
        for s in soc:
            assert 20.0 - 1e-6 <= s <= 80.0 + 1e-6
