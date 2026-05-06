"""Tests for StorageUnit (the generic, non-composite storage primitive).

Init/repr stay; SOC behaviour is checked numerically against the analytical
balance equation. The fixture ``storage_test_network`` provides a 1-bus
system with a generator (mc=50), a 100 MWh storage, and a 20 MW load.
"""
import polars as pl
import pyoptinterface as poi

import qwenaplan as qp


class TestStorageUnitInit:
    def test_default_parameters(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0)
        assert s.e_nom == 100.0
        assert s.eff_in == 1.0
        assert s.eff_out == 1.0
        assert s.initial_soc == 0.0
        assert s.soc_min == 0.0
        assert s.soc_max == 100.0
        assert s._influx == 0.0

    def test_custom_parameters(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        s = n.add(qp.StorageUnit, "S", bus=bus,
                  e_nom=200.0, p_nom_in=50.0, p_nom_out=60.0,
                  eff_in=0.9, eff_out=0.95,
                  initial_soc=100.0, soc_min=20.0, soc_max=180.0, influx=5.0)
        assert (s.p_nom_in, s.p_nom_out) == (50.0, 60.0)
        assert (s.eff_in, s.eff_out) == (0.9, 0.95)
        assert (s.initial_soc, s.soc_min, s.soc_max) == (100.0, 20.0, 180.0)
        assert s._influx == 5.0


class TestStorageUnitInflux:
    def test_constant_influx_broadcast_to_series(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0, influx=5.0)
        n.set_snapshots(pl.Series("time", list(range(5))))
        assert list(s._influx_series) == [5.0] * 5

    def test_set_influx_profile_replaces_constant(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0, influx=0.0)
        s.set_influx_profile(pl.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        n.set_snapshots(pl.Series("time", list(range(5))))
        assert list(s._influx_series) == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestStorageSOCDynamics:
    """SOC balance: soc(t+1) = soc(t) + (p_in*eff_in - p_out/eff_out) * Δt."""

    def test_initial_soc_constraint_holds(self, snapshots):
        """The SOC trajectory must satisfy the analytical balance equation
        at every step, including t=0 anchored to ``initial_soc``."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=20.0)
        # Eff 1 keeps the equation simple to verify analytically.
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0,
                  p_nom_in=10.0, p_nom_out=10.0,
                  eff_in=1.0, eff_out=1.0,
                  initial_soc=30.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        soc = s.sol.soc_t["soc"].to_list()
        p_in = s.sol.p_in_t["p_in"].to_list()
        p_out = s.sol.p_out_t["p_out"].to_list()

        # t=0: soc(0) = initial_soc + (p_in(0) - p_out(0)) * Δt
        assert abs(soc[0] - (30.0 + p_in[0] - p_out[0])) < 1e-6
        # t>=1: soc(t) = soc(t-1) + (p_in(t) - p_out(t)) * Δt
        for t in range(1, 4):
            expected = soc[t - 1] + p_in[t] - p_out[t]
            assert abs(soc[t] - expected) < 1e-6

    def test_storage_arbitrages_charge_then_discharge(self, snapshots):
        """Cheap energy at t=0,1; expensive at t=2,3. Storage should charge
        early, discharge later. Total energy balance must close on eff_in*eff_out."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        # Time-varying gen marginal cost via two gens: cheap one with a
        # p_max_pu profile that makes it cheap-then-unavailable.
        cheap_profile = pl.Series("time", [1.0, 1.0, 0.0, 0.0])
        n.add(qp.Generator, "Cheap", bus=bus, p_nom=100.0,
              marginal_cost=10.0, p_max_pu=cheap_profile)
        n.add(qp.Generator, "Expensive", bus=bus, p_nom=100.0,
              marginal_cost=100.0)
        # Constant load 30 MW.
        n.add(qp.Load, "L", bus=bus, p_set=30.0)
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0,
                  p_nom_in=20.0, p_nom_out=20.0,
                  eff_in=1.0, eff_out=1.0,
                  initial_soc=10.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        p_in = s.sol.p_in_t["p_in"].to_list()
        p_out = s.sol.p_out_t["p_out"].to_list()
        soc = s.sol.soc_t["soc"].to_list()
        # Charge at t=0,1 (cheap), discharge at t=2,3 (expensive).
        assert p_in[0] > 0 or p_in[1] > 0
        assert p_out[2] > 0 or p_out[3] > 0
        # Symmetric efficiencies → energy in equals net dispatched + ΔSOC.
        # Net change in SOC over horizon = initial_soc - final_soc consumed
        # plus any net inflow:  Σ p_in - Σ p_out = soc[-1] - initial_soc
        assert abs(sum(p_in) - sum(p_out) - (soc[-1] - 10.0)) < 1e-6

    def test_efficiency_loss_in_round_trip(self, snapshots):
        """With eff_in=0.8, charging 10 MW for 1 h adds 8 MWh to SOC."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        # Cheap-then-disappears gen plus expensive backup.
        n.add(qp.Generator, "Free", bus=bus, p_nom=100.0,
              marginal_cost=0.0,
              p_max_pu=pl.Series("time", [1.0, 0.0, 0.0, 0.0]))
        n.add(qp.Generator, "Expensive", bus=bus, p_nom=100.0,
              marginal_cost=100.0)
        n.add(qp.Load, "L", bus=bus, p_set=10.0)  # 10 MW constant
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0,
                  p_nom_in=20.0, p_nom_out=20.0,
                  eff_in=0.8, eff_out=1.0,
                  initial_soc=0.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        # At t=0 the optimal play is: free gen serves load (10 MW) AND
        # charges as much as it can — 20 MW total → 10 MW into storage at
        # eff_in 0.8 = 8 MWh added to SOC.
        soc = s.sol.soc_t["soc"].to_list()
        p_in = s.sol.p_in_t["p_in"].to_list()
        # SOC at t=0 = initial + p_in[0]*eff_in*Δt = 0 + 10*0.8*1 = 8
        assert abs(soc[0] - p_in[0] * 0.8) < 1e-6


class TestStorageBoundsAreEnforced:
    def test_soc_min_clamps_floor(self, storage_test_network, snapshots):
        """soc_min=20 must be respected even under heavy discharge demand."""
        n, _, _, storage, load = storage_test_network
        storage.initial_soc = 30.0
        storage.soc_min = 20.0
        storage.soc_max = 100.0
        # Add a backstop generator so the LP has a way out when storage hits
        # its floor — otherwise the load forces violation and the LP is
        # infeasible (which is also a valid sentinel, but here we're
        # checking the variable bound, not the feasibility wall).
        n.add(qp.Generator, "Backstop", bus=storage.bus, p_nom=100.0,
              marginal_cost=1000.0)
        # Bigger demand so the optimizer wants to drain storage.
        load._p_set = 40.0
        load._p_set_profile = None

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        soc = storage.sol.soc_t["soc"].to_list()
        for s in soc:
            assert s >= 20.0 - 1e-6
