"""Infeasibility / failure-mode tests.

The LP must say "infeasible" (and we must surface it) when the user asks
for something physically impossible. These tests pin down how the solver's
verdict propagates through ``Network.optimize()``.

We accept any non-OPTIMAL termination here — HiGHS can report INFEASIBLE
(2) or PRIMAL_INFEASIBLE depending on presolve. The contract is just:
the solve must NOT report OPTIMAL, since that would mean the LP silently
accepted an infeasible setup.
"""
import polars as pl
import pyoptinterface as poi
import pytest

import qwenaplan as qp


def _is_infeasible_status(status):
    """HiGHS may report INFEASIBLE under several termination codes; accept
    anything that isn't OPTIMAL as long as the LP didn't crash."""
    return status != poi.TerminationStatusCode.OPTIMAL


class TestDemandExceedsCapacity:
    def test_load_above_total_generator_capacity(self, snapshots):
        """Load 200 MW served only by a 100 MW generator → infeasible."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=200.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert _is_infeasible_status(n.optimize())

    def test_load_at_remote_bus_no_path(self, snapshots):
        """Load at bus2 cannot be reached from gen at bus1 if there is no
        line nor link between them → infeasible."""
        n = qp.Network()
        bus1 = n.add(qp.Bus, "Bus1")
        bus2 = n.add(qp.Bus, "Bus2")
        n.add(qp.Generator, "G", bus=bus1, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus2, p_set=10.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert _is_infeasible_status(n.optimize())


class TestRampLimitTooTight:
    def test_step_change_exceeds_ramp_capability(self):
        """ramp_limit_up=0.0 means the generator cannot change its output
        at all. If load steps from 10 → 80 there's no way to follow it
        with a single non-rampable generator. Infeasible — even with no
        load profile constraint, because the only generator must produce
        a constant value (= p[0] forever) but demand is varying."""
        load_profile = [10, 80, 80, 80]
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0,
              marginal_cost=10.0,
              ramp_limit_up=0.0, ramp_limit_down=0.0)
        n.add(qp.Load, "L", bus=bus,
              p_set=pl.Series("time", load_profile))
        n.set_snapshots(pl.Series("time", list(range(4))))
        n.create_model()
        assert _is_infeasible_status(n.optimize())


class TestStorageBoundsViolation:
    def test_initial_soc_below_soc_min(self, snapshots):
        """If initial_soc < soc_min, the very first snapshot violates the
        SOC floor unless the storage immediately charges into the band.
        With limited charge power and a load drawing from the bus, this
        forces infeasibility."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=10.0)
        # initial_soc=5 but soc_min=20 → the first snapshot's SOC is
        # bounded below by 20, but the SOC equation says soc(0) =
        # 5 + (p_in - p_out)*Δt. Storage cannot charge by 15 MWh in 1 h
        # if p_nom_in is only 5 MW.
        n.add(qp.StorageUnit, "S", bus=bus,
              e_nom=100.0, p_nom_in=5.0, p_nom_out=5.0,
              eff_in=1.0, eff_out=1.0,
              initial_soc=5.0, soc_min=20.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert _is_infeasible_status(n.optimize())

    def test_initial_soc_above_soc_max(self, snapshots):
        """initial_soc=80, soc_max=50 with no discharge available → cannot
        bring SOC into the band at t=0; LP infeasible."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=10.0)
        n.add(qp.StorageUnit, "S", bus=bus,
              e_nom=100.0, p_nom_in=0.0, p_nom_out=5.0,
              eff_in=1.0, eff_out=1.0,
              initial_soc=80.0, soc_max=50.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert _is_infeasible_status(n.optimize())


class TestModelLifecycleErrors:
    def test_optimize_before_create_model_raises(self, network, snapshots):
        """Calling optimize() before create_model() should raise a clear
        RuntimeError, not an opaque AttributeError on n.model."""
        with pytest.raises(RuntimeError, match="create_model"):
            network.optimize()

    def test_create_model_without_snapshots_raises(self, network):
        with pytest.raises(RuntimeError, match="set_snapshots"):
            network.create_model()
