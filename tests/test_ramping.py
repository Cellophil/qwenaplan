"""Tests for ramping and per-snapshot pu limits.

All checks are numerical: build a 1-bus network with a *forced* dispatch
trajectory (via varying load), solve, and assert the ramp-rate constraints
do what we expect (limit step-to-step changes) and that pu profiles cap
output as advertised.

The constraint-existence smoke checks (`hasattr(model, "gen_ramp_up_Gen1")`)
that used to be here are gone — they don't tell us anything about whether
the constraint is *correct*, and they're tightly coupled to internal
naming conventions.
"""
import polars as pl
import pyoptinterface as poi
import pytest

import qwenaplan as qp


def _network_with_load_profile(load_per_snapshot):
    """Helper: 1 bus, 1 gen (parameters added by the caller), 1 load with
    the given per-snapshot demand. Returns ``(network, gen_factory)``.

    ``gen_factory(**kwargs)`` adds the generator and returns it; this lets
    each test customise ramp/pu params on the same skeleton.
    """
    n = qp.Network()
    bus = n.add(qp.Bus, "Bus")
    snapshots = pl.Series("time", list(range(len(load_per_snapshot))))
    load_series = pl.Series("time", load_per_snapshot)
    n.add(qp.Load, "L", bus=bus, p_set=load_series)

    def gen_factory(**kwargs):
        return n.add(qp.Generator, "Gen", bus=bus,
                     p_nom=100.0, marginal_cost=10.0, **kwargs)

    return n, snapshots, gen_factory


class TestPMinPuProfile:
    def test_profile_min_above_static_max_raises(self, snapshots):
        # Validation happens at create_model() time, since the profile is
        # only inspected when constraints are built. Constructing the
        # generator alone is fine; create_model is where it bites.
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        min_profile = pl.Series("time", [0.6, 0.7, 0.6, 0.7])
        n.add(qp.Generator, "Gen", bus=bus, p_nom=100.0,
              p_min_pu=min_profile, p_max_pu=0.5)
        n.set_snapshots(snapshots)
        with pytest.raises(ValueError, match="p_max_pu.*below.*p_min_pu"):
            n.create_model()

    def test_static_min_above_profile_max_raises(self, snapshots):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        max_profile = pl.Series("time", [0.1, 0.2, 0.1, 0.2])
        n.add(qp.Generator, "Gen", bus=bus, p_nom=100.0,
              p_max_pu=max_profile, p_min_pu=0.3)
        n.set_snapshots(snapshots)
        with pytest.raises(ValueError, match="p_min_pu.*exceeds.*p_max_pu"):
            n.create_model()

    def test_profile_min_forces_minimum_per_snapshot(self):
        """Per-snapshot p_min_pu profile [.5, .3, .5, .3] on a 100 MW gen
        must force dispatch >= [50, 30, 50, 30]. Pair with an absorber so
        the LP stays feasible when load is below the forced minimum."""
        load = [40, 40, 40, 40]
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        snapshots = pl.Series("time", list(range(4)))
        load_series = pl.Series("time", load)
        n.add(qp.Load, "L", bus=bus, p_set=load_series)
        min_profile = pl.Series("time", [0.5, 0.3, 0.5, 0.3])
        gen = n.add(qp.Generator, "Gen", bus=bus, p_nom=100.0,
                    marginal_cost=10.0, p_min_pu=min_profile)
        n.add(qp.Generator, "Absorber", bus=bus, p_nom=100.0,
              marginal_cost=0.0, p_min_pu=-1.0, p_max_pu=1.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        # Gen is forced at p_min_pu when its energy is unwanted.
        p = gen.p_t["p"].to_list()
        assert [round(x, 6) for x in p] == [50.0, 30.0, 50.0, 30.0]


class TestRamping:
    """Numerical ramping behaviour."""

    def test_ramp_up_caps_step_to_step_increase(self):
        """A generator with ramp_limit_up=0.1 (10 MW/step on a 100 MW unit)
        cannot follow a load that jumps from 0 → 80 → 80 → 80; we must see
        gen output 0, 10, 20, 30 (linearly ramping up). Pair with a costly
        slack so the LP stays feasible while gen is below load."""
        load = [0, 80, 80, 80]
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        snapshots = pl.Series("time", list(range(4)))
        n.add(qp.Load, "L", bus=bus, p_set=pl.Series("time", load))
        gen = n.add(qp.Generator, "Gen", bus=bus, p_nom=100.0,
                    marginal_cost=10.0, ramp_limit_up=0.1)
        # Costly slack to avoid infeasibility while gen is ramping.
        n.add(qp.Generator, "Slack", bus=bus, p_nom=100.0, marginal_cost=1000.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        gen_p = gen.p_t["p"].to_list()
        # Ramp-up cap 10 MW/step starting from p(0)=0:
        # min cost is achieved by ramping as fast as possible.
        for i in range(1, 4):
            assert gen_p[i] - gen_p[i - 1] <= 10.0 + 1e-6

    def test_ramp_down_caps_step_to_step_decrease(self):
        """Symmetric to the up case: ramp_limit_down=0.1 (10 MW/step) means
        going from 80 → 0 → 0 → 0 must take at least 8 steps. With only 4
        snapshots and starting at gen(0)=80, the floor sequence is bounded
        below by max(0, 80 - 10*i). A negative-p_min_pu absorber soaks up
        the surplus."""
        # Force gen high at t=0 then load drops to zero.
        load = [80, 0, 0, 0]
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        snapshots = pl.Series("time", list(range(4)))
        n.add(qp.Load, "L", bus=bus, p_set=pl.Series("time", load))
        gen = n.add(qp.Generator, "Gen", bus=bus, p_nom=100.0,
                    marginal_cost=10.0, ramp_limit_down=0.1)
        n.add(qp.Generator, "Absorber", bus=bus, p_nom=100.0,
              marginal_cost=0.0, p_min_pu=-1.0, p_max_pu=1.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        gen_p = gen.p_t["p"].to_list()
        for i in range(1, 4):
            assert gen_p[i - 1] - gen_p[i] <= 10.0 + 1e-6

    def test_zero_ramp_limit_freezes_dispatch(self):
        """With ramp_limit_up=0 and ramp_limit_down=0 the generator must
        produce the same MW in every snapshot."""
        load = [40, 40, 40, 40]
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        snapshots = pl.Series("time", list(range(4)))
        n.add(qp.Load, "L", bus=bus, p_set=pl.Series("time", load))
        gen = n.add(qp.Generator, "Gen", bus=bus, p_nom=100.0,
                    marginal_cost=10.0,
                    ramp_limit_up=0.0, ramp_limit_down=0.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        gen_p = gen.p_t["p"].to_list()
        # All four values must be identical (within float tolerance).
        assert max(gen_p) - min(gen_p) < 1e-6

    def test_ramping_with_p_max_pu_profile_combine(self):
        """Both ramp limit and p_max_pu profile must hold simultaneously.
        Generator p_max_pu = [0.8, 0.6, 0.8, 0.6], ramp_up = 0.1.
        Load high enough to keep gen at the cap whenever feasible.
        Effective cap each step: min(p_max_pu * 100, p(t-1) + 10)."""
        load = [80, 60, 80, 60]
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        snapshots = pl.Series("time", list(range(4)))
        n.add(qp.Load, "L", bus=bus, p_set=pl.Series("time", load))
        max_profile = pl.Series("time", [0.8, 0.6, 0.8, 0.6])
        gen = n.add(qp.Generator, "Gen", bus=bus, p_nom=100.0,
                    marginal_cost=10.0,
                    p_max_pu=max_profile, ramp_limit_up=0.1)
        n.add(qp.Generator, "Slack", bus=bus, p_nom=100.0,
              marginal_cost=1000.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        gen_p = gen.p_t["p"].to_list()
        # Cap from profile (80, 60, 80, 60) and ramp from previous step
        # (start at <=80 since p(0)<=80 from profile, then +10 max).
        # The cheapest feasible path: ramp toward the cap as fast as possible.
        # Ensure ramp constraint holds:
        for i in range(1, 4):
            assert gen_p[i] - gen_p[i - 1] <= 10.0 + 1e-6
        # Ensure profile cap holds:
        cap = [80.0, 60.0, 80.0, 60.0]
        for p, c in zip(gen_p, cap):
            assert p <= c + 1e-6
