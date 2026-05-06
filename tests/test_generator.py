"""Tests for the Generator component.

Init/repr smoke tests are kept short. Constraint-existence checks have
been replaced with end-to-end numerical assertions against analytical
solutions: the dispatch you'd compute by hand is what the LP returns.
"""
import polars as pl
import pyoptinterface as poi
import pytest

import qwenaplan as qp


class TestGeneratorInit:
    def test_default_parameters(self, network):
        bus = network.add(qp.Bus, "Bus1")
        gen = network.add(qp.Generator, "Gen1", bus=bus)
        assert gen.bus is bus
        assert gen.p_nom == 0.0
        assert gen.marginal_cost == 0.0
        assert gen.carrier == ""

    def test_custom_parameters(self, network):
        bus = network.add(qp.Bus, "Bus1")
        gen = network.add(qp.Generator, "Gen1", bus=bus,
                          p_nom=100.0, marginal_cost=50.0, carrier="solar")
        assert gen.p_nom == 100.0
        assert gen.marginal_cost == 50.0
        assert gen.carrier == "solar"

    def test_repr(self, network):
        bus = network.add(qp.Bus, "Bus1")
        gen = network.add(qp.Generator, "Gen1", bus=bus, p_nom=100.0)
        assert repr(gen) == "<Generator(name=Gen1, bus=Bus1, p_nom=100.0, marginal_cost=0.0)>"


class TestGeneratorValidation:
    def test_p_min_pu_above_p_max_pu_static_raises(self, network):
        bus = network.add(qp.Bus, "Bus1")
        with pytest.raises(ValueError, match="p_min_pu.*must be <= p_max_pu"):
            network.add(qp.Generator, "Gen1", bus=bus, p_nom=100.0,
                        p_min_pu=0.8, p_max_pu=0.2)


class TestGeneratorDispatch:
    """Numerical dispatch tests against analytical truth.

    Each test sets up a 1-bus network with two generators and a load. The
    optimal dispatch is straightforward to compute by hand; the LP must
    reproduce it.
    """

    def _build(self, snapshots, gen_specs, load_p_set):
        """Helper: 1 bus, N generators with the given (p_nom, marginal_cost,
        **extras) tuples, one constant Load. Returns ``(network, gens, load)``."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        gens = []
        for i, (p_nom, mc, *extra) in enumerate(gen_specs, start=1):
            kwargs = dict(p_nom=p_nom, marginal_cost=mc)
            if extra:
                kwargs.update(extra[0])
            g = n.add(qp.Generator, f"Gen{i}", bus=bus, **kwargs)
            gens.append(g)
        load = n.add(qp.Load, "Load", bus=bus, p_set=load_p_set)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        return n, gens, load

    def test_cheap_gen_serves_load_until_capped(self, snapshots):
        """Demand = 70 MW, cheap gen capped at 50 MW. Cheap is fully used,
        the rest comes from the expensive one."""
        n, (cheap, expensive), _ = self._build(
            snapshots,
            [(50.0, 10.0), (100.0, 50.0)],
            load_p_set=70.0,
        )
        assert cheap.p_t["p"].to_list() == [50.0] * 4
        assert expensive.p_t["p"].to_list() == [20.0] * 4
        # Objective: 4 snapshots × (50×10 + 20×50) = 4 × 1500 = 6000.
        assert n.objective_value == 6000.0

    def test_p_max_pu_caps_output_below_p_nom(self, snapshots):
        """Demand = 70 MW. Cheap gen has p_nom=100 but p_max_pu=0.5 so it
        can deliver only 50 MW; expensive must cover the remaining 20 MW."""
        n, (cheap, expensive), _ = self._build(
            snapshots,
            [(100.0, 10.0, {"p_max_pu": 0.5}),
             (100.0, 50.0)],
            load_p_set=70.0,
        )
        assert cheap.p_t["p"].to_list() == [50.0] * 4
        assert expensive.p_t["p"].to_list() == [20.0] * 4

    def test_p_min_pu_forces_minimum_output(self, snapshots):
        """If a cheap gen has p_min_pu=0.3 and load only 20 MW, the cheap gen
        is *forced* to produce 30 MW; the surplus has to go somewhere — only
        the second generator can absorb (with p_min_pu < 0)."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        cheap = n.add(qp.Generator, "Cheap", bus=bus, p_nom=100.0,
                      marginal_cost=10.0, p_min_pu=0.3)
        # An "absorber" generator with negative p_min_pu acts like a sink.
        absorber = n.add(qp.Generator, "Absorber", bus=bus, p_nom=100.0,
                         marginal_cost=0.0, p_min_pu=-1.0, p_max_pu=1.0)
        n.add(qp.Load, "L", bus=bus, p_set=20.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        # Cheap is at minimum 30 MW (forced); load is 20 → absorber is -10.
        assert cheap.p_t["p"].to_list() == [30.0] * 4
        assert absorber.p_t["p"].to_list() == [-10.0] * 4

    def test_p_max_pu_profile_varies_per_snapshot(self, snapshots):
        """A renewable profile of [0.1, 0.5, 0.8, 0.0] times p_nom=100 must
        cap the cheap gen to [10, 50, 80, 0] MW. With load 100 MW the
        expensive gen fills the rest exactly."""
        profile = pl.Series("time", [0.1, 0.5, 0.8, 0.0])
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        renew = n.add(qp.Generator, "Renew", bus=bus, p_nom=100.0,
                      marginal_cost=0.0, p_max_pu=profile)
        backup = n.add(qp.Generator, "Backup", bus=bus, p_nom=200.0,
                       marginal_cost=50.0)
        n.add(qp.Load, "L", bus=bus, p_set=100.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        renew_p = renew.p_t["p"].to_list()
        backup_p = backup.p_t["p"].to_list()
        assert renew_p == [10.0, 50.0, 80.0, 0.0]
        assert backup_p == [90.0, 50.0, 20.0, 100.0]

    def test_marginal_cost_zero_yields_no_objective_term(self, snapshots):
        """Zero-cost generator contributes nothing to the objective; an LP
        with only zero-cost generation and a feasible load solves to
        objective 0."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "Free", bus=bus, p_nom=100.0, marginal_cost=0.0)
        n.add(qp.Load, "L", bus=bus, p_set=30.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        assert n.objective_value == 0.0
