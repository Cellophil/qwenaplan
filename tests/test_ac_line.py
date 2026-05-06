"""Tests for the ACLine component (DC linear power flow).

Init/repr smoke tests stay; the constraint-existence smoke tests have been
replaced with real KVL behaviour assertions: a single line carrying load,
and a parallel-line case where the two lines split flow inversely with
their reactances.
"""
import polars as pl
import pyoptinterface as poi

import qwenaplan as qp


class TestACLineInit:
    def test_default_parameters(self, network):
        bus1 = network.add(qp.Bus, "Bus1")
        bus2 = network.add(qp.Bus, "Bus2")
        line = network.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2)
        assert line.x_pu == 0.1
        assert line.s_nom == 0.0
        assert line.from_bus is bus1
        assert line.to_bus is bus2

    def test_custom_parameters(self, network):
        bus1 = network.add(qp.Bus, "Bus1")
        bus2 = network.add(qp.Bus, "Bus2")
        line = network.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.2, s_nom=100.0)
        assert line.x_pu == 0.2
        assert line.s_nom == 100.0

    def test_repr(self, network):
        bus1 = network.add(qp.Bus, "Bus1")
        bus2 = network.add(qp.Bus, "Bus2")
        line = network.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=100.0)
        assert repr(line) == "<ACLine(name=Line1, Bus1->Bus2, x_pu=0.1, s_nom=100.0)>"


class TestACLinePhysics:
    """KVL behaviour: phase angles and reactances determine flow split."""

    def test_single_line_carries_full_load(self, snapshots):
        """One line, one load. Flow == load every snapshot."""
        n = qp.Network()
        bus1 = n.add(qp.Bus, "Bus1")
        bus2 = n.add(qp.Bus, "Bus2")
        n.add(qp.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
        line = n.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=200.0)
        n.add(qp.Load, "Load1", bus=bus2, p_set=40.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        assert line.sol.p_t["p"].to_list() == [40.0, 40.0, 40.0, 40.0]

    def test_parallel_lines_split_flow_inverse_to_reactance(self, snapshots):
        """Two lines in parallel between bus1 and bus2 with x_pu = 0.1 and
        0.3 must split flow 3:1 (KVL forces equal angle drop, so line with
        smaller x carries more current). Total = load = 40 MW → 30 / 10."""
        n = qp.Network()
        bus1 = n.add(qp.Bus, "Bus1")
        bus2 = n.add(qp.Bus, "Bus2")
        n.add(qp.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
        line_a = n.add(qp.ACLine, "LineA", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=200.0)
        line_b = n.add(qp.ACLine, "LineB", from_bus=bus1, to_bus=bus2, x_pu=0.3, s_nom=200.0)
        n.add(qp.Load, "Load1", bus=bus2, p_set=40.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        flow_a = line_a.sol.p_t["p"].to_list()
        flow_b = line_b.sol.p_t["p"].to_list()
        # Each snapshot: 30 + 10 = 40 MW, ratio 3:1.
        for fa, fb in zip(flow_a, flow_b):
            assert abs(fa - 30.0) < 1e-6
            assert abs(fb - 10.0) < 1e-6

    def test_thermal_limit_binds_when_undersized(self, snapshots):
        """If s_nom < required flow, the LP is infeasible."""
        n = qp.Network()
        bus1 = n.add(qp.Bus, "Bus1")
        bus2 = n.add(qp.Bus, "Bus2")
        n.add(qp.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=20.0)
        n.add(qp.Load, "Load1", bus=bus2, p_set=40.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() != poi.TerminationStatusCode.OPTIMAL

    def test_no_thermal_limit_when_s_nom_zero(self, snapshots):
        """s_nom=0 disables thermal limit (default behaviour); LP solves
        even with arbitrarily large transfer."""
        n = qp.Network()
        bus1 = n.add(qp.Bus, "Bus1")
        bus2 = n.add(qp.Bus, "Bus2")
        n.add(qp.Generator, "Gen1", bus=bus1, p_nom=10000.0, marginal_cost=10.0)
        line = n.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=0.0)
        n.add(qp.Load, "Load1", bus=bus2, p_set=5000.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        assert line.sol.p_t["p"].to_list() == [5000.0] * 4
