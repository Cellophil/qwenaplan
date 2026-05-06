"""Tests for the Link component (controllable inter-bus flow).

Init / repr smoke tests verify the constructor stores parameters; one
optimization test verifies the Link actually carries power against a real
load on the receiving bus.
"""
import polars as pl
import pyoptinterface as poi

import qwenaplan as qp


class TestLinkInit:
    def test_default_parameters(self, network):
        bus1 = network.add(qp.Bus, "Bus1")
        bus2 = network.add(qp.Bus, "Bus2")
        link = network.add(qp.Link, "Link1", from_bus=bus1, to_bus=bus2)

        assert link.from_bus is bus1
        assert link.to_bus is bus2
        assert link.p_nom == 0.0
        assert link.efficiency == 1.0
        assert link.carrier == ""

    def test_custom_parameters(self, network):
        bus1 = network.add(qp.Bus, "Bus1")
        bus2 = network.add(qp.Bus, "Bus2")
        link = network.add(
            qp.Link, "Link1", from_bus=bus1, to_bus=bus2,
            p_nom=50.0, efficiency=0.95, carrier="DC",
        )
        assert link.p_nom == 50.0
        assert link.efficiency == 0.95
        assert link.carrier == "DC"

    def test_repr(self, network):
        bus1 = network.add(qp.Bus, "Bus1")
        bus2 = network.add(qp.Bus, "Bus2")
        link = network.add(qp.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=50.0)
        assert repr(link) == "<Link(name=Link1, Bus1->Bus2, p_nom=50.0)>"


class TestLinkOptimization:
    def test_link_carries_demand_to_remote_bus(self, snapshots):
        """A 30 MW load at bus2 must be served from gen at bus1 via the Link.

        With no AC line between them, the Link is the only path: solving the
        problem must yield link.sol.p_t == 30 MW for every snapshot.
        """
        n = qp.Network()
        bus1 = n.add(qp.Bus, "Bus1")
        bus2 = n.add(qp.Bus, "Bus2")
        gen = n.add(qp.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
        link = n.add(qp.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=50.0)
        n.add(qp.Load, "Load1", bus=bus2, p_set=30.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        # Link from bus1 → bus2 is carrying the full demand each snapshot.
        flows = link.sol.p_t["p"].to_list()
        assert flows == [30.0, 30.0, 30.0, 30.0]

        # Generator covers exactly that demand (no other sinks/sources).
        gen_p = gen.sol.p_t["p"].to_list()
        assert gen_p == [30.0, 30.0, 30.0, 30.0]

    def test_link_capacity_binds(self, snapshots):
        """If the load exceeds the Link's p_nom and the Link is the only path,
        the LP must be infeasible — confirms the limit constraint is active."""
        n = qp.Network()
        bus1 = n.add(qp.Bus, "Bus1")
        bus2 = n.add(qp.Bus, "Bus2")
        n.add(qp.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=20.0)
        n.add(qp.Load, "Load1", bus=bus2, p_set=30.0)

        n.set_snapshots(snapshots)
        n.create_model()
        status = n.optimize()
        # HiGHS reports INFEASIBLE for an over-constrained LP.
        assert status != poi.TerminationStatusCode.OPTIMAL
