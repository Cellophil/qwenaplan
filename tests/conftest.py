"""Shared fixtures for the qwenaplan test suite.

Pattern: every fixture returns a network with components already added but
``set_snapshots()`` NOT yet called. That way tests can still mutate the
fixture (add more components, change parameters) before variables are
materialised.

Naming: every fixture that builds a meaningful test network ends in
``_network`` and returns a tuple ``(network, *components)``. The component
order is fixed so unpacking stays stable.
"""
import pytest
import polars as pl

import qwenaplan as qp


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

@pytest.fixture
def network():
    """Empty network."""
    return qp.Network()


@pytest.fixture
def snapshots():
    """Default 4-snapshot index. Enough for ramping / SOC trajectories."""
    return pl.Series("time", [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Pre-configured test networks
# ---------------------------------------------------------------------------
# These return a network *before* set_snapshots() so tests can still mutate
# parameters or add components.

@pytest.fixture
def two_bus_network(network):
    """Two buses, one generator at bus1, one AC line.

    Topology::

        Bus1 -- ACLine -- Bus2
          |
       [Gen1]

    Returns ``(network, bus1, bus2, gen1, line1)``. There is *no* load — add
    one (or hand-set demand via additional components) inside the test.
    """
    bus1 = network.add(qp.Bus, "Bus1", v_nom=1.0)
    bus2 = network.add(qp.Bus, "Bus2", v_nom=1.0)
    gen1 = network.add(qp.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
    line1 = network.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=200.0)
    return network, bus1, bus2, gen1, line1


@pytest.fixture
def three_bus_network(network):
    """Three buses, two generators, two AC lines, one Link.

    Topology::

        Bus1 -- Line1 -- Bus2 -- Line2 -- Bus3
          \\___________ Link1 ____________/

    Returns ``(network, bus1, bus2, bus3, gen1, gen2, line1, line2, link1)``.
    """
    bus1 = network.add(qp.Bus, "Bus1", v_nom=1.0)
    bus2 = network.add(qp.Bus, "Bus2", v_nom=1.0)
    bus3 = network.add(qp.Bus, "Bus3", v_nom=1.0)

    gen1 = network.add(qp.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
    gen2 = network.add(qp.Generator, "Gen2", bus=bus2, p_nom=80.0, marginal_cost=20.0)

    line1 = network.add(qp.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=200.0)
    line2 = network.add(qp.ACLine, "Line2", from_bus=bus2, to_bus=bus3, x_pu=0.15, s_nom=200.0)
    link1 = network.add(qp.Link, "Link1", from_bus=bus1, to_bus=bus3, p_nom=50.0)

    return network, bus1, bus2, bus3, gen1, gen2, line1, line2, link1


@pytest.fixture
def storage_test_network(network):
    """One bus, one generator (cheap), one storage unit, one constant load.

    Topology::

        Bus1 -- [Gen1, marginal_cost=50]
              -- [StorageUnit1, e_nom=100, p_nom 30/30, eff 0.9/0.9]
              -- [Load1, p_set=20]

    Returns ``(network, bus, gen, storage, load)``.
    """
    bus = network.add(qp.Bus, "Bus1", v_nom=1.0)
    gen = network.add(qp.Generator, "Gen1", bus=bus, p_nom=100.0, marginal_cost=50.0)
    storage = network.add(
        qp.StorageUnit, "Storage1", bus=bus,
        e_nom=100.0, p_nom_in=30.0, p_nom_out=30.0,
        eff_in=0.9, eff_out=0.9, initial_soc=20.0,
    )
    load = network.add(qp.Load, "Load1", bus=bus, p_set=20.0)
    return network, bus, gen, storage, load


@pytest.fixture
def battery_test_network(network):
    """One bus, one generator, one battery, one constant load.

    Topology::

        Bus1 -- [Gen1, marginal_cost=50]
              -- [Battery1, e_nom=100, p_nom=30, eff 0.95/0.95]
              -- [Load1, p_set=20]

    Returns ``(network, bus, gen, battery, load)``.
    """
    bus = network.add(qp.Bus, "Bus1", v_nom=1.0)
    gen = network.add(qp.Generator, "Gen1", bus=bus, p_nom=100.0, marginal_cost=50.0)
    battery = network.add(
        qp.Battery, "Battery1", bus=bus,
        e_nom=100.0, p_nom=30.0,
        eff_store=0.95, eff_dispatch=0.95, initial_soc=20.0,
    )
    load = network.add(qp.Load, "Load1", bus=bus, p_set=20.0)
    return network, bus, gen, battery, load


@pytest.fixture
def phs_test_network(network):
    """One bus, one generator, one pumped-hydro plant, one constant load."""
    bus = network.add(qp.Bus, "Bus1", v_nom=1.0)
    gen = network.add(qp.Generator, "Gen1", bus=bus, p_nom=100.0, marginal_cost=50.0)
    phs = network.add(
        qp.PumpedHydroStorage, "PHS1", bus=bus,
        e_nom=1000.0, p_nom_turbine=50.0, p_nom_pump=40.0,
        eff_store=0.85, eff_dispatch=0.9, gen_efficiency=0.9,
        initial_soc=500.0,
    )
    load = network.add(qp.Load, "Load1", bus=bus, p_set=30.0)
    return network, bus, gen, phs, load
