"""Shared fixtures and configuration for qwenaplan tests.

This module provides pre-configured network fixtures that follow the pattern:
1. Create network and add components
2. Return network WITHOUT calling set_snapshots()
3. Test can modify network before applying snapshots

This is the key pattern for efficient testing - components can be added/modified
before set_snapshots() is called, which triggers variable creation.
"""
import pytest
import polars as pl
import qwenaplan as pypsa


# =============================================================================
# Basic Fixtures
# =============================================================================

@pytest.fixture
def network():
    """Create an empty network instance."""
    return pypsa.Network()


@pytest.fixture
def snapshots():
    """Create a default series of snapshots for testing (4 time steps)."""
    return pl.Series("time", [0, 1, 2, 3])


# =============================================================================
# Pre-configured Test Networks
# =============================================================================
# These fixtures return networks WITHOUT calling set_snapshots().
# Tests can modify the network before applying snapshots.

@pytest.fixture
def test_network(network):
    """Create a simple 2-bus test network with generator and line.
    
    Network structure:
        Bus1 --[Generator]-- ACLine -- Bus2
    
    Snapshots are NOT applied - the test can modify the network first.
    
    Returns:
        tuple: (network, bus1, bus2, gen1, line1)
    """
    bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
    bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
    gen1 = network.add(pypsa.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
    line1 = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1)
    
    return network, bus1, bus2, gen1, line1


@pytest.fixture
def multi_bus_network(network):
    """Create a 3-bus meshed network with mixed components.
    
    Network structure:
        Bus1 --[Generator]-- Bus2 --[Generator]-- Bus3
          |                    |                    |
          +------- ACLine -----+                    |
          |                    |                    |
          +--------- Link ---------+                |
          |                                         |
          +---------------- ACLine ----------------+
    
    Snapshots are NOT applied - the test can modify the network first.
    
    Returns:
        tuple: (network, bus1, bus2, bus3, gen1, gen2, line1, line2, link1)
    """
    bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
    bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
    bus3 = network.add(pypsa.Bus, "Bus3", v_nom=1.0)
    
    gen1 = network.add(pypsa.Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=10.0)
    gen2 = network.add(pypsa.Generator, "Gen2", bus=bus2, p_nom=80.0, marginal_cost=20.0)
    
    line1 = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1)
    line2 = network.add(pypsa.ACLine, "Line2", from_bus=bus2, to_bus=bus3, x_pu=0.15)
    link1 = network.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus3, p_nom=50.0)
    
    return network, bus1, bus2, bus3, gen1, gen2, line1, line2, link1


@pytest.fixture
def storage_test_network(network):
    """Create a test network with generator and storage unit.
    
    Network structure:
        Bus1 --[Generator]-- [StorageUnit]
    
    Snapshots are NOT applied - the test can modify the network first.
    
    Returns:
        tuple: (network, bus, gen, storage)
    """
    bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
    gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, marginal_cost=50.0)
    storage = network.add(
        pypsa.StorageUnit, "Storage1", bus=bus,
        e_nom=100.0, p_nom_in=30.0, p_nom_out=30.0,
        eff_in=0.9, eff_out=0.9, initial_soc=20.0
    )
    
    return network, bus, gen, storage


@pytest.fixture
def battery_test_network(network):
    """Create a test network with generator and battery.
    
    Network structure:
        Bus1 --[Generator]-- [Battery]
    
    Snapshots are NOT applied - the test can modify the network first.
    
    Returns:
        tuple: (network, bus, gen, battery)
    """
    bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
    gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, marginal_cost=50.0)
    battery = network.add(
        pypsa.Battery, "Battery1", bus=bus,
        e_nom=100.0, p_nom=30.0,
        eff_store=0.95, eff_dispatch=0.95, initial_soc=20.0
    )
    
    return network, bus, gen, battery


@pytest.fixture
def phs_test_network(network):
    """Create a test network with generator and pumped hydro storage.
    
    Network structure:
        Bus1 --[Generator]-- [PumpedHydroStorage]
    
    Snapshots are NOT applied - the test can modify the network first.
    
    Returns:
        tuple: (network, bus, gen, phs)
    """
    bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
    gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, marginal_cost=50.0)
    phs = network.add(
        pypsa.PumpedHydroStorage, "PHS1", bus=bus,
        e_nom=1000.0, p_nom_turbine=50.0, p_nom_pump=40.0,
        eff_store=0.85, eff_dispatch=0.9, gen_efficiency=0.9,
        initial_soc=500.0
    )
    
    return network, bus, gen, phs
