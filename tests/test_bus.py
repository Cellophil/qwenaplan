"""Tests for Bus component."""
import pytest
import polars as pl
import qwenaplan as pypsa


class TestBusInit:
    """Test Bus initialization and parameter handling."""
    
    def test_default_parameters(self, network):
        """Test default values are set correctly."""
        bus = network.add(pypsa.Bus, "Bus1")
        
        assert bus.name == "Bus1"
        assert bus.network == network
        assert bus.v_nom == 1.0
        assert bus.carrier == "AC"
        assert bus.x == 0.0
        assert bus.y == 0.0
    
    def test_custom_parameters(self, network):
        """Test custom parameter values are accepted."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=380.0, carrier="HV", x=10.5, y=20.3)
        
        assert bus.v_nom == 380.0
        assert bus.carrier == "HV"
        assert bus.x == 10.5
        assert bus.y == 20.3
    
    def test_added_to_network(self, network):
        """Test Bus is properly registered in network.buses."""
        bus = network.add(pypsa.Bus, "Bus1")
        
        assert "Bus1" in network.buses
        assert network.buses["Bus1"] is bus
    
    def test_repr(self, network):
        """Test Bus string representation."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        
        expected = "<Bus(name=Bus1, v_nom=1.0)>"
        assert repr(bus) == expected


class TestBusVariables:
    """Test variable creation for Bus."""
    
    def test_variables_created(self, network, snapshots):
        """Test that Bus creates p_net and theta variables on instance."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        network.set_snapshots(snapshots)
        
        assert hasattr(bus, "p_net")
        assert hasattr(bus, "theta")
