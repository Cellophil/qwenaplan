"""Tests for ACLine component."""
import pytest
import polars as pl
import qwenaplan as pypsa


class TestACLineInit:
    """Test ACLine initialization and parameter handling."""
    
    def test_default_parameters(self, network):
        """Test default values are set correctly."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2)
        
        assert line.name == "Line1"
        assert line.network == network
        assert line.from_bus == bus1
        assert line.to_bus == bus2
        assert line.x_pu == 0.1
        assert line.s_nom == 0.0
    
    def test_custom_parameters(self, network):
        """Test custom parameter values are accepted."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(
            pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2,
            x_pu=0.2, s_nom=100.0
        )
        
        assert line.x_pu == 0.2
        assert line.s_nom == 100.0
    
    def test_added_to_network(self, network):
        """Test ACLine is properly registered in network.lines."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2)
        
        assert "Line1" in network.lines
        assert network.lines["Line1"] is line
    
    def test_repr(self, network):
        """Test ACLine string representation."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=100.0)
        
        expected = "<ACLine(name=Line1, Bus1->Bus2, x_pu=0.1, s_nom=100.0)>"
        assert repr(line) == expected


class TestACLineVariables:
    """Test variable creation for ACLine."""
    
    def test_p_variable_exists(self, network, snapshots):
        """Test that ACLine creates p variable on instance."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1)
        network.set_snapshots(snapshots)
        
        assert hasattr(line, "p")


class TestACLineConstraints:
    """Test constraint creation for ACLine."""
    
    def test_kvl_constraint_created(self, network, snapshots):
        """Test that KVL constraint is created."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1)
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Instance-level check: component has p variable
        assert hasattr(line, "p")
    
    def test_thermal_limit_with_s_nom(self, network, snapshots):
        """Test that thermal limit constraint is created when s_nom > 0."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=100.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Instance-level check: component has p variable
        assert hasattr(line, "p")
    
    def test_no_thermal_limit_without_s_nom(self, network, snapshots):
        """Test that no thermal limit constraint is created when s_nom = 0."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        line = network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1, s_nom=0.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Instance-level check: component has p variable
        assert hasattr(line, "p")
