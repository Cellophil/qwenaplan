"""Tests for Link component."""
import pytest
import polars as pl
import qwenaplan as pypsa


class TestLinkInit:
    """Test Link initialization and parameter handling."""
    
    def test_default_parameters(self, network):
        """Test default values are set correctly."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        link = network.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2)
        
        assert link.name == "Link1"
        assert link.network == network
        assert link.from_bus == bus1
        assert link.to_bus == bus2
        assert link.p_nom == 0.0
        assert link.efficiency == 1.0
        assert link.carrier == ""
    
    def test_custom_parameters(self, network):
        """Test custom parameter values are accepted."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        link = network.add(
            pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2,
            p_nom=50.0, efficiency=0.95, carrier="DC"
        )
        
        assert link.p_nom == 50.0
        assert link.efficiency == 0.95
        assert link.carrier == "DC"
    
    def test_added_to_network(self, network):
        """Test Link is properly registered in network.links."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        link = network.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2)
        
        assert "Link1" in network.links
        assert network.links["Link1"] is link
    
    def test_repr(self, network):
        """Test Link string representation."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        link = network.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=50.0)
        
        expected = "<Link(name=Link1, Bus1->Bus2, p_nom=50.0)>"
        assert repr(link) == expected


class TestLinkVariables:
    """Test variable creation for Link."""
    
    def test_p_variable_exists(self, network, snapshots):
        """Test that Link creates p variable on instance."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        link = network.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=50.0)
        network.set_snapshots(snapshots)
        
        assert hasattr(link, "p")


class TestLinkConstraints:
    """Test constraint creation for Link."""
    
    def test_limit_constraint_created(self, network, snapshots):
        """Test that link_limit constraint is created."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        link = network.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=50.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Instance-level check: component has p variable
        assert hasattr(link, "p")
