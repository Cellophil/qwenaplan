"""Integration tests for complete network build workflow."""
import pytest
import polars as pl
import pyoptinterface as poi
import pyoframe as pf
import qwenaplan as pypsa


class TestNetworkBuild:
    """Test complete network build from bottom-up."""
    
    def test_network_bottom_up_build(self, test_network, snapshots):
        """Test building a complete network with all component types."""
        n, bus1, bus2, gen1, line1 = test_network
        
        # Add link
        link1 = n.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=50.0)
        
        # Set snapshots (triggers variable creation)
        n.set_snapshots(snapshots)
        
        # Create model (triggers constraint creation)
        n.create_model()
        
        # Verify model is a pyoframe.Model
        assert isinstance(n.model, pf.Model)
        
        # Verify all variables are registered via component instances
        assert hasattr(gen1, "p")
        assert hasattr(line1, "p")
        assert hasattr(link1, "p")
    
    def test_linear_expression_with_component_variables(self, test_network, snapshots):
        """Test creating linear expressions with component variables."""
        n, bus1, bus2, gen1, line1 = test_network
        
        link1 = n.add(pypsa.Link, "Link1", from_bus=bus1, to_bus=bus2, p_nom=50.0)
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Test creating a linear expression
        lin_expr = line1.p + link1.p
        assert lin_expr is not None


class TestNetworkHelpers:
    """Test network helper methods."""
    
    def test_get_connected_power_elements(self, multi_bus_network, snapshots):
        """Test getting all generators connected to a bus."""
        n, bus1, bus2, bus3, gen1, gen2, line1, line2, link1 = multi_bus_network
        
        connected = n.get_connected_power_elements(bus1)
        assert gen1 in connected
        assert gen2 not in connected
    
    def test_get_connected_lines(self, multi_bus_network, snapshots):
        """Test getting all lines connected to a bus."""
        n, bus1, bus2, bus3, gen1, gen2, line1, line2, link1 = multi_bus_network
        
        connected_bus1 = n.get_connected_lines(bus1)
        assert line1 in connected_bus1
        assert link1 in connected_bus1


class TestOptimizationWorkflow:
    """Test complete optimization workflow."""
    
    def test_cost_minimization(self, test_network, snapshots):
        """Test cost minimization with two generators."""
        n, bus1, bus2, gen1, line1 = test_network
        
        # Add second generator at bus2
        gen2 = n.add(pypsa.Generator, "Gen2", bus=bus2, p_nom=50.0, marginal_cost=50.0)
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Cost minimization
        n.model.minimize = (gen1.p * gen1.marginal_cost + gen2.p * gen2.marginal_cost).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
    
    def test_storage_integration(self, storage_test_network, snapshots):
        """Test storage unit integration with generator."""
        n, bus, gen, storage = storage_test_network
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Cost minimization
        n.model.minimize = (gen.p * gen.marginal_cost).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        # Storage should have variables
        assert hasattr(storage, "soc")
        assert hasattr(storage, "p_store")
        assert hasattr(storage, "p_dispatch")
    
    def test_battery_integration(self, battery_test_network, snapshots):
        """Test battery integration with generator."""
        n, bus, gen, battery = battery_test_network
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Cost minimization
        n.model.minimize = (gen.p * gen.marginal_cost).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        # Battery should have variables
        assert hasattr(battery, "soc")
        assert hasattr(battery, "p")
