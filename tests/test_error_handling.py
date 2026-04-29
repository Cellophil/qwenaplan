"""Tests for error handling and edge cases."""
import pytest
import polars as pl
import qwenaplan as pypsa


class TestInvalidBusReferences:
    """Test that invalid bus references raise appropriate errors."""
    
    def test_generator_with_nonexistent_bus_raises_error(self, network):
        """Test that adding a Generator with a non-existent bus raises TypeError."""
        with pytest.raises(TypeError, match="Expected Bus object"):
            network.add(pypsa.Generator, "Gen1", bus="Bus1", p_nom=100.0)
    
    def test_ac_line_with_nonexistent_from_bus_raises_error(self, network):
        """Test that adding an ACLine with a non-existent from_bus raises TypeError."""
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        with pytest.raises(TypeError, match="from_bus and to_bus must be Bus objects"):
            network.add(pypsa.ACLine, "Line1", from_bus="Bus1", to_bus=bus2, x_pu=0.1)
    
    def test_ac_line_with_nonexistent_to_bus_raises_error(self, network):
        """Test that adding an ACLine with a non-existent to_bus raises TypeError."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        with pytest.raises(TypeError, match="from_bus and to_bus must be Bus objects"):
            network.add(pypsa.ACLine, "Line1", from_bus=bus1, to_bus="Bus2", x_pu=0.1)
    
    def test_link_with_nonexistent_from_bus_raises_error(self, network):
        """Test that adding a Link with a non-existent from_bus raises TypeError."""
        bus2 = network.add(pypsa.Bus, "Bus2", v_nom=1.0)
        with pytest.raises(TypeError, match="from_bus and to_bus must be Bus objects"):
            network.add(pypsa.Link, "Link1", from_bus="Bus1", to_bus=bus2, p_nom=50.0)
    
    def test_link_with_nonexistent_to_bus_raises_error(self, network):
        """Test that adding a Link with a non-existent to_bus raises TypeError."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        with pytest.raises(TypeError, match="from_bus and to_bus must be Bus objects"):
            network.add(pypsa.Link, "Link1", from_bus=bus1, to_bus="Bus2", p_nom=50.0)


class TestNetworkLockedErrors:
    """Test errors when network is locked after create_model."""
    
    def test_add_component_after_lock_raises_error(self, network, snapshots):
        """Test that adding a component after create_model raises RuntimeError."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        with pytest.raises(RuntimeError, match="Network is locked"):
            network.add(pypsa.Bus, "Bus2", v_nom=1.0)
    
    def test_set_snapshots_after_lock_raises_error(self, network, snapshots):
        """Test that setting snapshots after create_model raises RuntimeError."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        new_snapshots = pl.Series("time", [5])
        with pytest.raises(RuntimeError, match="Network is locked"):
            network.set_snapshots(new_snapshots)


class TestModelCreationErrors:
    """Test errors during model creation."""
    
    def test_create_model_without_snapshots_raises_error(self, network):
        """Test that create_model without set_snapshots raises RuntimeError."""
        bus1 = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        with pytest.raises(RuntimeError, match="set_snapshots"):
            network.create_model()
    
    def test_create_model_with_empty_network(self, network, snapshots):
        """Test that create_model with empty network (no components) works."""
        network.set_snapshots(snapshots)
        # Should not raise an error
        network.create_model()


class TestUnsupportedComponent:
    """Test error handling for unsupported component types."""
    
    def test_add_unsupported_component_raises_error(self, network):
        """Test that adding an unsupported component class raises ValueError."""
        class FakeComponent:
            pass
        
        with pytest.raises(ValueError, match="Unsupported component class"):
            network.add(FakeComponent, "Fake1")
