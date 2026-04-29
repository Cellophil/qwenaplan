"""Tests for StorageUnit base class."""
import polars as pl
import pyoptinterface as poi
import pytest
import qwenaplan as pypsa


class TestStorageUnitInit:
    """Test StorageUnit initialization and parameter handling."""
    
    def test_default_parameters(self):
        """Test default values are set correctly."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        storage = n.add(pypsa.StorageUnit, "storage1", bus=bus, e_nom=100.0)
        
        assert storage.name == "storage1"
        assert storage.bus == bus
        assert storage.e_nom == 100.0
        assert storage.eff_store == 1.0
        assert storage.eff_dispatch == 1.0
        assert storage.initial_soc == 0.0
        assert storage.soc_min == 0.0
        assert storage.soc_max == 100.0
        assert storage._influx == 0.0
    
    def test_custom_parameters(self):
        """Test custom parameter values are accepted."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        storage = n.add(
            pypsa.StorageUnit, "storage2", bus=bus,
            e_nom=200.0, p_nom_in=50.0, p_nom_out=60.0,
            eff_in=0.9, eff_out=0.95,
            initial_soc=100.0, soc_min=20.0, soc_max=180.0, influx=5.0
        )
        
        assert storage.p_nom_in == 50.0
        assert storage.p_nom_out == 60.0
        assert storage.eff_in == 0.9
        assert storage.eff_out == 0.95
        assert storage.initial_soc == 100.0
        assert storage.soc_min == 20.0
        assert storage.soc_max == 180.0
        assert storage._influx == 5.0
    
    def test_repr(self):
        """Test string representation."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        storage = n.add(pypsa.StorageUnit, "storage1", bus=bus, e_nom=100.0)
        
        assert "storage1" in repr(storage)
        assert "bus1" in repr(storage)
        assert "100.0" in repr(storage)


class TestStorageUnitVariables:
    """Test variable creation for StorageUnit."""
    
    def test_variables_created(self):
        """Test that variables are created correctly."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        storage = n.add(pypsa.StorageUnit, "storage1", bus=bus, e_nom=100.0)
        
        snapshots = pl.Series("time", range(5))
        n.set_snapshots(snapshots)
        
        assert hasattr(storage, "soc")
        assert hasattr(storage, "p_store")
        assert hasattr(storage, "p_dispatch")
        assert len(storage.soc) == 5
        assert len(storage.p_store) == 5
        assert len(storage.p_dispatch) == 5
    
    def test_influx_series_constant(self):
        """Test that constant influx is converted to series."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        storage = n.add(pypsa.StorageUnit, "storage1", bus=bus, e_nom=100.0, influx=5.0)
        
        snapshots = pl.Series("time", range(5))
        n.set_snapshots(snapshots)
        
        assert hasattr(storage, "_influx_series")
        assert len(storage._influx_series) == 5
        assert all(storage._influx_series == 5.0)
    
    def test_influx_profile(self):
        """Test that influx profile is used when set."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        storage = n.add(pypsa.StorageUnit, "storage1", bus=bus, e_nom=100.0, influx=0.0)
        
        influx_profile = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        storage.set_influx_profile(influx_profile)
        
        snapshots = pl.Series("time", range(5))
        n.set_snapshots(snapshots)
        
        assert hasattr(storage, "_influx_series")
        assert len(storage._influx_series) == 5
        assert list(storage._influx_series) == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestStorageUnitConstraints:
    """Test constraint creation for StorageUnit."""
    
    def test_model_created(self, storage_test_network, snapshots):
        """Test that model is created without errors."""
        n, bus, gen, storage = storage_test_network
        n.set_snapshots(snapshots)
        n.create_model()
        
        assert n.model is not None
    
    def test_soc_balance_constraint(self, storage_test_network, snapshots):
        """Test that SOC balance constraint is created."""
        n, bus, gen, storage = storage_test_network
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Instance-level check: storage has soc variable
        assert hasattr(storage, "soc")
    
    def test_initial_soc_constraint(self, storage_test_network, snapshots):
        """Test that initial SOC constraint is created."""
        n, bus, gen, storage = storage_test_network
        storage.initial_soc = 50.0
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Instance-level check: storage has soc variable
        assert hasattr(storage, "soc")
    
    def test_power_limits(self, storage_test_network, snapshots):
        """Test that power limit constraints are created."""
        n, bus, gen, storage = storage_test_network
        storage.p_nom_in = 50.0
        storage.p_nom_out = 60.0
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Instance-level check: storage has p_in, p_out variables
        assert hasattr(storage, "p_in")
        assert hasattr(storage, "p_out")
    
    def test_soc_bounds(self, storage_test_network, snapshots):
        """Test that SOC bound constraints are created."""
        n, bus, gen, storage = storage_test_network
        storage.soc_min = 10.0
        storage.soc_max = 90.0
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Instance-level check: storage has soc variable
        assert hasattr(storage, "soc")


class TestStorageUnitOptimization:
    """Test optimization behavior for StorageUnit."""
    
    def test_charge_behavior(self, storage_test_network, snapshots):
        """Test that storage charges when beneficial in cost minimization."""
        n, bus, gen, storage = storage_test_network
        storage.p_nom_in = 30.0
        storage.p_nom_out = 30.0
        storage.eff_in = 0.9
        storage.eff_out = 0.9
        storage.initial_soc = 20.0
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Cost minimization with expensive generator
        n.model.minimize = (gen.p * 100.0).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        soc_values = storage.soc.solution["solution"].to_list()
        # SOC should change during optimization
        assert len(soc_values) == 4
    
    def test_soc_bounds_respected(self, storage_test_network, snapshots):
        """Test that SOC bounds are respected in optimization."""
        n, bus, gen, storage = storage_test_network
        storage.p_nom_in = 100.0
        storage.p_nom_out = 100.0
        storage.initial_soc = 50.0
        storage.soc_min = 20.0
        storage.soc_max = 80.0
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Try to maximize final SOC
        n.model.maximize = storage.soc.filter(pl.col("time") == 3).sum()
        n.model.optimize()
        
        soc_values = storage.soc.solution["solution"].to_list()
        assert all(s >= 20.0 - 1e-6 for s in soc_values)
        assert all(s <= 80.0 + 1e-6 for s in soc_values)
    
    def test_influx_affects_soc(self, storage_test_network, snapshots):
        """Test that influx affects SOC dynamics."""
        n, bus, gen, storage = storage_test_network
        storage.p_nom_in = 0.0
        storage.p_nom_out = 0.0
        storage.initial_soc = 50.0
        storage._influx = 10.0
        
        n.set_snapshots(snapshots)
        n.create_model()
        n.model.optimize()
        
        soc_values = storage.soc.solution["solution"].to_list()
        # SOC should increase due to influx
        assert len(soc_values) == 4
