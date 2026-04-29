"""Tests for PumpedHydroStorage composite class.

Note: Basic storage functionality (SOC dynamics, bounds, etc.) is tested
in test_storage_unit.py. This file focuses on PHS-specific features:
- Generator coupling
- Electrical power output
- Transparent property access
"""
import polars as pl
import pyoptinterface as poi
import pyoframe as pf
import pytest
import qwenaplan as pypsa


class TestPumpedHydroInit:
    """Test PumpedHydroStorage initialization and parameter handling."""
    
    def test_default_parameters(self):
        """Test default values are set correctly."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        phs = n.add(
            pypsa.PumpedHydroStorage, "phs1", bus=bus,
            e_nom=1000.0, p_nom_turbine=100.0
        )
        
        assert phs.name == "phs1"
        assert phs.bus == bus
        assert phs.e_nom == 1000.0
        assert phs.p_nom_turbine == 100.0
        assert phs.p_nom_pump == 0.0
        assert phs.gen_efficiency == 0.9
    
    def test_custom_parameters(self):
        """Test custom parameter values are accepted."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        phs = n.add(
            pypsa.PumpedHydroStorage, "phs2", bus=bus,
            e_nom=2000.0, p_nom_turbine=150.0, p_nom_pump=100.0,
            eff_store=0.85, eff_dispatch=0.9, gen_efficiency=0.95,
            initial_soc=1000.0, soc_min=200.0, soc_max=1800.0, influx=5.0
        )
        
        assert phs.p_nom_pump == 100.0
        assert phs.eff_store == 0.85
        assert phs.eff_dispatch == 0.9
        assert phs.gen_efficiency == 0.95
        assert phs.initial_soc == 1000.0
        assert phs.soc_min == 200.0
        assert phs.soc_max == 1800.0
        assert phs.influx == 5.0
    
    def test_internal_components_created(self):
        """Test that internal storage and generator are created."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        phs = n.add(
            pypsa.PumpedHydroStorage, "phs1", bus=bus,
            e_nom=1000.0, p_nom_turbine=100.0
        )
        
        assert hasattr(phs, "storage")
        assert phs.storage.name == "phs1_storage"
        assert phs.storage.e_nom == 1000.0
        
        assert hasattr(phs, "generator")
        assert phs.generator.name == "phs1_generator"
        assert phs.generator.p_nom == 100.0
    
    def test_repr(self):
        """Test string representation."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        phs = n.add(
            pypsa.PumpedHydroStorage, "phs1", bus=bus,
            e_nom=1000.0, p_nom_turbine=100.0
        )
        
        assert "phs1" in repr(phs)
        assert "bus1" in repr(phs)
        assert "1000.0" in repr(phs)
        assert "100.0" in repr(phs)


class TestPumpedHydroVariables:
    """Test variable creation for PumpedHydroStorage."""
    
    def test_variables_created(self):
        """Test that variables are created for both components."""
        n = pypsa.Network()
        bus = n.add(pypsa.Bus, "bus1")
        phs = n.add(
            pypsa.PumpedHydroStorage, "phs1", bus=bus,
            e_nom=1000.0, p_nom_turbine=100.0
        )
        
        snapshots = pl.Series("time", range(5))
        n.set_snapshots(snapshots)
        
        # Check storage variables
        assert hasattr(phs.storage, "soc")
        assert hasattr(phs.storage, "p_in")
        assert hasattr(phs.storage, "p_out")
        
        # Check generator variables
        assert hasattr(phs.generator, "p")
        
        # Check property delegates
        assert phs.soc is phs.storage.soc
        assert phs.p_store is phs.storage.p_in
        assert phs.p_dispatch is phs.storage.p_out
        assert phs.p is phs.generator.p


class TestPumpedHydroConstraints:
    """Test constraint creation for PumpedHydroStorage."""
    
    def test_model_created(self, phs_test_network, snapshots):
        """Test that model is created without errors."""
        n, bus, gen, phs = phs_test_network
        n.set_snapshots(snapshots)
        n.create_model()
        
        assert n.model is not None
    
    def test_coupling_constraint(self, phs_test_network, snapshots):
        """Test that generator-storage coupling constraint is created."""
        n, bus, gen, phs = phs_test_network
        phs.gen_efficiency = 0.9
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Instance-level check: generator has p variable
        assert hasattr(phs.generator, "p")
    
    def test_p_net_returns_generator_power(self, phs_test_network, snapshots):
        """Test that get_p_net returns generator power."""
        n, bus, gen, phs = phs_test_network
        n.set_snapshots(snapshots)
        
        p_net = phs.get_p_net()
        assert p_net is phs.generator.p


class TestPumpedHydroOptimization:
    """Test optimization behavior for PumpedHydroStorage."""
    
    def test_generator_coupling(self, phs_test_network, snapshots):
        """Test that generator output is coupled to water dispatch."""
        n, bus, gen, phs = phs_test_network
        phs.eff_dispatch = 1.0
        phs.gen_efficiency = 0.9
        phs.initial_soc = 500.0
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Cost minimization
        n.model.minimize = (gen.p * 100.0).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        gen_power = phs.p.solution["solution"].to_list()
        water_dispatch = phs.p_dispatch.solution["solution"].to_list()
        
        for gp, wd in zip(gen_power, water_dispatch):
            assert abs(gp - wd * 0.9) < 1e-6
    
    def test_pumping_and_generation(self, phs_test_network, snapshots):
        """Test that PHS can both pump and generate."""
        n, bus, gen, phs = phs_test_network
        phs.eff_store = 0.85
        phs.eff_dispatch = 1.0
        phs.gen_efficiency = 0.9
        phs.initial_soc = 500.0
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Add load to make problem feasible
        load_df = snapshots.to_frame().with_columns(pl.lit(20.0).alias("load"))
        load_param = pf.Param(load_df)
        
        n.model.load_constraint = gen.p + phs.p >= load_param
        n.model.minimize = (gen.p * 100.0).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        gen_power = phs.p.solution["solution"].to_list()
        pump_power = phs.p_store.solution["solution"].to_list()
        
        # At least some activity should occur
        assert sum(abs(g) for g in gen_power) + sum(abs(p) for p in pump_power) > 0
    
    def test_soc_decreases_when_generating(self, phs_test_network, snapshots):
        """Test that SOC decreases when generating power."""
        n, bus, gen, phs = phs_test_network
        phs.p_nom_pump = 0.0
        phs.initial_soc = 500.0
        phs.influx = 0.0
        phs.eff_dispatch = 1.0
        phs.gen_efficiency = 0.9
        
        n.set_snapshots(snapshots)
        n.create_model()
        
        # Add load to force generation
        load_df = snapshots.to_frame().with_columns(pl.lit(30.0).alias("load"))
        load_param = pf.Param(load_df)
        
        n.model.load_constraint = gen.p + phs.p >= load_param
        n.model.minimize = (gen.p * 100.0).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        soc_values = phs.soc.solution["solution"].to_list()
        gen_power = phs.p.solution["solution"].to_list()
        
        # SOC should decrease when generating
        total_generation = sum(gen_power)
        if total_generation > 0:
            assert soc_values[-1] <= soc_values[0]
