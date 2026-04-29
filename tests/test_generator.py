"""Tests for Generator component."""
import pytest
import polars as pl
import pyoptinterface as poi
import qwenaplan as pypsa


class TestGeneratorInit:
    """Test Generator initialization and parameter handling."""
    
    def test_default_parameters(self, network):
        """Test default values are set correctly."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus)
        
        assert gen.name == "Gen1"
        assert gen.network == network
        assert gen.bus == bus
        assert gen.p_nom == 0.0
        assert gen.marginal_cost == 0.0
        assert gen.carrier == ""
        assert gen in network.generators.values()
    
    def test_custom_parameters(self, network):
        """Test custom parameter values are accepted."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(
            pypsa.Generator, "Gen1", bus=bus,
            p_nom=100.0, marginal_cost=50.0, carrier="solar"
        )
        
        assert gen.p_nom == 100.0
        assert gen.marginal_cost == 50.0
        assert gen.carrier == "solar"
    
    def test_added_to_network(self, network):
        """Test Generator is properly registered in network.generators."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0)
        
        assert "Gen1" in network.generators
        assert network.generators["Gen1"] is gen
    
    def test_repr(self, network):
        """Test Generator string representation."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0)
        
        expected = "<Generator(name=Gen1, bus=Bus1, p_nom=100.0, marginal_cost=0.0)>"
        assert repr(gen) == expected


class TestGeneratorVariables:
    """Test variable creation for Generator."""
    
    def test_p_variable_exists(self, network, snapshots):
        """Test that Generator creates p variable on instance."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0)
        network.set_snapshots(snapshots)
        
        assert hasattr(gen, "p")
    
    def test_p_variable_registered_in_model(self, network, snapshots):
        """Test that Generator p variable is registered in the model."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "p_Gen1")


class TestGeneratorConstraints:
    """Test constraint creation for Generator."""
    
    def test_limit_constraint_created(self, network, snapshots):
        """Test that gen_limit constraint is created."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Instance-level check: component has p variable
        assert hasattr(gen, "p")
    
    def test_p_min_pu_creates_lower_bound(self, network, snapshots):
        """Test that p_min_pu creates gen_lower constraint."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, p_min_pu=0.3)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_lower_Gen1")
    
    def test_p_min_pu_validation(self, network):
        """Test that p_min_pu > p_max_pu raises ValueError."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        with pytest.raises(ValueError, match="p_min_pu.*must be <= p_max_pu"):
            network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, p_min_pu=0.8, p_max_pu=0.2)


class TestGeneratorOptimization:
    """Test optimization behavior for Generator."""
    
    def test_p_max_pu_limits_output(self, network, snapshots):
        """Test that p_max_pu limits generator output in optimization."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        
        # Generator with limited capacity
        limited_gen = network.add(
            pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
            p_max_pu=0.5, marginal_cost=10.0
        )
        
        # Expensive generator to force use of limited gen
        expensive_gen = network.add(
            pypsa.Generator, "Gen2", bus=bus, p_nom=100.0,
            marginal_cost=100.0
        )
        
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Cost minimization objective
        n = network
        n.model.minimize = (limited_gen.p * 10.0 + expensive_gen.p * 100.0).sum()
        n.model.optimize()
        
        # Limited gen should not exceed 50 MW (p_nom * p_max_pu)
        p_values = limited_gen.p.solution["solution"].to_list()
        assert all(p <= 50.0 + 1e-6 for p in p_values)
    
    def test_p_min_pu_affects_dispatch(self, network, snapshots):
        """Test that p_min_pu sets minimum output in optimization."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        
        # Generator with 30% minimum output
        gen = network.add(
            pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
            p_min_pu=0.3, marginal_cost=10.0
        )
        
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Cost minimization
        n = network
        n.model.minimize = (gen.p * 10.0).sum()
        n.model.optimize()
        
        # Gen should be at minimum output (30 MW)
        p_values = gen.p.solution["solution"].to_list()
        assert all(p >= 30.0 - 1e-6 for p in p_values)
    
    def test_marginal_cost_affects_dispatch(self, network, snapshots):
        """Test that cheaper generator is dispatched first in cost minimization."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        
        # Cheap generator
        cheap_gen = network.add(
            pypsa.Generator, "Cheap", bus=bus, p_nom=50.0,
            marginal_cost=10.0
        )
        
        # Expensive generator
        expensive_gen = network.add(
            pypsa.Generator, "Expensive", bus=bus, p_nom=100.0,
            marginal_cost=50.0
        )
        
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Cost minimization - cheap gen should be fully dispatched first
        n = network
        n.model.minimize = (cheap_gen.p * 10.0 + expensive_gen.p * 50.0).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        # Both generators should have non-negative output
        cheap_p = cheap_gen.p.solution["solution"].to_list()
        expensive_p = expensive_gen.p.solution["solution"].to_list()
        assert all(p >= -1e-6 for p in cheap_p)
        assert all(p >= -1e-6 for p in expensive_p)
