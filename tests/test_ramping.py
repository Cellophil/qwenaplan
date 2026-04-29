"""Tests for ramping constraints and p_min_pu support."""
import pytest
import polars as pl
import pyoptinterface as poi
import qwenaplan as pypsa


class TestPMinPuStatic:
    """Test static p_min_pu support for generators."""
    
    def test_generator_with_p_min_pu_static(self, network, snapshots):
        """Test Generator respects p_min_pu static limit."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, p_min_pu=0.3)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_lower_Gen1")
    
    def test_generator_with_p_min_pu_and_p_max_pu(self, network, snapshots):
        """Test Generator with both p_min_pu and p_max_pu."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, p_min_pu=0.2, p_max_pu=0.8)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_limit_Gen1")
        assert hasattr(network.model, "gen_lower_Gen1")
    
    def test_p_min_pu_validation_static(self, network):
        """Test that p_min_pu > p_max_pu raises ValueError."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        with pytest.raises(ValueError, match="p_min_pu.*must be <= p_max_pu"):
            network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, p_min_pu=0.8, p_max_pu=0.2)
    
    def test_generator_default_p_min_pu(self, network, snapshots):
        """Test Generator without p_min_pu has lower bound 0."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_lower_Gen1")


class TestPMinPuProfile:
    """Test profile-based p_min_pu support for generators."""
    
    def test_generator_with_p_min_pu_profile(self, network, snapshots):
        """Test Generator respects p_min_pu profile."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        min_profile = pl.Series(snapshots.name, [0.3, 0.4, 0.3, 0.4])
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, p_min_pu=min_profile)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_lower_Gen1")
    
    def test_p_min_pu_profile_exceeds_static_max(self, network, snapshots):
        """Test that p_min_pu profile exceeding static p_max_pu raises error."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        min_profile = pl.Series(snapshots.name, [0.6, 0.7, 0.6, 0.7])
        with pytest.raises(ValueError, match="p_max_pu.*below.*p_min_pu"):
            network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
                       p_min_pu=min_profile, p_max_pu=0.5)
            network.set_snapshots(snapshots)
            network.create_model()
    
    def test_p_max_pu_profile_below_static_min(self, network, snapshots):
        """Test that p_max_pu profile below static p_min_pu raises error."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        max_profile = pl.Series(snapshots.name, [0.1, 0.2, 0.1, 0.2])
        with pytest.raises(ValueError, match="p_min_pu.*exceeds.*p_max_pu"):
            network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
                       p_max_pu=max_profile, p_min_pu=0.3)
            network.set_snapshots(snapshots)
            network.create_model()


class TestRampingConstraints:
    """Test ramping constraints for generators."""
    
    def test_generator_with_ramp_limit_up(self, network, snapshots):
        """Test Generator with ramp-up limit creates constraint."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, ramp_limit_up=0.2)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_ramp_up_Gen1")
    
    def test_generator_with_ramp_limit_down(self, network, snapshots):
        """Test Generator with ramp-down limit creates constraint."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, ramp_limit_down=0.15)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_ramp_down_Gen1")
    
    def test_generator_with_both_ramp_limits(self, network, snapshots):
        """Test Generator with both ramp-up and ramp-down limits."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
                         ramp_limit_up=0.2, ramp_limit_down=0.1)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_ramp_up_Gen1")
    
    def test_generator_without_ramp_limits(self, network, snapshots):
        """Test Generator without ramp limits has no ramp constraint."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert not hasattr(network.model, "gen_ramp_up_Gen1")
    
    def test_ramp_limit_zero(self, network, snapshots):
        """Test Generator with zero ramp limit (must stay constant)."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
                         ramp_limit_up=0.0, ramp_limit_down=0.0)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_ramp_up_Gen1")


class TestRampingWithProfiles:
    """Test ramping constraints combined with profile limits."""
    
    def test_ramp_with_p_max_pu_profile(self, network, snapshots):
        """Test ramping works with p_max_pu profile."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        max_profile = pl.Series(snapshots.name, [0.8, 0.6, 0.8, 0.6])
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
                         p_max_pu=max_profile, ramp_limit_up=0.2)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_limit_Gen1")
        assert hasattr(network.model, "gen_ramp_up_Gen1")
    
    def test_ramp_with_p_min_pu_profile(self, network, snapshots):
        """Test ramping works with p_min_pu profile."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        min_profile = pl.Series(snapshots.name, [0.3, 0.4, 0.3, 0.4])
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
                         p_min_pu=min_profile, ramp_limit_down=0.1)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_lower_Gen1")
        assert hasattr(network.model, "gen_ramp_down_Gen1")


class TestRampingLimitValues:
    """Test that ramping limits are correctly calculated."""
    
    def test_ramp_limit_up_value(self, network, snapshots):
        """Test ramp-up limit is correctly calculated as p_nom * ramp_limit_up."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0, ramp_limit_up=0.25)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_ramp_up_Gen1")
    
    def test_ramp_limit_down_value(self, network, snapshots):
        """Test ramp-down limit is correctly calculated as p_nom * ramp_limit_down."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=200.0, ramp_limit_down=0.1)
        network.set_snapshots(snapshots)
        network.create_model()
        
        assert hasattr(network.model, "gen_ramp_down_Gen1")


class TestRampingOptimization:
    """Test ramping constraints in optimization."""
    
    def test_ramp_limit_up_affects_dispatch(self, network, snapshots):
        """Test that ramp-up limit affects generator dispatch."""
        bus = network.add(pypsa.Bus, "Bus1", v_nom=1.0)
        
        # Generator with tight ramp-up limit
        gen = network.add(pypsa.Generator, "Gen1", bus=bus, p_nom=100.0,
                         ramp_limit_up=0.1, marginal_cost=10.0)
        
        network.set_snapshots(snapshots)
        network.create_model()
        
        # Cost minimization
        n = network
        n.model.minimize = (gen.p * 10.0).sum()
        n.model.optimize()
        
        assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
        
        # Check that ramp-up constraint is respected
        p_values = gen.p.solution["solution"].to_list()
        for i in range(1, len(p_values)):
            assert p_values[i] - p_values[i-1] <= 10.0 + 1e-6  # 10% of 100 MW
