# PyPSA2 Test Guidelines

## Table of Contents
1. [Fixture Architecture](#fixture-architecture)
2. [Writing Tests](#writing-tests)
3. [Test Organization](#test-organization)
4. [Best Practices](#best-practices)

---

## Fixture Architecture

### Pre-configured Test Networks

The test suite provides pre-configured network fixtures that can be modified before `set_snapshots()` is called. This is the key pattern for efficient testing.

```python
# Fixture lifecycle:
# 1. network() - empty network, ready for component addition
# 2. Test modifies network by adding components
# 3. Test applies snapshots via network.set_snapshots(snapshots)
# 4. Test calls network.create_model() if needed
# 5. Test runs optimization and checks results
```

### Available Fixtures

| Fixture | Description | When to Use |
|---------|-------------|-------------|
| `network` | Empty network instance | Basic tests, simple setups |
| `snapshots` | Default 2-time-step snapshots | Tests not requiring custom time series |
| `test_network` | 2-bus system with generator and line (snapshots NOT yet applied) | Most component tests |
| `multi_bus_network` | 3-bus meshed network with mixed components | Integration tests, KCL/KVL verification |
| `storage_test_network` | Single bus with generator, storage unit, and load | Storage behavior tests |
| `battery_test_network` | Single bus with generator, battery, and price signal | Battery optimization tests |

### Modifying Fixtures

**CRITICAL**: All modifications to a fixture must happen BEFORE calling `network.set_snapshots()`. Once snapshots are applied, variables are created and the network state is frozen for variable setup.

```python
def test_my_feature(test_network):
    n, bus1, bus2, gen1, line1 = test_network
    
    # ADD components before set_snapshots - THIS WORKS
    gen2 = n.add(Generator, "Gen2", bus=bus1, p_nom=50.0, marginal_cost=50.0)
    
    # NOW apply snapshots
    n.set_snapshots(pl.Series("time", [0, 1, 2, 3]))
    n.create_model()
    
    # Check existence within the component instance
    assert hasattr(gen2, "p")
```

---

## Writing Tests

### 1. Use Cost Minimization as Default

Prefer minimizing total generation cost as the objective function. This is the most common and intuitive optimization goal.

```python
# GOOD: Cost minimization
n.create_model()
n.model.minimize = gen1.p * 10.0 + gen2.p * 50.0
n.model.optimize()

# AVOID: Fancy objectives unless they test specific behavior
n.model.maximize = battery.p.sum()  # Only if testing battery-specific behavior
```

### 2. Check Existence Within Component Instance

Embrace object-oriented design. Tests should verify properties and methods on the component instance, not reach into the model directly.

```python
# GOOD: Check within component instance
assert hasattr(gen, "p")
assert gen.p_nom == 100.0
assert gen.carrier == "solar"

# AVOID: Checking model attributes directly (unless testing model structure)
assert hasattr(n.model, "p_Gen1")  # Only if testing model naming convention
```

### 3. Test Component Variables and Constraints via Instance

When testing that variables/constraints are created, check the component instance first:

```python
def test_generator_variables(gen):
    """Test that Generator creates p variable."""
    assert hasattr(gen, "p")

def test_generator_constraints(gen):
    """Test that Generator creates limit constraint."""
    n = gen.network
    n.create_model()
    # The constraint exists within the model, but we verify via component
    assert gen.p_nom == 100.0  # Component state is correct
```

### 4. Test Structure Pattern

Each test file should follow this structure:

```python
"""Tests for [Component] component."""
import pytest
import polars as pl
from pypsa2.network import Network
from pypsa2.components import [Component]


class Test[Component]Init:
    """Test [Component] initialization and parameter handling."""
    
    def test_default_parameters(self, network):
        """Test default values are set correctly."""
        
    def test_custom_parameters(self, network):
        """Test custom parameter values are accepted."""


class Test[Component]Variables:
    """Test variable creation for [Component]."""
    
    def test_required_variables(self, network, snapshots):
        """Test that required variables are created."""


class Test[Component]Constraints:
    """Test constraint creation for [Component]."""
    
    def test_required_constraints(self, network, snapshots):
        """Test that required constraints are created."""


class Test[Component]Optimization:
    """Test optimization behavior for [Component]."""
    
    def test_expected_behavior(self, network, snapshots):
        """Test that component behaves correctly in optimization."""
```

---

## Test Organization

### File Structure

| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures, network configurations |
| `test_bus.py` | Bus component tests |
| `test_generator.py` | Generator component tests |
| `test_ac_line.py` | ACLine component tests |
| `test_link.py` | Link component tests |
| `test_storage_unit.py` | StorageUnit base class tests |
| `test_battery.py` | Battery composite class tests |
| `test_pumped_hydro.py` | PumpedHydroStorage composite class tests |
| `test_ramping.py` | Ramping constraints tests |
| `test_error_handling.py` | Error handling and edge cases |
| `test_integration.py` | Multi-component integration tests |

### When to Create New Test Files

- **New component type**: Create a new test file for the component
- **Cross-cutting feature** (e.g., ramping, profiles): Create a dedicated test file
- **Integration scenario**: Create a test file if the scenario involves 3+ component types interacting

---

## Best Practices

### 1. Keep Tests Focused

Each test should verify ONE specific behavior or feature. Avoid testing multiple unrelated things in a single test.

### 2. Remove Redundant Tests

The following types of tests should be removed or consolidated:

- Tests that only verify basic attribute copying (e.g., `assert gen.p_nom == 100.0` without any meaningful behavior)
- Tests that duplicate functionality already covered by other tests
- Tests that only check for the existence of model attributes with naming conventions

### 3. Use Meaningful Test Names

Test names should describe WHAT is being tested, not HOW:

```python
# GOOD
def test_p_min_pu_creates_lower_bound_constraint(network, snapshots):
    """Test that p_min_pu creates gen_lower constraint."""

# AVOID
def test_generator_with_p_min_pu(network, snapshots):
    """Test Generator with p_min_pu."""
```

### 4. Test Realistic Scenarios

For optimization tests, create realistic scenarios that demonstrate the feature's purpose:

```python
def test_battery_arbitrage(network, snapshots):
    """Test battery charges during low-price periods and discharges during high-price."""
    # Create a price signal scenario
    # Verify battery charges when price < marginal_cost of expensive generator
    # Verify battery discharges when price > marginal_cost
```

### 5. Use Fixtures Efficiently

Leverage fixtures to avoid repeating network setup code:

```python
# GOOD: Use fixture and modify
def test_with_extra_component(test_network):
    n, bus1, bus2, gen1, line1 = test_network
    # Add additional component
    ...

# AVOID: Build entire network from scratch
def test_with_extra_component():
    n = Network()
    bus1 = n.add(Bus, "Bus1", v_nom=1.0)
    bus2 = n.add(Bus, "Bus2", v_nom=1.0)
    ...
```

### 6. Solution Access Pattern

When accessing optimization results, use the standard pattern:

```python
# Access solution values from component variables
p_values = gen.p.solution["solution"].to_list()
soc_values = battery.soc.solution["solution"].to_list()
```

### 7. Termination Status Check

Always verify optimization completed successfully:

```python
import pyoptinterface as poi

n.model.optimize()
assert n.model.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL
```

### 8. Skip Tests for Known Issues

When a test cannot pass due to known implementation limitations, mark it as skipped with a clear explanation:

```python
@pytest.mark.skip(reason="Requires mixed-integer formulation (binary variables). "
          "Current LP-only formulation may allow simultaneous charge/discharge.")
def test_no_simultaneous_charge_discharge(self):
    ...
```

---

## Quick Reference: Common Patterns

### Adding a Component with Custom Parameters

```python
gen = n.add(Generator, "Gen1", bus=bus1, p_nom=100.0, marginal_cost=50.0, carrier="gas")
```

### Setting Snapshots and Creating Model

```python
snapshots = pl.Series("time", range(24))
n.set_snapshots(snapshots)
n.create_model()
```

### Running Cost Minimization

```python
n.model.minimize = (gen1.p * gen1.marginal_cost + gen2.p * gen2.marginal_cost).sum()
n.model.optimize()
```

### Adding a Load Constraint

```python
load_df = snapshots.to_frame().with_columns(pl.lit(40.0).alias("load"))
load_param = pf.Param(load_df)
n.model.power_balance = (gen1.p + gen2.p + battery.p >= load_param)
```

### Checking Optimization Results

```python
p_values = gen1.p.solution["solution"].to_list()
assert all(p >= 0 for p in p_values)
```
