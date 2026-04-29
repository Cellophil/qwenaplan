# qwenaplan

A modern power system optimization framework built with Polars and PyOframe.

## Installation

### Development Installation

```bash
# Activate your conda environment
conda activate highs

# Install in editable mode
cd qwenaplan
pip install -e ".[dev]"
```

### Regular Installation

```bash
pip install qwenaplan
```

## Quick Start

```python
import polars as pl
from qwenaplan import Network, Bus, Generator, ACLine

# Create a network
network = Network()

# Add buses
bus1 = network.add(Bus, "Bus1", v_nom=1.0)
bus2 = network.add(Bus, "Bus2", v_nom=1.0)

# Add generator
gen1 = network.add(Generator, "Gen1", bus=bus1, p_nom=100.0)

# Add line
line1 = network.add(ACLine, "Line1", from_bus=bus1, to_bus=bus2, x_pu=0.1)

# Set snapshots
snapshots = pl.Series(["2026-01-01 00:00", "2026-01-01 01:00"])
network.set_snapshots(snapshots)

# Create optimization model
network.create_model()
```

## Running Tests

```bash
# Activate your conda environment
conda activate highs

# Run tests
cd qwenaplan
pytest
```

## Project Structure

```
qwenaplan/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── src/
│   └── qwenaplan/         # Package source code
│       ├── __init__.py
│       ├── base.py        # Base component classes
│       ├── components.py  # Network components (Bus, Generator, etc.)
│       ├── network.py     # Network class
│       └── physics.py     # Physics engine (DC power flow)
└── tests/                  # Test suite
    ├── conftest.py
    └── test_*.py
```

## Dependencies

- Python >= 3.9
- polars
- pyoframe

## License

MIT
