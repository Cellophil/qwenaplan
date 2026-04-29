# qwenaplan Project Structure

## Overview

The project "qwenaplan" is located at `/Users/andreas/pypsa/qwenaplan`. This is where development happens.

The workspace also contains git clones of PyPSA and pyoframe for code lookup if necessary. qwenaplan is written in pyoframe/polars, but motivated by PyPSA (which uses linopy / pandas).

## Directory Structure

```
/Users/andreas/pypsa/
├── qwenaplan/               # Main project (modern Python package)
│   ├── pyproject.toml       # Package configuration
│   ├── README.md            # Project documentation
│   ├── src/
│   │   └── qwenaplan/      # Package source code
│   │       ├── __init__.py
│   │       ├── base.py      # Base component classes
│   │       ├── components.py # Network components
│   │       ├── network.py   # Network class
│   │       └── physics.py   # Physics engine
│   ├── tests/               # Test suite
│   │   ├── conftest.py
│   │   └── test_*.py
│   └── .roo/
│       └── rules/          # LLM configuration rules
├── PyPSA/                   # Reference implementation (linopy/pandas)
└── pyoframe/                # Optimization framework dependency
```

## Key Points

1. **Source code** is in `src/qwenaplan/` (modern Python package layout)
2. **Tests** are in `tests/` directory
3. **Conda environment** is `highs` (must be activated for running tests)
4. **Package is installable** via `pip install -e ".[dev]"`

## Running Tests

```bash
# Activate conda and run all tests
source ~/miniforge3/etc/profile.d/conda.sh && conda activate highs && cd /Users/andreas/pypsa/qwenaplan && pytest

# Run specific test file
source ~/miniforge3/etc/profile.d/conda.sh && conda activate highs && cd /Users/andreas/pypsa/qwenaplan && pytest tests/test_bus.py -v
```
