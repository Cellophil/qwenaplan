# Conda Setup for qwenaplan

## Environment

The project uses a conda environment named `highs` with miniforge3.

## Running Tests

### Option 1: One-liner (for LLMs)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate highs && cd /Users/andreas/pypsa/qwenaplan && pytest
```

### Option 2: Interactive (for humans)

```bash
conda activate highs
cd /Users/andreas/pypsa/qwenaplan
pytest
```

## Development Installation

After activating the conda environment:

```bash
cd /Users/andreas/pypsa/qwenaplan
pip install -e ".[dev]"
```

This installs the package in editable mode with development dependencies.

## Running a Single Test File

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate highs && cd /Users/andreas/pypsa/qwenaplan && pytest tests/test_bus.py -v
```
