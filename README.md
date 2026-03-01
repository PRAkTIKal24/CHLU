![Release-version](https://img.shields.io/github/v/tag/PRAkTIKal24/CHLU?include_prereleases&label=latest%20release&color=blue)
![GitHub Release-date](https://img.shields.io/github/release-date-pre/PRAkTIKal24/CHLU?style=flat&color=blue)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub forks](https://img.shields.io/github/forks/PRAkTIKal24/CHLU?style=flat&color=blue)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/PRAkTIKal24/CHLU)

[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


# CHLU
Causal Hamiltonian Learning Unit

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/PRAkTIKal24/CHLU.git
cd CHLU

# Install dependencies with uv
uv sync
```

### Run Experiments

```bash
# Create a project
uv run chlu project create myexperiment

# Run experiments
uv run chlu exp-a --project myexperiment  # Stability test
uv run chlu exp-b --project myexperiment  # Noise rejection
uv run chlu exp-c --project myexperiment  # Generative dreaming
uv run chlu all-experiments --project myexperiment  # Run all

# Quick mode (reduced epochs for testing)
uv run chlu exp-a --project myexperiment --quick
```

### View System Info

```bash
uv run chlu info
uv run chlu --help
```

## Documentation

- **[CLI Guide](docs/CLI_GUIDE.md)** - Complete CLI documentation and usage examples

## Features

- **Project Management**: Organized workspace with automatic directory structure
- **Centralized Configuration**: YAML-based config system with per-project customization
- **Three Experiments**:
  - Experiment A: Long-term stability (100x extrapolation)
  - Experiment B: Energy-based noise rejection
  - Experiment C: Generative dreaming with MNIST
- **Multiple Models**: CHLU, Neural ODE, LSTM baselines
- **Flexible CLI**: Run experiments, train models, generate data

## Project Structure

```
projects/<name>/          # Your projects
  ├── config/             # Configuration files
  ├── plots/              # Generated visualizations
  ├── results/            # Experiment results
  └── models/             # Trained model checkpoints

chlu/                     # Main package
  ├── core/               # CHLU unit, baselines, integrators
  ├── experiments/        # Experiment implementations
  ├── training/           # Training loops and losses
  ├── data/               # Data generation
  ├── cli/                # Command-line interface
  └── utils/              # Plotting and metrics
```

## License

See [LICENSE](LICENSE) file for details.
