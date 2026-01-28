# CHLU CLI and Configuration System

## Overview

The CHLU CLI provides a unified interface for managing projects, running experiments, training models, and generating data. All parameters are centralized in a hierarchical configuration system that can be customized per-project.

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Verify installation
uv run chlu --help
```

### Create Your First Project

```bash
# Create a new project
uv run chlu project create myexperiment --description "Testing stability"

# This creates:
# projects/myexperiment/
# ├── config/config.yaml    # Editable configuration
# ├── plots/                # Output visualizations
# ├── results/              # Numerical results
# └── models/               # Trained model checkpoints
```

### Run Experiments

```bash
# Run individual experiments
uv run chlu exp-a --project myexperiment
uv run chlu exp-b --project myexperiment
uv run chlu exp-c --project myexperiment

# Run all experiments
uv run chlu all-experiments --project myexperiment

# Quick mode (reduced epochs for testing)
uv run chlu exp-a --project myexperiment --quick
```

## Available Commands

### Project Management

```bash
# Create a project
uv run chlu project create <name> [--description "..."]

# List all projects
uv run chlu project list

# Delete a project
uv run chlu project delete <name> [--force]
```

### Experiments

| Command | Description |
|---------|-------------|
| `exp-a` | Stability test (100x extrapolation) |
| `exp-b` | Energy-based noise rejection |
| `exp-c` | Generative dreaming (MNIST) |
| `all-experiments` | Run all experiments sequentially |

**Options:**
- `--project <name>` - Use specific project (reads config, saves to project dirs)
- `--seed <int>` - Override random seed
- `--quick` - Quick mode with reduced epochs

### Training

```bash
# Train CHLU model
uv run chlu train chlu --data figure8 --project myproj

# Train baselines
uv run chlu train node --data sine --epochs 500
uv run chlu train lstm --data figure8 --lr 0.001

# Options:
# --project <name>  - Use specific project
# --data {figure8,sine,mnist} - Dataset to train on
# --epochs <int>    - Number of training epochs
# --lr <float>      - Learning rate
```

### Data Generation

```bash
# Generate figure-8 trajectory
uv run chlu data figure8 --steps 1000 --output data.npz

# Generate sine waves
uv run chlu data sine --n-waves 100 --steps 200

# Prepare MNIST with PCA
uv run chlu data mnist --pca-dim 32 --output-dir ./data
```

### Utilities

```bash
# Show system info (Python, JAX, devices)
uv run chlu info

# Show configuration
uv run chlu config show
uv run chlu config show --project myproj
```

## Configuration System

### Structure

The configuration is organized hierarchically:

```yaml
model:                    # Architecture parameters
  hidden_dim: 32
  rest_mass: 1.0
  log_mass_init_scale: 0.1

training:                 # Training hyperparameters
  epochs: 1000
  learning_rate: 0.001
  batch_size: 32
  dt: 0.01
  lyapunov_lambda: 0.01
  sleep_steps: 10
  sleep_frequency: 5
  buffer_capacity: 1024

experiment_a:             # Stability experiment settings
  train_steps: 100
  test_steps: 10000
  train_epochs: 500
  dt: 0.01
  chlu_dim: 2
  node_dim: 4
  hidden_dim: 32

experiment_b:             # Noise rejection settings
  n_waves: 100
  steps: 200
  train_epochs: 500
  sigma_min: 0.1
  sigma_max: 1.0
  n_sigma: 10

experiment_c:             # Dreaming experiment settings
  pca_dim: 32
  train_epochs: 1000
  n_samples: 5000
  dream_steps: 100
  friction: 0.01

data:                     # Data generation parameters
  figure8_dt: 0.01
  sine_freq_min: 0.5
  sine_freq_max: 2.0
  mnist_pca_dim: 32

project:                  # Project-level settings
  name: myproject
  seed: 42
  device: auto
```

### Customizing Configuration

Edit `projects/<your-project>/config/config.yaml`:

```bash
# View current config
uv run chlu config show --project myproj

# Edit manually
nano projects/myproj/config/config.yaml

# Changes take effect immediately on next run
```

### Default vs Project Configuration

- **Without `--project`**: Uses default config values, saves to `./results/`
- **With `--project <name>`**: Loads config from project, saves to project directories

## Project Directory Structure

```
projects/
└── myexperiment/
    ├── config/
    │   └── config.yaml          # Editable configuration
    ├── plots/
    │   ├── exp_a_stability.png
    │   ├── exp_b_noise.png
    │   └── exp_c_dreams.png
    ├── results/
    │   ├── exp_a_metrics.npz
    │   └── exp_b_results.npz
    ├── models/
    │   ├── chlu_figure8.pt
    │   └── chlu_mnist.pt
    ├── project.json             # Metadata
    └── README.md                # Project info
```

## Workflow Examples

### Example 1: Parameter Sweep

```bash
# Create project for each configuration
uv run chlu project create lr_0001 --description "lr=0.001"
uv run chlu project create lr_0010 --description "lr=0.010"

# Edit configs
# projects/lr_0001/config/config.yaml: training.learning_rate = 0.001
# projects/lr_0010/config/config.yaml: training.learning_rate = 0.010

# Run experiments
uv run chlu exp-a --project lr_0001
uv run chlu exp-a --project lr_0010

# Compare results in plots/
```

### Example 2: Quick Testing

```bash
# Create test project
uv run chlu project create quicktest

# Run in quick mode (50 epochs instead of 500)
uv run chlu all-experiments --project quicktest --quick

# Check outputs
ls projects/quicktest/plots/
```

### Example 3: Reproduce Published Results

```bash
# Create project
uv run chlu project create paper_results

# Use default config (already set to paper values)
# Run full experiments
uv run chlu all-experiments --project paper_results --seed 42

# Results saved to projects/paper_results/
```

## Environment Setup

### Using uv (Recommended)

```bash
# Commands automatically use the right environment
uv run chlu <command>
```

### Alternative: Activate Virtual Environment

```bash
# One-time activation per session
source .venv/bin/activate

# Then use directly
chlu exp-a --project myproj
chlu project list
```

### Shell Alias (Optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias chlu='uv run chlu'
```

Then use:

```bash
chlu exp-a --project myproj  # Shorter!
```

## Migration from run_experiments.py

The old `run_experiments.py` is deprecated but still works for backward compatibility:

**Old way:**
```bash
python run_experiments.py --experiment a --save-dir results/ --seed 42 --quick
```

**New way:**
```bash
# Create project
uv run chlu project create myexp

# Run experiment
uv run chlu exp-a --project myexp --seed 42 --quick
```

**Benefits of new system:**
- Organized output directories
- Per-project configuration
- No manual save-dir management
- Model checkpointing
- Project metadata tracking

## Troubleshooting

### Module not found errors

```bash
# Sync dependencies
uv sync

# Verify installation
uv run python -c "from chlu.config import get_default_config; print('OK')"
```

### Config not loading

```bash
# Check config file exists
ls projects/myproj/config/config.yaml

# Validate YAML syntax
uv run python -c "import yaml; yaml.safe_load(open('projects/myproj/config/config.yaml'))"
```

### Project not found

```bash
# List all projects
uv run chlu project list

# Check projects directory
ls -la projects/
```

## Advanced Usage

### Programmatic Access

```python
from chlu.config import get_default_config, load_config
from chlu.project import ProjectManager

# Get default config
config = get_default_config()

# Load project config
pm = ProjectManager()
config = pm.load('myproject')

# Modify and use
config.training.learning_rate = 0.01
config.experiment_a.train_epochs = 100

# Get project paths
paths = pm.get_paths('myproject')
plot_dir = paths['plots']
model_dir = paths['models']
```

### Custom Experiment Scripts

```python
from chlu.project import ProjectManager
from chlu.experiments.exp_a_stability import run_experiment_a

# Load project config
pm = ProjectManager()
config = pm.load('myproject')
paths = pm.get_paths('myproject')

# Run with project settings
run_experiment_a(
    seed=config.project.seed,
    train_steps=config.experiment_a.train_steps,
    train_epochs=config.experiment_a.train_epochs,
    save_dir=str(paths['plots'])
)

# Update metadata
pm.update_last_run('myproject')
```

## See Also

- [Development Guide](code_plan.md) - Code architecture and design
- [Experiment Details](plan-chluImplementation.md) - Experiment descriptions
- [README](../README.md) - Project overview
