# CHLU Projects

This directory contains your CHLU projects. Each project has its own configuration, results, plots, and trained models.

## Creating a New Project

```bash
uv run chlu project create myproject
```

This creates a directory structure:

```
projects/myproject/
├── config/
│   └── config.yaml      # Editable configuration
├── plots/               # Generated visualizations
├── results/             # Experiment results (numpy arrays, metrics)
├── models/              # Trained model checkpoints
├── project.json         # Project metadata
└── README.md            # Project information
```

## Running Experiments

```bash
# Run experiments in a specific project
uv run chlu exp-a --project myproject
uv run chlu exp-b --project myproject
uv run chlu exp-c --project myproject

# Run all experiments
uv run chlu all-experiments --project myproject
```

## Listing Projects

```bash
uv run chlu project list
```

## Customizing Configuration

Edit the `config/config.yaml` file in your project directory to customize parameters:

- Model architecture (hidden dimensions, rest mass)
- Training hyperparameters (learning rate, epochs, batch size)
- Experiment-specific settings
- Data generation parameters

Changes to the configuration will be automatically picked up when running experiments in that project.

## Project Management

```bash
# Create a project
uv run chlu project create <name> --description "My experiment"

# List all projects
uv run chlu project list

# Delete a project
uv run chlu project delete <name>
```
