"""
Central configuration management for CHLU.

This module defines all configurable parameters using dataclasses,
providing type safety and defaults for all experiments, training, and models.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    hidden_dim: int = 64
    rest_mass: float = 1.0
    log_mass_init_scale: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 32
    dt: float = 0.05
    lyapunov_lambda: float = 0.01
    sleep_steps: int = 200
    clamp_strength: float = 100.0
    clamp_ramp: float = 0.25
    sleep_friction: float = 0.0
    sleep_frequency: int = 5
    buffer_capacity: int = 1024


@dataclass
class ExperimentAConfig:
    """Configuration for Experiment A: Stability Test."""

    # Cycle-based parameters for geometry learning
    dt: float = 0.02  # Time step
    n_train_cycles: int = 3  # Train on 3 full cycles
    n_test_cycles: int = 50  # Test on 50 full cycles
    window_size: int = 64  # Window size for sub-sequence sampling
    n_final_cycles_to_plot: int = 3  # Number of final cycles to show in plots
    train_epochs: int = 1000
    use_pretrained: bool = False  # Load pre-trained models if available
    kinetic_energy_mode: str = "newtonian_identity"  # KE calculation mode
    # Note: chlu_dim is always 2 for Figure-8 (not configurable)
    node_dim: int = 4
    hidden_dim: int = 64
    
    @property
    def steps_per_cycle(self) -> int:
        """Number of steps per cycle (period = 2π)."""
        import math
        return int(2 * math.pi / self.dt)
    
    @property
    def train_steps(self) -> int:
        """Total training steps."""
        return self.n_train_cycles * self.steps_per_cycle
    
    @property
    def test_steps(self) -> int:
        """Total test steps."""
        return self.n_test_cycles * self.steps_per_cycle


@dataclass
class ExperimentBConfig:
    """Configuration for Experiment B: Noise Rejection."""

    n_waves: int = 100
    steps: int = 1000
    train_epochs: int = 1000
    use_pretrained: bool = False  # Load pre-trained models if available
    kinetic_energy_mode: str = "newtonian_learned"  # KE calculation mode
    sleep_friction: float = 0.01
    friction_ramp: float = 0.2
    dt: float = 0.05
    sigma_min: float = 0.1
    sigma_max: float = 1.0
    n_sigma: int = 10
    chlu_dim: int = 1
    node_dim: int = 2
    hidden_dim: int = 64


@dataclass
class ExperimentCConfig:
    """Configuration for Experiment C: Dreaming/Generation."""

    pca_dim: int = 64
    train_epochs: int = 1000
    use_pretrained: bool = False  # Load pre-trained models if available
    kinetic_energy_mode: str = "relativistic"  # KE calculation mode
    n_samples: int = 5000
    dream_steps: int = 100
    friction: float = 0.01
    dt: float = 0.05
    n_dreams: int = 64
    hidden_dim: int = 64
    p_train_scale: float = 0.1
    q_noise_scale: float = 2.0
    p_noise_scale: float = 0.5
    snapshot_steps: List[int] = field(default_factory=lambda: [0, 10, 30, 50, 70, 100])


@dataclass
class DataConfig:
    """Data generation and processing parameters."""

    figure8_dt: float = 0.01
    figure8_scale: float = 1.0
    sine_dt: float = 0.01
    sine_freq_min: float = 0.5
    sine_freq_max: float = 2.0
    sine_amp_min: float = 0.5
    sine_amp_max: float = 1.5
    mnist_pca_dim: int = 64
    train_test_split: float = 0.8


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str = "default"
    seed: int = 42
    device: str = "auto"
    save_dir: Optional[str] = None


@dataclass
class CHLUConfig:
    """Master configuration containing all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_a: ExperimentAConfig = field(default_factory=ExperimentAConfig)
    experiment_b: ExperimentBConfig = field(default_factory=ExperimentBConfig)
    experiment_c: ExperimentCConfig = field(default_factory=ExperimentCConfig)
    data: DataConfig = field(default_factory=DataConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)


def get_default_config() -> CHLUConfig:
    """Get default configuration with all parameters set to their defaults."""
    return CHLUConfig()


def load_config(path: Path) -> CHLUConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        CHLUConfig object with values from file
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    def filter_valid_fields(config_class, data_dict):
        """Filter dict to only include fields that exist in the dataclass."""
        if not data_dict:
            return {}
        valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
        return {k: v for k, v in data_dict.items() if k in valid_fields}

    # Reconstruct nested dataclasses with field filtering
    config = CHLUConfig(
        model=ModelConfig(**filter_valid_fields(ModelConfig, data.get("model", {}))),
        training=TrainingConfig(
            **filter_valid_fields(TrainingConfig, data.get("training", {}))
        ),
        experiment_a=ExperimentAConfig(
            **filter_valid_fields(ExperimentAConfig, data.get("experiment_a", {}))
        ),
        experiment_b=ExperimentBConfig(
            **filter_valid_fields(ExperimentBConfig, data.get("experiment_b", {}))
        ),
        experiment_c=ExperimentCConfig(
            **filter_valid_fields(ExperimentCConfig, data.get("experiment_c", {}))
        ),
        data=DataConfig(**filter_valid_fields(DataConfig, data.get("data", {}))),
        project=ProjectConfig(
            **filter_valid_fields(ProjectConfig, data.get("project", {}))
        ),
    )
    return config


def save_config(config: CHLUConfig, path: Path) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: CHLUConfig object to save
        path: Path where to save the YAML file
    """
    # Convert to nested dict
    config_dict = {
        "model": asdict(config.model),
        "training": asdict(config.training),
        "experiment_a": asdict(config.experiment_a),
        "experiment_b": asdict(config.experiment_b),
        "experiment_c": asdict(config.experiment_c),
        "data": asdict(config.data),
        "project": asdict(config.project),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
