"""
Central configuration management for CHLU.

This module defines all configurable parameters using dataclasses,
providing type safety and defaults for all experiments, training, and models.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, List
import yaml


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    hidden_dim: int = 32
    rest_mass: float = 1.0
    log_mass_init_scale: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 32
    dt: float = 0.01
    lyapunov_lambda: float = 0.01
    sleep_steps: int = 10
    sleep_frequency: int = 5
    buffer_capacity: int = 1024


@dataclass
class ExperimentAConfig:
    """Configuration for Experiment A: Stability Test."""
    train_steps: int = 100
    test_steps: int = 10000
    train_epochs: int = 500
    dt: float = 0.01
    chlu_dim: int = 2
    node_dim: int = 4
    hidden_dim: int = 32


@dataclass
class ExperimentBConfig:
    """Configuration for Experiment B: Noise Rejection."""
    n_waves: int = 100
    steps: int = 200
    train_epochs: int = 500
    dt: float = 0.01
    sigma_min: float = 0.1
    sigma_max: float = 1.0
    n_sigma: int = 10
    chlu_dim: int = 1
    node_dim: int = 2
    hidden_dim: int = 32


@dataclass
class ExperimentCConfig:
    """Configuration for Experiment C: Dreaming/Generation."""
    pca_dim: int = 32
    train_epochs: int = 1000
    n_samples: int = 5000
    dream_steps: int = 100
    friction: float = 0.01
    dt: float = 0.01
    n_dreams: int = 32
    hidden_dim: int = 32
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
    sine_freq_range: Tuple[float, float] = (0.5, 2.0)
    sine_amp_range: Tuple[float, float] = (0.5, 1.5)
    mnist_pca_dim: int = 32
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
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Reconstruct nested dataclasses
    config = CHLUConfig(
        model=ModelConfig(**data.get('model', {})),
        training=TrainingConfig(**data.get('training', {})),
        experiment_a=ExperimentAConfig(**data.get('experiment_a', {})),
        experiment_b=ExperimentBConfig(**data.get('experiment_b', {})),
        experiment_c=ExperimentCConfig(**data.get('experiment_c', {})),
        data=DataConfig(**data.get('data', {})),
        project=ProjectConfig(**data.get('project', {}))
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
        'model': asdict(config.model),
        'training': asdict(config.training),
        'experiment_a': asdict(config.experiment_a),
        'experiment_b': asdict(config.experiment_b),
        'experiment_c': asdict(config.experiment_c),
        'data': asdict(config.data),
        'project': asdict(config.project)
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
