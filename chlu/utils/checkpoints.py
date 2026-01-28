"""Model checkpointing utilities for CHLU."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import equinox as eqx


def save_model(
    model, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a model to disk using Equinox's serialization.

    Args:
        model: Model to save (CHLU, NeuralODE, or LSTMPredictor)
        path: Path to save the model (.pkl extension recommended)
        metadata: Optional metadata dictionary (config, metrics, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {"model": model, "metadata": metadata or {}}

    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, checkpoint)


def load_model(path: Union[str, Path], model_template: Optional[Any] = None) -> Any:
    """
    Load a model from disk.

    Args:
        path: Path to the saved model
        model_template: Template model with same structure (if None, returns raw data)

    Returns:
        Loaded model (if template provided) or checkpoint dict
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        if model_template is not None:
            checkpoint_template = {"model": model_template, "metadata": {}}
            checkpoint = eqx.tree_deserialise_leaves(f, checkpoint_template)
            return checkpoint["model"]
        else:
            # Load raw data
            checkpoint = pickle.load(f)
            return checkpoint


def save_checkpoint(
    model,
    path: Union[str, Path],
    epoch: int,
    loss: float,
    config: Optional[Any] = None,
    **kwargs,
) -> None:
    """
    Save a training checkpoint with metadata.

    Args:
        model: Model to save
        path: Path to save checkpoint
        epoch: Current training epoch
        loss: Current loss value
        config: Configuration object
        **kwargs: Additional metadata to save
    """
    metadata = {"epoch": epoch, "loss": float(loss), "config": config, **kwargs}

    save_model(model, path, metadata)


def load_checkpoint(
    path: Union[str, Path], model_template: Optional[Any] = None
) -> tuple:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint
        model_template: Template model for structure

    Returns:
        (model, metadata) tuple
    """
    path = Path(path)

    with open(path, "rb") as f:
        if model_template is not None:
            checkpoint_template = {"model": model_template, "metadata": {}}
            checkpoint = eqx.tree_deserialise_leaves(f, checkpoint_template)
            return checkpoint["model"], checkpoint["metadata"]
        else:
            checkpoint = pickle.load(f)
            return checkpoint["model"], checkpoint.get("metadata", {})


def list_checkpoints(directory: Union[str, Path], pattern: str = "*.pkl") -> list:
    """
    List all checkpoint files in a directory.

    Args:
        directory: Directory to search
        pattern: File pattern to match

    Returns:
        List of Path objects for checkpoints
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    return sorted(directory.glob(pattern))


def get_latest_checkpoint(
    directory: Union[str, Path], pattern: str = "*.pkl"
) -> Optional[Path]:
    """
    Get the most recent checkpoint file in a directory.

    Args:
        directory: Directory to search
        pattern: File pattern to match

    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoints = list_checkpoints(directory, pattern)
    if not checkpoints:
        return None

    # Sort by modification time
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def create_checkpoint_name(
    base_name: str,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    suffix: str = ".pkl",
) -> str:
    """
    Create a standardized checkpoint filename.

    Args:
        base_name: Base name for the checkpoint (e.g., "chlu", "node")
        epoch: Training epoch (optional)
        loss: Loss value (optional)
        suffix: File suffix

    Returns:
        Formatted checkpoint filename

    Examples:
        >>> create_checkpoint_name("chlu", epoch=100, loss=0.0123)
        'chlu_epoch100_loss0.0123.pkl'
        >>> create_checkpoint_name("node")
        'node.pkl'
    """
    parts = [base_name]

    if epoch is not None:
        parts.append(f"epoch{epoch}")

    if loss is not None:
        parts.append(f"loss{loss:.4f}")

    return "_".join(parts) + suffix
