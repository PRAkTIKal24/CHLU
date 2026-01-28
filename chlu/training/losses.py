"""Loss functions for training."""

import jax
import jax.numpy as jnp


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    """
    Mean squared error loss.
    
    Args:
        pred: Predictions
        target: Ground truth
    
    Returns:
        MSE loss (scalar)
    """
    return jnp.mean((pred - target) ** 2)


def energy_loss(model, q: jnp.ndarray, p: jnp.ndarray) -> float:
    """
    Energy minimization loss for CHLU sleep phase.
    
    Encourages states to settle into low-energy configurations
    (the "energy valley").
    
    Args:
        model: CHLU model with H(q, p) method
        q: Position batch (batch_size, dim)
        p: Momentum batch (batch_size, dim)
    
    Returns:
        Mean energy (scalar)
    """
    # Compute energy for each state in batch
    energies = jax.vmap(model.H)(q, p)
    
    return jnp.mean(energies)
