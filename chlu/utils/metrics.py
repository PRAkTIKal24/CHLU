"""Metrics and tracking utilities."""

import jax
import jax.numpy as jnp


def compute_mse(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    """
    Compute mean squared error.
    
    Args:
        pred: Predictions
        target: Ground truth
    
    Returns:
        MSE (scalar)
    """
    return jnp.mean((pred - target) ** 2)


def track_energy(model, trajectory: jnp.ndarray) -> jnp.ndarray:
    """
    Track energy along a CHLU trajectory.
    
    Args:
        model: CHLU model with H(q, p) method
        trajectory: Trajectory (T, 2*dim) [q, p]
    
    Returns:
        Energy at each timestep (T,)
    """
    dim = trajectory.shape[1] // 2
    
    def compute_energy_single(state):
        q, p = state[:dim], state[dim:]
        return model.H(q, p)
    
    energies = jax.vmap(compute_energy_single)(trajectory)
    
    return energies


def count_params(model) -> int:
    """
    Count number of parameters in an Equinox model.
    
    Args:
        model: Equinox model
    
    Returns:
        Total number of parameters
    """
    import equinox as eqx
    
    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    
    return sum(x.size for x in leaves)
