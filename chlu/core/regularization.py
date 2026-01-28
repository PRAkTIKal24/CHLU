"""Lyapunov stability regularization for CHLU."""

import jax
import jax.numpy as jnp


def compute_lyapunov_loss(
    step_fn, 
    trajectory: jnp.ndarray, 
    n_samples: int = 10
) -> float:
    """
    Compute Lyapunov regularization loss to penalize chaos.
    
    This regularization enforces stability by penalizing large
    Lyapunov exponents, which are related to the singular values
    of the Jacobian of the dynamics.
    
    Loss = mean(log(singular_values(Jacobian)))
    
    Args:
        step_fn: Function (q, p) -> (q_next, p_next)
        trajectory: Trajectory array of shape (T, 2*dim) [q, p]
        n_samples: Number of points to sample from trajectory
    
    Returns:
        Lyapunov regularization loss (scalar)
    """
    T, state_dim = trajectory.shape
    dim = state_dim // 2
    
    # Sample random timesteps
    indices = jnp.linspace(0, T - 1, n_samples, dtype=jnp.int32)
    
    def compute_jacobian_log_singular_values(idx):
        """Compute log singular values of Jacobian at given trajectory point."""
        state = trajectory[idx]
        _q, _p = state[:dim], state[dim:]
        
        # Wrapper for step function that takes flat state
        def step_wrapper(flat_state):
            q_in = flat_state[:dim]
            p_in = flat_state[dim:]
            q_out, p_out = step_fn((q_in, p_in))
            return jnp.concatenate([q_out, p_out])
        
        # Compute Jacobian
        jacobian = jax.jacfwd(step_wrapper)(state)
        
        # Compute singular values
        singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
        
        # Return mean log singular value (positive values indicate expansion)
        return jnp.mean(jnp.log(singular_values + 1e-8))
    
    # Compute for sampled points
    log_singular_values = jax.vmap(compute_jacobian_log_singular_values)(indices)
    
    # Return mean (we want this close to 0 for stability)
    return jnp.mean(log_singular_values)
