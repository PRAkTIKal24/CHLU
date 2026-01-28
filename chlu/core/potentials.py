"""Learnable potential energy functions for CHLU."""

import jax
import jax.numpy as jnp
import equinox as eqx


class PotentialMLP(eqx.Module):
    """
    V(q) - Learnable potential energy function.
    
    A simple MLP that maps position q to scalar potential energy.
    Architecture: Linear(dim→hidden) → tanh → Linear(hidden→hidden) → tanh → Linear(hidden→1)
    """
    
    layers: list
    
    def __init__(self, dim: int, hidden: int = 32, key: jax.random.PRNGKey = None):
        """
        Initialize the potential network.
        
        Args:
            dim: Dimensionality of position space
            hidden: Number of hidden units (default: 32)
            key: JAX random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        keys = jax.random.split(key, 3)
        
        self.layers = [
            eqx.nn.Linear(dim, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, 1, key=keys[2]),
        ]
    
    def __call__(self, q: jnp.ndarray) -> float:
        """
        Compute potential energy V(q).
        
        Args:
            q: Position vector (dim,)
        
        Returns:
            Scalar potential energy
        """
        x = q
        # First layer + activation
        x = self.layers[0](x)
        x = jnp.tanh(x)
        
        # Second layer + activation
        x = self.layers[1](x)
        x = jnp.tanh(x)
        
        # Output layer (scalar)
        x = self.layers[2](x)
        
        # Return scalar (squeeze the output)
        return jnp.squeeze(x)
