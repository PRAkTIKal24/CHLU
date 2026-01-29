"""Learnable potential energy functions for CHLU."""

import equinox as eqx
import jax
import jax.numpy as jnp


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

        # Neural potential
        v_n = jnp.squeeze(x)

        # # Neural potential
        # v_n = 0.5 * jnp.sum(h**2)  # This is bounded (max 0.5 * dim)

        # Global confinement potential
        # This ensures V(q) -> inf as q -> inf
        v_g = 0.001 * jnp.sum(q**2)

        return v_n + v_g
