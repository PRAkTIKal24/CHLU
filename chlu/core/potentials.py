"""Learnable potential energy functions for CHLU."""

import equinox as eqx
import jax
import jax.nn as jnn
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
        v_g = 0.05 * jnp.sum(q**2)

        return v_n + v_g


class DeepPotentialMLP(eqx.Module):
    """
    High-Capacity Potential for 784-dim MNIST.
    Architecture: 784 -> 1024 -> 1024 -> 1024 -> 1
    Activation: Swish (SiLU) for better gradient flow.
    """

    layers: list

    def __init__(self, dim: int, hidden: int = 1024, key: jax.random.PRNGKey = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # 3 Hidden Layers (Depth = Sharpness)
        self.layers = [
            eqx.nn.Linear(dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, hidden, key=k3),
            eqx.nn.Linear(hidden, 1, key=k4),  # Output scalar Energy
        ]

    def __call__(self, q: jnp.ndarray) -> float:
        x = q

        # Swish Activation (x * sigmoid(x))
        # Much better for Physics/EBMs than tanh or ReLU
        x = jnn.swish(self.layers[0](x))
        x = jnn.swish(self.layers[1](x))
        x = jnn.swish(self.layers[2](x))

        # Final projection to Scalar Energy
        x = self.layers[3](x)
        v_n = jnp.squeeze(x)

        # CRITICAL CHANGE: NO GLOBAL CONFINEMENT (v_g)
        # We rely on jnp.clip(q, -1, 1) in the step function for boundaries.
        # We rely on L2 Regularization in the loss for stability.

        return v_n


class ConvPotential(eqx.Module):
    """
    Convolutional Potential for MNIST.
    Learns 'Local Physics' (edges, strokes) instead of 'Global Pixels'.
    Architecture: Conv layers to detect edges -> strokes -> curves -> digits -> scalar energy
    """

    layers: list

    def __init__(self, key: jax.random.PRNGKey):
        """
        Initialize convolutional potential network.

        Args:
            key: JAX random key for initialization
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.layers = [
            # Layer 1: Detect Edges (Strokes)
            # Input: 1 channel (Greyscale), Output: 16 Features
            eqx.nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1, key=k1),
            # Layer 2: Assemble Strokes into Curves
            eqx.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, key=k2),
            # Layer 3: Assemble Curves into Digits
            eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, key=k3),
            # Layer 4: Global Energy Assessment
            # Flatten -> Linear -> Scalar Energy
            eqx.nn.Linear(64 * 3 * 3, 1, key=k4),
        ]

    def __call__(self, q: jnp.ndarray) -> float:
        """
        Compute potential energy V(q) from image pixels.

        Args:
            q: Flattened image vector (784,)

        Returns:
            Scalar potential energy
        """
        # Reshape flat 784 -> Image (1, 28, 28)
        x = q.reshape(1, 28, 28)

        # Conv Operations with Swish (Smooth Physics)
        x = jnn.swish(self.layers[0](x))  # -> 14x14x16
        x = jnn.swish(self.layers[1](x))  # -> 7x7x32
        x = jnn.swish(self.layers[2](x))  # -> 4x4x64

        # Flatten and Project to Energy
        x = x.ravel()
        E = self.layers[3](x)

        # CRITICAL: Scale down by 100.0 to keep energies in reasonable range.
        # The ConvPotential sums outputs of thousands of neurons.
        # Without this scaling, energy magnitudes explode (e.g., -8000)
        # and temperature/noise parameters become ineffective.
        return jnp.squeeze(E) / 100.0
