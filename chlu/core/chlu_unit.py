"""Causal Hamiltonian Learning Unit (CHLU) - Core Implementation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.core.integrators import velocity_verlet_step
from chlu.core.potentials import PotentialMLP


class CHLU(eqx.Module):
    """
    Causal Hamiltonian Learning Unit.

    A dynamical system grounded in symplectic mechanics with a relativistic
    Hamiltonian that ensures energy stability and causal bounds.

    Hamiltonian:
        H(q, p) = sqrt(p^T M p + m^2) + V(q)

    where:
        - M: learnable positive-definite mass matrix (diagonal)
        - m: rest mass constant
        - V(q): learnable potential function (MLP)
    """

    potential_net: PotentialMLP
    log_mass: jnp.ndarray  # Log-parameterized for positivity
    rest_mass: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden: int = 32,
        rest_mass: float = 1.0,
        key: jax.random.PRNGKey = None,
    ):
        """
        Initialize CHLU.

        Args:
            dim: Dimensionality of position/momentum space
            hidden: Hidden units in potential network (default: 32)
            rest_mass: Rest mass constant m (default: 1.0)
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        k1, k2 = jax.random.split(key, 2)

        self.dim = dim
        self.rest_mass = rest_mass

        # Initialize potential network
        self.potential_net = PotentialMLP(dim, hidden, key=k1)

        # Initialize log mass (use log for positive-definiteness via softplus)
        self.log_mass = jax.random.normal(k2, (dim,)) * 0.1

    def H(self, q: jnp.ndarray, p: jnp.ndarray) -> float:
        """
        Compute the relativistic Hamiltonian.

        H(q, p) = sqrt(p^T M p + m^2) + V(q)

        Args:
            q: Position (dim,)
            p: Momentum (dim,)

        Returns:
            Total energy (scalar)
        """
        # Ensure mass matrix is positive-definite
        M = jax.nn.softplus(self.log_mass)

        # Relativistic kinetic energy: sqrt(p^T M p + m^2)
        kinetic = jnp.sqrt(jnp.sum(p * M * p) + self.rest_mass**2)

        # Potential energy
        potential = self.potential_net(q)

        return kinetic + potential

    def step(self, state: tuple, dt: float, gamma: float = 0.0) -> tuple:
        """
        Single time step using Velocity Verlet integrator.

        Args:
            state: (q, p) tuple
            dt: Time step

        Returns:
            (q_next, p_next): Updated state
        """
        q, p = state
        return velocity_verlet_step(self.H, q, p, dt, gamma)

    def __call__(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        gamma: float = 0.0,
    ) -> jnp.ndarray:
        """
        Unroll trajectory using jax.lax.scan for efficiency.

        Args:
            q0: Initial position (dim,)
            p0: Initial momentum (dim,)
            steps: Number of time steps
            dt: Time step size

        Returns:
            Trajectory of shape (steps, 2*dim) where each row is [q, p]
        """

        def scan_fn(state, _):
            q, p = state
            q_next, p_next = self.step((q, p), dt, gamma)
            # Concatenate q and p for output
            output = jnp.concatenate([q_next, p_next])
            return (q_next, p_next), output

        # Run scan
        _, trajectory = jax.lax.scan(scan_fn, (q0, p0), None, length=steps)

        return trajectory
