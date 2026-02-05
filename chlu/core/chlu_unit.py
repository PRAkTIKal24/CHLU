"""Causal Hamiltonian Learning Unit (CHLU) - Core Implementation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.core.integrators import velocity_verlet_step, langevin_step
from chlu.core.potentials import PotentialMLP, DeepPotentialMLP, ConvPotential


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

    potential_net: eqx.Module  # PotentialMLP, DeepPotentialMLP, or ConvPotential
    log_mass: jnp.ndarray  # Log-parameterized for positivity
    rest_mass: float = eqx.field(static=True)
    c: float = eqx.field(static=True)  # Speed of causality
    dim: int = eqx.field(static=True)
    kinetic_mode: str = eqx.field(
        static=True
    )  # "newtonian_identity", "newtonian_learned", "relativistic"
    potential_type: str = eqx.field(static=True)  # "mlp", "deep_mlp", "conv"

    def __init__(
        self,
        dim: int,
        hidden: int = 32,
        rest_mass: float = 1.0,
        c: float = 1.0,
        kinetic_mode: str = "newtonian_identity",
        potential_type: str = "mlp",
        key: jax.random.PRNGKey = None,
    ):
        """
        Initialize CHLU.

        Args:
            dim: Dimensionality of position/momentum space
            hidden: Hidden units in potential network (default: 32)
            rest_mass: Rest mass constant m (default: 1.0)
            c: Speed of causality (default: 1.0)
            kinetic_mode: Kinetic energy calculation mode (default: "newtonian_identity")
                         Options: "newtonian_identity", "newtonian_learned", "relativistic"
            potential_type: Potential network architecture (default: "mlp")
                           Options: "mlp" (standard), "deep_mlp" (high-capacity), "conv" (convolutional)
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        k1, k2 = jax.random.split(key, 2)

        self.dim = dim
        self.rest_mass = rest_mass
        self.c = c
        self.kinetic_mode = kinetic_mode
        self.potential_type = potential_type

        # Initialize potential network based on potential_type
        if potential_type == "conv":
            # Convolutional network for spatial structure (MNIST images)
            if dim != 784:
                raise ValueError(
                    f"ConvPotential requires dim=784 (28x28 images), got dim={dim}"
                )
            self.potential_net = ConvPotential(key=k1)
        elif potential_type == "deep_mlp":
            # High-capacity MLP for high-dimensional data
            self.potential_net = DeepPotentialMLP(dim, hidden, key=k1)
        elif potential_type == "mlp":
            # Standard MLP for low-dimensional dynamics
            self.potential_net = PotentialMLP(dim, hidden, key=k1)
        else:
            raise ValueError(
                f"Unknown potential_type: {potential_type}. "
                f"Must be 'mlp', 'deep_mlp', or 'conv'."
            )

        # Initialize log mass (use log for positive-definiteness via softplus)
        self.log_mass = jax.random.normal(k2, (dim,)) * 0.1

    def H(self, q: jnp.ndarray, p: jnp.ndarray) -> float:
        """
        Compute the Hamiltonian with selectable kinetic energy mode.

        H(q, p) = T(p) + V(q)

        Where T(p) depends on kinetic_mode:
        - "newtonian_identity": T = 0.5 * p^2 (identity mass, classic)
        - "newtonian_learned": T = 0.5 * p^T M^-1 p (learned mass, classic)
        - "relativistic": T = sqrt(p^T M^-1 p + m^2) (learned mass, relativistic)

        Args:
            q: Position (dim,)
            p: Momentum (dim,)

        Returns:
            Total energy (scalar)
        """
        # Compute mass vector (always prepared, used if needed)
        M = jax.nn.softplus(self.log_mass)  # Ensure positive-definite
        M_inv = 1.0 / (M + 1e-6)  # Inverse mass with numerical stability

        # Select kinetic energy calculation based on mode
        if self.kinetic_mode == "newtonian_identity":
            # Classic T = 0.5 * p^2 (identity mass)
            # Best for: Lemniscate/Figure-8 to preserve geometric properties
            kinetic = 0.5 * jnp.sum(p * p)

        elif self.kinetic_mode == "newtonian_learned":
            # T = 0.5 * p^T M^-1 p (learned diagonal mass)
            # Best for: Systems with varying inertia across dimensions
            kinetic = 0.5 * jnp.sum((p * p) * M_inv)

        elif self.kinetic_mode == "relativistic":
            # T = c(sqrt(p^T M^-1 p + (mc)^2)) (relativistic with learned mass)
            # Best for: High-dimensional systems, bounded velocities, noise robustness
            p_norm_squared = jnp.sum((p * p) * M_inv)

            # Compute rest energy term
            rest_energy = (self.rest_mass * self.c) ** 2

            kinetic = self.c * jnp.sqrt(p_norm_squared + rest_energy)

        else:
            raise ValueError(
                f"Unknown kinetic mode: {self.kinetic_mode}. "
                f"Must be 'newtonian_identity', 'newtonian_learned', or 'relativistic'."
            )

        # Potential energy (always computed the same way)
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

    def stochastic_step(
        self,
        state: tuple,
        dt: float,
        gamma: float,
        temperature: float,
        key: jax.random.PRNGKey,
    ) -> tuple:
        """
        Single stochastic time step using Langevin dynamics.

        Adds temperature-scaled Gaussian noise to enable exploration of the
        energy landscape. The system can escape local minima and discover
        multiple modes in the distribution.

        Args:
            state: (q, p) tuple
            dt: Time step
            gamma: Friction coefficient (required for temperature to have effect)
            temperature: Temperature parameter (0 = deterministic, >0 = stochastic)
            key: JAX random key for reproducible noise generation

        Returns:
            (q_next, p_next, new_key): Updated state and split key
        """
        q, p = state
        return langevin_step(self.H, q, p, dt, gamma, temperature, key)

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

    def governed_rollout(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        target_energy: float,
        sensitivity: float = 1.0,
    ) -> jnp.ndarray:
        """
        Unroll trajectory with energy-based governor (active limit cycle control).

        The governor dynamically adjusts friction based on energy error:
        - If current_energy > target_energy (noisy): Apply positive friction (brake)
        - If current_energy < target_energy (damped): Coast (maintain energy)

        This creates a Van der Pol-like limit cycle attractor at the target energy.

        Args:
            q0: Initial position (dim,)
            p0: Initial momentum (dim,)
            steps: Number of time steps
            dt: Time step size
            target_energy: Target Hamiltonian energy (learned from training data)
            sensitivity: Governor sensitivity (default: 1.0). Controls correction speed.

        Returns:
            Trajectory of shape (steps, 2*dim) where each row is [q, p]
        """

        def scan_fn(state, _):
            q, p = state

            # Compute current energy
            current_energy = self.H(q, p)

            # Energy error: positive if above target (noise), negative if below (damped)
            energy_error = current_energy - target_energy

            # Symmetric control: tanh clamps to [-1, 1] for stability
            # Positive error → positive gamma (friction/brake)
            # Negative error → zero gamma (frictionless coasting)
            gamma = sensitivity * jnp.tanh(jnp.maximum(0, energy_error))

            # Step with dynamic gamma
            q_next, p_next = self.step((q, p), dt, gamma)

            # Concatenate q and p for output
            output = jnp.concatenate([q_next, p_next])
            return (q_next, p_next), output

        # Run scan for steps-1, then prepend initial condition
        # This ensures output length = steps and includes (q0, p0) as first point
        _, trajectory = jax.lax.scan(scan_fn, (q0, p0), None, length=steps - 1)

        # Prepend initial condition to match LSTM/NODE behavior
        initial_state = jnp.concatenate([q0, p0])[None, :]
        return jnp.concatenate([initial_state, trajectory], axis=0)

    def stochastic_rollout(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        gamma: float,
        temperature: float | jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Unroll stochastic trajectory using Langevin dynamics.

        This method uses jax.lax.scan for efficiency while properly threading
        the random key through each step to ensure reproducible stochastic behavior.

        Args:
            q0: Initial position (dim,)
            p0: Initial momentum (dim,)
            steps: Number of time steps
            dt: Time step size
            gamma: Friction coefficient
            temperature: Temperature parameter for thermal noise.
                        Can be scalar (constant) or array of shape (steps,) for annealing.
            key: JAX random key for reproducible stochastic evolution

        Returns:
            Trajectory of shape (steps, 2*dim) where each row is [q, p]
        """
        # Convert temperature to array schedule
        # If scalar, broadcast to constant schedule
        temp_schedule = jnp.atleast_1d(temperature)
        if temp_schedule.shape[0] == 1:
            temp_schedule = jnp.repeat(temp_schedule, steps)
        elif temp_schedule.shape[0] != steps:
            raise ValueError(f"Temperature schedule length {temp_schedule.shape[0]} must match steps {steps}")

        def scan_fn(carry, temp_t):
            q, p, key_state = carry
            q_next, p_next, new_key = self.stochastic_step(
                (q, p), dt, gamma, temp_t, key_state
            )
            # Concatenate q and p for output
            output = jnp.concatenate([q_next, p_next])
            return (q_next, p_next, new_key), output

        # Run scan with key threading, iterating over temperature schedule
        _, trajectory = jax.lax.scan(scan_fn, (q0, p0, key), temp_schedule)

        return trajectory
