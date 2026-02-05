"""Symplectic integrators for Hamiltonian dynamics."""

import jax
import jax.numpy as jnp


def velocity_verlet_step(
    H_fn, q: jnp.ndarray, p: jnp.ndarray, dt: float, gamma: float = 0.0
) -> tuple:
    """
    Velocity Verlet (Leapfrog) symplectic integrator.

    This integrator preserves phase space volume (det(Jacobian) = 1)
    and approximately conserves energy over long trajectories.

    Algorithm:
        1. p_half = p - 0.5 * dt * ∂H/∂q(q, p)
        2. q_next = q + dt * ∂H/∂p(q, p_half)
        3. p_next = p_half - 0.5 * dt * ∂H/∂q(q_next, p_half)

    Args:
        H_fn: Hamiltonian function H(q, p) -> scalar
        q: Position (dim,)
        p: Momentum (dim,)
        dt: Time step
        gamma: Friction coefficient (default: 0.0, no friction)

    Returns:
        (q_next, p_next): Updated state
    """
    # Compute gradients of Hamiltonian
    # ∂H/∂q and ∂H/∂p
    grad_H_q = jax.grad(H_fn, argnums=0)
    grad_H_p = jax.grad(H_fn, argnums=1)

    # Half-step momentum update
    p_half = p - 0.5 * dt * grad_H_q(q, p)

    # Full-step position update using half-step momentum
    q_next = q + dt * grad_H_p(q, p_half)

    # Half-step momentum update to complete the step
    p_next = p_half - 0.5 * dt * grad_H_q(q_next, p_half)

    # Apply friction if gamma > 0
    p_next = (1.0 - gamma) * p_next

    return q_next, p_next


def langevin_step(
    H_fn,
    q: jnp.ndarray,
    p: jnp.ndarray,
    dt: float,
    gamma: float,
    temperature: float,
    key: jax.random.PRNGKey,
) -> tuple:
    """
    Velocity Verlet integrator with Langevin thermal noise.

    Extends the deterministic Velocity Verlet algorithm with temperature-scaled
    Gaussian noise following the fluctuation-dissipation theorem. This allows the
    system to explore energy landscapes rather than deterministically settling into
    the nearest minimum.

    Physical interpretation:
        - At temperature=0: Identical to velocity_verlet_step with friction
        - At temperature>0: Particles undergo Brownian motion, can escape local minima
        - Noise scale follows fluctuation-dissipation: sqrt(2 * gamma * T * dt)

    Algorithm:
        1. p_half = p - 0.5 * dt * ∂H/∂q(q, p)
        2. q_next = q + dt * ∂H/∂p(q, p_half)
        3. p_next = p_half - 0.5 * dt * ∂H/∂q(q_next, p_half)
        4. p_next = (1 - gamma) * p_next  [friction]
        5. p_next += sqrt(2 * gamma * T * dt) * N(0,1)  [thermal noise]

    Args:
        H_fn: Hamiltonian function H(q, p) -> scalar
        q: Position (dim,)
        p: Momentum (dim,)
        dt: Time step
        gamma: Friction coefficient (must be > 0 for temperature to have effect)
        temperature: Temperature parameter (0 = deterministic, >0 = stochastic)
        key: JAX random key for noise generation

    Returns:
        (q_next, p_next, new_key): Updated state and split key for future use
    """
    # Compute gradients of Hamiltonian
    grad_H_q = jax.grad(H_fn, argnums=0)
    grad_H_p = jax.grad(H_fn, argnums=1)

    # Half-step momentum update
    p_half = p - 0.5 * dt * grad_H_q(q, p)

    # Full-step position update using half-step momentum
    q_next = q + dt * grad_H_p(q, p_half)

    # Half-step momentum update to complete the step
    p_next = p_half - 0.5 * dt * grad_H_q(q_next, p_half)

    # Apply friction
    p_next = (1.0 - gamma) * p_next

    # Add Langevin thermal noise
    # Fluctuation-dissipation theorem: noise_scale = sqrt(2 * gamma * kT * dt)
    # (Here temperature already includes Boltzmann constant k)
    # Always split key and compute noise, but use jnp.where to conditionally apply
    # This ensures the function is traceable in JAX (no Python conditionals on traced values)
    key, subkey = jax.random.split(key)
    noise_scale = jnp.sqrt(jnp.maximum(0.0, 2.0 * gamma * temperature * dt))
    noise = jax.random.normal(subkey, p.shape) * noise_scale
    # Apply noise only if temperature > 0 (using jnp.where for traceability)
    p_next = jnp.where(temperature > 0.0, p_next + noise, p_next)

    return q_next, p_next, key


def get_temperature_schedule(
    start: float, end: float, steps: int, schedule_type: str = "exponential"
) -> jnp.ndarray:
    """
    Generate a temperature annealing schedule.

    Creates an array of temperature values that decrease from start to end over
    the specified number of steps. Used for simulated annealing where the system
    starts hot (high exploration) and cools down (converges to minima).

    Args:
        start: Initial temperature (high for exploration)
        end: Final temperature (low for convergence)
        steps: Number of temperature values to generate
        schedule_type: Type of decay schedule
            - "exponential": T(t) = start * (end/start)^(t/steps)
                            Cools slowly at first, faster near the end
            - "linear": T(t) = start - (start-end) * (t/steps)
                       Constant cooling rate

    Returns:
        Temperature array of shape (steps,)

    Example:
        >>> temps = get_temperature_schedule(1.0, 0.01, 1000, "exponential")
        >>> temps[0]   # 1.0
        >>> temps[500] # ~0.1
        >>> temps[-1]  # 0.01
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if start <= 0 or end <= 0:
        raise ValueError(f"Temperatures must be positive, got start={start}, end={end}")

    t = jnp.linspace(0, 1, steps)

    if schedule_type == "exponential":
        # Exponential decay: T(t) = start * (end/start)^t
        # This gives slower cooling initially, faster later
        return start * jnp.power(end / start, t)

    elif schedule_type == "linear":
        # Linear decay: T(t) = start - (start - end) * t
        return start - (start - end) * t

    else:
        raise ValueError(
            f"Unknown schedule_type: {schedule_type}. Must be 'exponential' or 'linear'."
        )
