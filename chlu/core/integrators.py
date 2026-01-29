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
