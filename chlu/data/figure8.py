"""Figure-8 (Lemniscate of Bernoulli) trajectory generator."""

import jax
import jax.numpy as jnp


def generate_figure8(
    key: jax.random.PRNGKey,
    n_cycles: int,
    dt: float = 0.02,
    scale: float = 1.0,
) -> jnp.ndarray:
    """
    Generate Lemniscate of Bernoulli (Figure-8 curve) trajectory.
    
    Parametric equations:
        x(t) = scale * sin(t) / (1 + cos²(t))
        y(t) = scale * sin(t) * cos(t) / (1 + cos²(t))
    
    Also computes velocities analytically: vx = dx/dt, vy = dy/dt
    
    Args:
        key: JAX random key (for potential randomization)
        n_cycles: Number of complete cycles to generate (period = 2π)
        dt: Time step size
        scale: Scaling factor for the curve
    
    Returns:
        Trajectory of shape (total_steps, 4): [x, y, vx, vy] ≡ [q, p]
        where total_steps = n_cycles * steps_per_cycle
    """
    # Compute total number of steps for n_cycles
    steps_per_cycle = int(2 * jnp.pi / dt)
    total_steps = n_cycles * steps_per_cycle
    
    # Time parameter using actual time-stepping (not linspace)
    t = jnp.arange(total_steps) * dt
    
    # Position: Lemniscate of Bernoulli
    cos_t = jnp.cos(t)
    sin_t = jnp.sin(t)
    denom = 1 + cos_t ** 2
    
    x = scale * sin_t / denom
    y = scale * sin_t * cos_t / denom
    
    # Velocity: analytical derivatives
    # dx/dt = d/dt[scale * sin(t) / (1 + cos²(t))]
    # Using quotient rule and chain rule
    
    # Numerator: cos(t) * (1 + cos²(t))
    # Denominator derivative: -2 * cos(t) * sin(t)
    numerator_x = cos_t * (1 + cos_t ** 2) - sin_t * (-2 * cos_t * (-sin_t))
    vx = scale * numerator_x / (denom ** 2)
    
    # dy/dt = d/dt[scale * sin(t) * cos(t) / (1 + cos²(t))]
    numerator_y = (cos_t ** 2 - sin_t ** 2) * (1 + cos_t ** 2) - sin_t * cos_t * (-2 * cos_t * (-sin_t))
    vy = scale * numerator_y / (denom ** 2)
    
    # Stack into trajectory: [x, y, vx, vy]
    trajectory = jnp.stack([x, y, vx, vy], axis=-1)
    
    return trajectory
