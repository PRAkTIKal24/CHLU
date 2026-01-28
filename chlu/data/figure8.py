"""Figure-8 (Lemniscate of Bernoulli) trajectory generator."""

import jax
import jax.numpy as jnp


def generate_figure8(
    key: jax.random.PRNGKey,
    steps: int,
    dt: float = 0.01,
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
        steps: Number of time steps
        dt: Time step size
        scale: Scaling factor for the curve
    
    Returns:
        Trajectory of shape (steps, 4): [x, y, vx, vy] ≡ [q, p]
    """
    # Time parameter covering one full period
    # One period of the lemniscate is 2π
    t = jnp.linspace(0, 2 * jnp.pi, steps)
    
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
