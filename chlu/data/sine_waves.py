"""Sine wave data generator for noise robustness experiments."""

import jax
import jax.numpy as jnp


def generate_sine_waves(
    key: jax.random.PRNGKey,
    n_waves: int,
    steps: int,
    dt: float = 0.01,
    freq: float = 1.0,
    amp: float = 1.0,
) -> jnp.ndarray:
    """
    Generate clean sine wave trajectories with random phases.
    
    All waves have the SAME frequency and amplitude, only phase differs.
    This ensures a single point (q, p) uniquely determines the trajectory,
    making single-timestep inference physically valid.
    
    Each wave is represented as [x, dx/dt] where:
        x(t) = A * sin(2π * f * t + φ)
        dx/dt = A * 2π * f * cos(2π * f * t + φ)
    
    Args:
        key: JAX random key
        n_waves: Number of sine waves to generate
        steps: Number of time steps per wave
        dt: Time step size
        freq: Frequency in Hz (fixed for all waves)
        amp: Amplitude (fixed for all waves)
    
    Returns:
        Sine waves of shape (n_waves, steps, 2): [x, dx/dt]
    """
    # Random phases for each wave
    phases = jax.random.uniform(key, (n_waves,), minval=0.0, maxval=2.0 * jnp.pi)
    
    # Time array
    t = jnp.arange(steps) * dt  # (steps,)
    
    # Generate waves
    def generate_single_wave(phase):
        """Generate single sine wave with given phase."""
        omega = 2 * jnp.pi * freq
        x = amp * jnp.sin(omega * t + phase)
        dx_dt = amp * omega * jnp.cos(omega * t + phase)
        return jnp.stack([x, dx_dt], axis=-1)  # (steps, 2)
    
    # Vectorize over all waves
    waves = jax.vmap(generate_single_wave)(phases)  # (n_waves, steps, 2)
    
    return waves


def add_noise(
    data: jnp.ndarray, 
    key: jax.random.PRNGKey, 
    sigma: float
) -> jnp.ndarray:
    """
    Add Gaussian noise N(0, sigma) to data.
    
    Args:
        data: Clean data of any shape
        key: JAX random key
        sigma: Noise standard deviation
    
    Returns:
        Noisy data with same shape as input
    """
    noise = sigma * jax.random.normal(key, data.shape)
    return data + noise
