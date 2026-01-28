"""Sine wave data generator for noise robustness experiments."""

import jax
import jax.numpy as jnp


def generate_sine_waves(
    key: jax.random.PRNGKey,
    n_waves: int,
    steps: int,
    dt: float = 0.01,
    freq_range: tuple = (0.5, 2.0),
    amp_range: tuple = (0.5, 1.5),
) -> jnp.ndarray:
    """
    Generate clean sine wave trajectories with random frequencies and amplitudes.
    
    Each wave is represented as [x, dx/dt] where:
        x(t) = A * sin(2π * f * t)
        dx/dt = A * 2π * f * cos(2π * f * t)
    
    Args:
        key: JAX random key
        n_waves: Number of sine waves to generate
        steps: Number of time steps per wave
        dt: Time step size
        freq_range: (min_freq, max_freq) in Hz
        amp_range: (min_amp, max_amp)
    
    Returns:
        Sine waves of shape (n_waves, steps, 2): [x, dx/dt]
    """
    k1, k2 = jax.random.split(key, 2)
    
    # Random frequencies and amplitudes for each wave
    freqs = jax.random.uniform(k1, (n_waves,), minval=freq_range[0], maxval=freq_range[1])
    amps = jax.random.uniform(k2, (n_waves,), minval=amp_range[0], maxval=amp_range[1])
    
    # Time array
    t = jnp.arange(steps) * dt  # (steps,)
    
    # Generate waves
    def generate_single_wave(freq, amp):
        """Generate single sine wave."""
        omega = 2 * jnp.pi * freq
        x = amp * jnp.sin(omega * t)
        dx_dt = amp * omega * jnp.cos(omega * t)
        return jnp.stack([x, dx_dt], axis=-1)  # (steps, 2)
    
    # Vectorize over all waves
    waves = jax.vmap(generate_single_wave)(freqs, amps)  # (n_waves, steps, 2)
    
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
