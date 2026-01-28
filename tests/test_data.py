"""Tests for data generators."""

import pytest
import jax
import jax.numpy as jnp

from chlu.data.figure8 import generate_figure8
from chlu.data.sine_waves import generate_sine_waves, add_noise


def test_figure8_shape():
    """Figure-8 generator produces correct shape."""
    data = generate_figure8(jax.random.PRNGKey(0), steps=1000)
    
    assert data.shape == (1000, 4)  # (x, y, vx, vy)
    assert jnp.all(jnp.isfinite(data))


def test_figure8_periodic():
    """Figure-8 should be approximately periodic."""
    data = generate_figure8(jax.random.PRNGKey(0), steps=1000, dt=0.01)
    
    # The lemniscate has period 2π, so at t≈2π should be back to start
    # With dt=0.01, one period is ~628 steps
    period_steps = int(2 * jnp.pi / 0.01)
    
    if period_steps < len(data):
        start = data[0]
        end = data[min(period_steps, len(data) - 1)]
        
        # Should be close (allowing some numerical error in derivatives)
        distance = jnp.sqrt(jnp.sum((start[:2] - end[:2]) ** 2))
        assert distance < 1.0, f"Periodicity check failed: distance={distance}"


def test_sine_waves_shape():
    """Sine wave generator produces correct shape."""
    data = generate_sine_waves(jax.random.PRNGKey(0), n_waves=10, steps=100)
    
    assert data.shape == (10, 100, 2)  # (batch, time, [x, dx/dt])
    assert jnp.all(jnp.isfinite(data))


def test_add_noise():
    """Noise addition works correctly."""
    data = jnp.ones((10, 10))
    noisy = add_noise(data, jax.random.PRNGKey(0), sigma=0.1)
    
    assert noisy.shape == data.shape
    assert not jnp.allclose(noisy, data)  # Should be different
    assert jnp.all(jnp.isfinite(noisy))
