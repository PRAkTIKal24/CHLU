"""Core functionality tests for CHLU."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from chlu.core.chlu_unit import CHLU
from chlu.core.baselines import NeuralODE, LSTMPredictor
from chlu.utils.metrics import count_params


def test_chlu_initialization():
    """Test CHLU can be initialized."""
    model = CHLU(dim=2, hidden=32, key=jax.random.PRNGKey(0))
    assert model.dim == 2
    assert model.rest_mass == 1.0


def test_symplectic_determinant():
    """Verify det(Jacobian) ≈ 1.0 for CHLU step (symplecticity)."""
    model = CHLU(dim=2, hidden=32, key=jax.random.PRNGKey(0))
    state = (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    
    def step_wrapper(flat_state):
        q, p = flat_state[:2], flat_state[2:]
        q_new, p_new = model.step((q, p), dt=0.01)
        return jnp.concatenate([q_new, p_new])
    
    J = jax.jacfwd(step_wrapper)(jnp.concatenate([state[0], state[1]]))
    det = jnp.linalg.det(J)
    
    # Symplectic integrator should preserve phase space volume
    assert jnp.isclose(det, 1.0, atol=1e-4), f"Symplecticity violated: det={det}"


def test_energy_conservation():
    """Energy should be approximately conserved over trajectory."""
    model = CHLU(dim=2, hidden=32, key=jax.random.PRNGKey(0))
    q0, p0 = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    
    trajectory = model(q0, p0, steps=1000, dt=0.01)
    
    # Compute energies at start and end
    energies = [model.H(trajectory[i, :2], trajectory[i, 2:]) 
                for i in [0, -1]]
    
    energy_drift = jnp.abs(energies[-1] - energies[0]) / energies[0]
    
    # Energy drift should be small (<5% for 1000 steps)
    assert energy_drift < 0.05, f"Energy drift {energy_drift:.2%} exceeds 5%"


def test_gradients_flow():
    """Verify gradients flow through CHLU."""
    model = CHLU(dim=2, hidden=32, key=jax.random.PRNGKey(0))
    q0, p0 = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    
    def loss_fn(model):
        traj = model(q0, p0, steps=10, dt=0.01)
        return jnp.sum(traj ** 2)
    
    grads = eqx.filter_grad(loss_fn)(model)
    
    # Check that at least some gradients are non-zero
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    has_nonzero = any(jnp.any(leaf != 0) for leaf in leaves if leaf.size > 0)
    
    assert has_nonzero, "No gradients computed"


def test_mass_positive_definite():
    """Mass matrix M must always be positive."""
    model = CHLU(dim=4, hidden=32, key=jax.random.PRNGKey(0))
    M = jax.nn.softplus(model.log_mass)
    assert jnp.all(M > 0), "Mass matrix has non-positive elements"


def test_lstm_forward():
    """LSTM produces valid predictions."""
    model = LSTMPredictor(dim=4, hidden_size=32, key=jax.random.PRNGKey(0))
    x_seq = jnp.ones((10, 4))
    
    preds = model(x_seq)
    
    assert preds.shape == (10, 4)
    assert jnp.all(jnp.isfinite(preds))


def test_lstm_generate():
    """LSTM generate() produces trajectory without errors."""
    model = LSTMPredictor(dim=4, hidden_size=32, key=jax.random.PRNGKey(0))
    x0 = jnp.array([1.0, 0.0, 0.0, 1.0])
    
    traj = model.generate(x0, steps=100)
    
    assert traj.shape == (100, 4)
    assert jnp.all(jnp.isfinite(traj))


def test_neural_ode_forward():
    """NeuralODE produces valid trajectories."""
    model = NeuralODE(dim=4, hidden=32, key=jax.random.PRNGKey(0))
    z0 = jnp.array([1.0, 0.0, 0.0, 1.0])
    
    traj = model(z0, t_span=(0, 1), dt=0.01)
    
    assert traj.shape[0] > 0
    assert jnp.all(jnp.isfinite(traj))


def test_parameter_counts_comparable():
    """All 3 models should have similar parameter counts."""
    chlu = CHLU(dim=2, hidden=32, key=jax.random.PRNGKey(0))
    node = NeuralODE(dim=4, hidden=32, key=jax.random.PRNGKey(1))
    lstm = LSTMPredictor(dim=4, hidden_size=32, key=jax.random.PRNGKey(2))
    
    counts = {
        "CHLU": count_params(chlu),
        "NODE": count_params(node),
        "LSTM": count_params(lstm),
    }
    
    print(f"\nParameter counts: {counts}")
    
    max_c, min_c = max(counts.values()), min(counts.values())
    ratio = max_c / min_c
    
    # LSTM has more params due to gates, allow up to 5x difference
    assert ratio < 5.0, f"Param imbalance: {counts}"
