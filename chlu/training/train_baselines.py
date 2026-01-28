"""Training functions for baseline models (Neural ODE and LSTM)."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm

from chlu.training.losses import mse_loss


def train_neural_ode(
    model,
    data: jnp.ndarray,
    key: jax.random.PRNGKey,
    epochs: int = 1000,
    lr: float = 1e-3,
    dt: float = 0.01,
):
    """
    Train Neural ODE with standard supervised learning.
    
    Args:
        model: NeuralODE model
        data: Training data (n_trajectories, T, dim) or (T, dim)
        key: JAX random key
        epochs: Number of epochs
        lr: Learning rate
        dt: Time step
    
    Returns:
        (trained_model, losses): Trained model and loss history
    """
    # Handle data shape
    if data.ndim == 2:
        data = data[None, :, :]
    
    n_trajectories, T, dim = data.shape
    
    # Initialize optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    losses = []
    
    @eqx.filter_jit
    def train_step(model, opt_state, trajectory):
        """Single training step."""
        def loss_fn(model):
            # Predict trajectory from initial state
            z0 = trajectory[0]
            t_span = (0.0, T * dt)
            pred_trajectory = model(z0, t_span, dt)
            
            # MSE loss
            return mse_loss(pred_trajectory, trajectory)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Neural ODE"):
        # Sample random trajectory
        traj_idx = jax.random.randint(key, (), 0, n_trajectories)
        key = jax.random.split(key)[0]
        
        trajectory = data[traj_idx]
        model, opt_state, loss = train_step(model, opt_state, trajectory)
        
        losses.append(float(loss))
    
    return model, jnp.array(losses)


def train_lstm(
    model,
    data: jnp.ndarray,
    key: jax.random.PRNGKey,
    epochs: int = 1000,
    lr: float = 1e-3,
):
    """
    Train LSTM for next-step prediction.
    
    Args:
        model: LSTMPredictor model
        data: Training data (n_sequences, T, dim) or (T, dim)
        key: JAX random key
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        (trained_model, losses): Trained model and loss history
    """
    # Handle data shape
    if data.ndim == 2:
        data = data[None, :, :]
    
    n_sequences, T, dim = data.shape
    
    # Initialize optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    losses = []
    
    @eqx.filter_jit
    def train_step(model, opt_state, sequence):
        """Single training step."""
        def loss_fn(model):
            # Input: x[:-1], Target: x[1:]
            inputs = sequence[:-1]
            targets = sequence[1:]
            
            # Predict next steps
            predictions = model(inputs)
            
            # MSE loss
            return mse_loss(predictions, targets)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training LSTM"):
        # Sample random sequence
        seq_idx = jax.random.randint(key, (), 0, n_sequences)
        key = jax.random.split(key)[0]
        
        sequence = data[seq_idx]
        model, opt_state, loss = train_step(model, opt_state, sequence)
        
        losses.append(float(loss))
    
    return model, jnp.array(losses)
