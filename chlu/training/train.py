"""PCD (Wake-Sleep) training for CHLU."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm

from chlu.training.replay_buffer import ReplayBuffer
from chlu.training.losses import mse_loss, energy_loss
from chlu.core.regularization import compute_lyapunov_loss


def train_chlu(
    model,
    data: jnp.ndarray,
    key: jax.random.PRNGKey,
    epochs: int = 1000,
    lr: float = 1e-3,
    lyapunov_lambda: float = 0.01,
    sleep_steps: int = 10,
    buffer_capacity: int = 1024,
    batch_size: int = 32,
    dt: float = 0.01,
):
    """
    Train CHLU using Persistent Contrastive Divergence (Wake-Sleep).
    
    Wake Phase: Supervised learning on data (MSE + Lyapunov regularization)
    Sleep Phase: Unsupervised energy minimization on replay buffer
    
    Args:
        model: CHLU model
        data: Training data of shape (n_trajectories, T, 2*dim) or (T, 2*dim)
        key: JAX random key
        epochs: Number of training epochs
        lr: Learning rate
        lyapunov_lambda: Weight for Lyapunov regularization
        sleep_steps: Number of dynamics steps in sleep phase
        buffer_capacity: Replay buffer capacity
        batch_size: Batch size for sleep phase
        dt: Time step for dynamics
    
    Returns:
        (trained_model, losses): Trained model and loss history
    """
    # Handle data shape
    if data.ndim == 2:
        data = data[None, :, :]  # Add batch dimension
    
    n_trajectories, T, state_dim = data.shape
    dim = state_dim // 2
    
    # Initialize optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Initialize replay buffer
    k1, k2 = jax.random.split(key)
    buffer = ReplayBuffer(capacity=buffer_capacity, dim=dim)
    buffer.initialize_random(k1, scale=1.0)
    
    losses = []
    
    @eqx.filter_jit
    def wake_step(model, opt_state, trajectory, key):
        """Wake phase: supervised learning."""
        q_true = trajectory[:, :dim]
        p_true = trajectory[:, dim:]
        
        def loss_fn(model):
            # Run CHLU dynamics from initial state
            q0, p0 = q_true[0], p_true[0]
            pred_trajectory = model(q0, p0, steps=len(trajectory), dt=dt)
            
            # MSE loss
            mse = mse_loss(pred_trajectory, trajectory)
            
            # Lyapunov regularization
            lyap_loss = compute_lyapunov_loss(
                lambda state: model.step(state, dt),
                pred_trajectory,
                n_samples=min(10, len(trajectory) // 2)
            )
            
            return mse + lyapunov_lambda * lyap_loss
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    @eqx.filter_jit
    def sleep_step(model, opt_state, buffer, key):
        """Sleep phase: energy minimization."""
        # Sample from buffer
        q_batch, p_batch, indices = buffer.sample(key, batch_size)
        
        def loss_fn(model):
            # Evolve states for k steps
            def evolve_single(q, p):
                state = (q, p)
                for _ in range(sleep_steps):
                    state = model.step(state, dt)
                return state
            
            q_evolved, p_evolved = jax.vmap(evolve_single)(q_batch, p_batch)
            
            # Minimize final energy
            return energy_loss(model, q_evolved, p_evolved)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        # Update buffer with evolved states (not implemented in-place, return info)
        return model, opt_state, loss
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training CHLU"):
        k2, k3 = jax.random.split(k2)
        
        # Wake phase: train on random trajectory
        traj_idx = jax.random.randint(k2, (), 0, n_trajectories)
        trajectory = data[traj_idx]
        
        model, opt_state, wake_loss = wake_step(model, opt_state, trajectory, k3)
        
        # Sleep phase (every few epochs to save compute)
        if epoch % 5 == 0:
            k3, k4 = jax.random.split(k3)
            model, opt_state, sleep_loss = sleep_step(model, opt_state, buffer, k4)
        
        losses.append(float(wake_loss))
    
    return model, jnp.array(losses)
