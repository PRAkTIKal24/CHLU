"""PCD (Wake-Sleep) training for CHLU."""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from chlu.config import CHLUConfig, get_default_config
from chlu.core.regularization import compute_lyapunov_loss
from chlu.training.losses import energy_loss, mse_loss
from chlu.training.replay_buffer import ReplayBuffer


def train_chlu(
    model,
    data: jnp.ndarray,
    key: jax.random.PRNGKey,
    config: Optional[CHLUConfig] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    lyapunov_lambda: Optional[float] = None,
    sleep_steps: Optional[int] = None,
    buffer_capacity: Optional[int] = None,
    batch_size: Optional[int] = None,
    dt: Optional[float] = None,
):
    """
    Train CHLU using Persistent Contrastive Divergence (Wake-Sleep).

    Wake Phase: Supervised learning on data (MSE + Lyapunov regularization)
    Sleep Phase: Unsupervised energy minimization on replay buffer

    Args:
        model: CHLU model
        data: Training data of shape (n_trajectories, T, 2*dim) or (T, 2*dim)
        key: JAX random key
        config: CHLUConfig object (if None, uses defaults)
        epochs: Number of training epochs (overrides config)
        lr: Learning rate (overrides config)
        lyapunov_lambda: Weight for Lyapunov regularization (overrides config)
        sleep_steps: Number of dynamics steps in sleep phase (overrides config)
        buffer_capacity: Replay buffer capacity (overrides config)
        batch_size: Batch size for sleep phase (overrides config)
        dt: Time step for dynamics (overrides config)

    Returns:
        (trained_model, losses): Trained model and loss history
    """
    # Load config with overrides
    if config is None:
        config = get_default_config()

    # Apply overrides
    if epochs is None:
        epochs = config.training.epochs
    if lr is None:
        lr = config.training.learning_rate
    if lyapunov_lambda is None:
        lyapunov_lambda = config.training.lyapunov_lambda
    if sleep_steps is None:
        sleep_steps = config.training.sleep_steps
    if buffer_capacity is None:
        buffer_capacity = config.training.buffer_capacity
    if batch_size is None:
        batch_size = config.training.batch_size
    if dt is None:
        dt = config.training.dt

    sleep_frequency = config.training.sleep_frequency
    sleep_friction = config.training.sleep_friction

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
    def wake_step(model, opt_state, trajectory, key, epoch, epochs):
        """Wake phase: supervised learning."""
        q_true = trajectory[:, :dim]
        p_true = trajectory[:, dim:]

        def loss_fn(model):
            # Run CHLU dynamics from initial state
            q0, p0 = q_true[0], p_true[0]
            pred_trajectory = model(q0, p0, steps=len(trajectory), dt=dt)

            # MSE loss, weighted by clamp_strength - start high
            clamp_strength = 100

            # clamp_strength annealing
            schedule = epoch / epochs
            clamp_strength = clamp_strength * (1 - schedule) + 0.1
            mse = clamp_strength * mse_loss(pred_trajectory, trajectory)

            # Lyapunov regularization
            lyap_loss = compute_lyapunov_loss(
                lambda state: model.step(state, dt),
                pred_trajectory,
                n_samples=min(10, len(trajectory) // 2),
            )

            return mse + lyapunov_lambda * lyap_loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    @eqx.filter_jit
    def sleep_step(model, opt_state, buffer, key, sleep_friction=0.0):
        """Sleep phase: energy minimization."""
        # Sample from buffer
        q_batch, p_batch, indices = buffer.sample(key, batch_size)

        def loss_fn(model):
            # Evolve states for k steps
            def evolve_single(q, p):
                state = (q, p)
                for _ in range(sleep_steps):
                    state = model.step(state, dt, gamma=sleep_friction)
                return state

            q_evolved, p_evolved = jax.vmap(evolve_single)(q_batch, p_batch)

            # Negative sign because we want to *maximize* sleep energy
            sleep_energy = -energy_loss(model, q_evolved, p_evolved)

            return sleep_energy

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

        # Convert epoch values to jax arrays for clamp_strength annealing
        epoch_jax = jnp.array(epoch)
        epochs_jax = jnp.array(epochs)

        model, opt_state, wake_loss = wake_step(
            model, opt_state, trajectory, k3, epoch_jax, epochs_jax
        )

        # Sleep phase (every few epochs to save compute)
        if epoch % sleep_frequency == 0:
            k3, k4 = jax.random.split(k3)
            model, opt_state, sleep_loss = sleep_step(
                model,
                opt_state,
                buffer,
                k4,
                sleep_friction,
            )

        losses.append(float(wake_loss))

    return model, jnp.array(losses)
