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


def sample_window(key: jax.random.PRNGKey, data: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """
    Sample a random window from trajectory data.
    
    Args:
        key: JAX random key
        data: Trajectory data of shape (T, dim)
        window_size: Size of the window to sample
        
    Returns:
        Window of shape (window_size, dim)
    """
    max_start = len(data) - window_size
    idx = jax.random.randint(key, (), 0, max_start)
    return data[idx : idx + window_size]


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
    window_size: Optional[int] = None,
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
        window_size: Window size for sub-sequence sampling (overrides config)

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
    if window_size is None:
        # Use experiment A config if available, otherwise use full trajectory
        if hasattr(config, 'experiment_a') and hasattr(config.experiment_a, 'window_size'):
            window_size = config.experiment_a.window_size
        else:
            window_size = None  # Will be set below

    sleep_frequency = config.training.sleep_frequency
    sleep_friction = config.training.sleep_friction
    clamp_strength = jnp.array(config.training.clamp_strength)
    clamp_ramp = config.training.clamp_ramp

    # Handle data shape
    if data.ndim == 2:
        data = data[None, :, :]  # Add batch dimension

    n_trajectories, T, state_dim = data.shape
    dim = state_dim // 2
    
    # Set window_size if not provided
    if window_size is None:
        window_size = T  # Use full trajectory if no window specified

    # Initialize optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Initialize replay buffer
    k1, k2 = jax.random.split(key)
    buffer = ReplayBuffer(capacity=buffer_capacity, dim=dim)
    buffer.initialize_random(k1, scale=1.0)

    losses = []

    @eqx.filter_jit
    def wake_step(
        model, opt_state, trajectory, key, epoch, epochs_ramp, clamp_strength
    ):
        """Wake phase: supervised learning on trajectory window."""
        q_true = trajectory[:, :dim]
        p_true = trajectory[:, dim:]

        # Compute clamp_strength annealing outside loss_fn to avoid recomputation
        schedule = epoch / epochs_ramp
        annealed_clamp = clamp_strength * (1 - schedule) + 1.0
        effective_clamp = jnp.where(epoch < epochs_ramp, annealed_clamp, 1.0)

        def loss_fn(model):
            # Run CHLU dynamics from initial state for window_size steps
            q0, p0 = q_true[0], p_true[0]
            pred_trajectory = model(q0, p0, steps=len(trajectory), dt=dt)

            # Use precomputed clamp strength
            mse = effective_clamp * mse_loss(pred_trajectory, trajectory)

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
            # Evolve states for k steps using scan to avoid slow compilation
            def evolve_single(q, p):
                def step_fn(state, _):
                    return model.step(state, dt, gamma=sleep_friction), None
                
                state = (q, p)
                final_state, _ = jax.lax.scan(step_fn, state, None, length=sleep_steps)
                return final_state

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

        # Sample random trajectory
        traj_idx = jax.random.randint(k2, (), 0, n_trajectories)
        full_trajectory = data[traj_idx]
        
        # Sample random window from the trajectory
        k3, k4 = jax.random.split(k3)
        trajectory = sample_window(k4, full_trajectory, window_size)

        # Convert epoch values to jax arrays for clamp_strength annealing
        epoch_jax = jnp.array(epoch)
        epochs_ramp_jax = jnp.array(clamp_ramp * epochs)

        k4, k5 = jax.random.split(k4)
        model, opt_state, wake_loss = wake_step(
            model,
            opt_state,
            trajectory,
            k5,
            epoch_jax,
            epochs_ramp_jax,
            clamp_strength,
        )

        # Sleep phase (every few epochs to save compute)
        if epoch % sleep_frequency == 0:
            k5, k6 = jax.random.split(k5)
            model, opt_state, sleep_loss = sleep_step(
                model,
                opt_state,
                buffer,
                k6,
                sleep_friction,
            )

        losses.append(float(wake_loss))

    return model, jnp.array(losses)
