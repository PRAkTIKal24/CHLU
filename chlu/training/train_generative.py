"""Pure Energy-Based Training for Generative Models (Experiment C).

This module implements Persistent Contrastive Divergence (PCD) for training
CHLU as an Energy-Based Model (EBM). Unlike the dynamics training in train.py,
this uses NO MSE loss - only energy sculpting via contrastive divergence.

Key differences from train.py:
- Wake Phase: Minimize H(Real) - push energy of real data down
- Sleep Phase: Maximize H(Dream) - pull energy of random/evolved states up
- Persistent Buffer: evolved states are stored back for cumulative exploration
- Random Re-initialization: 5% of chains reset to prevent mode collapse
- Pixel Clamping: hard clip to [-1, 1] to keep outputs valid
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from chlu.config import CHLUConfig, get_default_config
from chlu.training.replay_buffer import ReplayBuffer


def train_generative(
    model,
    data: jnp.ndarray,
    key: jax.random.PRNGKey,
    config: Optional[CHLUConfig] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    dt: Optional[float] = None,
    buffer_capacity: Optional[int] = None,
    reinit_prob: Optional[float] = None,
    k_steps: Optional[int] = None,
    clamp_outputs: Optional[bool] = None,
    energy_weight: Optional[float] = None,
    sleep_friction: Optional[float] = None,
    sleep_temperature: Optional[float] = None,
    input_noise_sigma: Optional[float] = None,
):
    """
    Train CHLU as an Energy-Based Model using Persistent Contrastive Divergence.

    This is a pure generative training method with NO MSE component.
    It learns to shape the energy landscape by:
    - Pushing down energy of real data (Wake)
    - Pulling up energy of hallucinated states (Sleep)

    Args:
        model: CHLU model
        data: Training data of shape (n_samples, dim) where dim is state dimension
              For MNIST: (n_samples, 784) or (n_samples, pca_dim)
              Data should be normalized to [-1, 1]
        key: JAX random key
        config: CHLUConfig object (if None, uses defaults)
        epochs: Number of training epochs (overrides config)
        lr: Learning rate (overrides config)
        batch_size: Batch size (overrides config)
        dt: Time step for dynamics (overrides config)
        buffer_capacity: Replay buffer capacity (overrides config)
        reinit_prob: Probability of resetting chains to noise (overrides config)
        k_steps: Number of negative phase evolution steps (overrides config)
        clamp_outputs: Enable pixel clamping to [-1, 1] (overrides config)
        energy_weight: Weight for contrastive energy loss (overrides config)
        sleep_friction: Friction during negative phase (overrides config)
        sleep_temperature: Temperature for Langevin noise during sleep phase (overrides config)
        input_noise_sigma: Gaussian noise std for real data (denoising EBM, overrides config)

    Returns:
        (trained_model, losses): Trained model and loss history dict
    """
    # Load config with overrides
    if config is None:
        config = get_default_config()

    # Apply overrides from training config
    if epochs is None:
        epochs = config.training.epochs
    if lr is None:
        lr = config.training.learning_rate
    if batch_size is None:
        batch_size = config.training.batch_size
    if dt is None:
        dt = config.training.dt
    if buffer_capacity is None:
        buffer_capacity = config.training.buffer_capacity
    if reinit_prob is None:
        reinit_prob = config.training.reinit_prob
    if k_steps is None:
        k_steps = config.training.k_steps
    if clamp_outputs is None:
        clamp_outputs = config.training.clamp_outputs
    if energy_weight is None:
        energy_weight = config.training.energy_weight
    if sleep_friction is None:
        sleep_friction = config.experiment_c.friction
    if sleep_temperature is None:
        sleep_temperature = config.training.sleep_temperature
    if input_noise_sigma is None:
        input_noise_sigma = config.training.input_noise_sigma

    # Handle data shape
    if data.ndim == 1:
        data = data[None, :]  # Add batch dimension

    n_samples, dim = data.shape

    # Initialize optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Initialize replay buffer
    k1, k2 = jax.random.split(key)
    buffer = ReplayBuffer(capacity=buffer_capacity, dim=dim)
    buffer.initialize_random(k1, scale=1.0)

    # Loss history
    losses = {"wake": [], "sleep": [], "total": []}

    @eqx.filter_jit
    def train_step(model, opt_state, x_real, buffer_q, buffer_p, key, noise_sigma):
        """
        Single train step combining wake and sleep phases.

        Args:
            model: CHLU model
            opt_state: Optimizer state
            x_real: Real data batch (batch_size, dim)
            buffer_q: Current buffer q states (batch_size, dim)
            buffer_p: Current buffer p states (batch_size, dim)
            key: JAX random key
            noise_sigma: Gaussian noise std for real data (denoising EBM)

        Returns:
            (loss_wake, loss_sleep, model, opt_state, q_fake, p_fake)
        """
        # ==== INPUT NOISE INJECTION (Denoising EBM) ====
        # Add small Gaussian noise to real data to widen the energy basin.
        # This prevents "Dirac delta" potentials (infinitely narrow energy wells)
        # and creates gentle slopes that guide sampling towards data modes.
        key, subkey = jax.random.split(key)
        x_real_noisy = x_real + jax.random.normal(subkey, x_real.shape) * noise_sigma
        x_real_noisy = jnp.clip(x_real_noisy, -1.0, 1.0)  # Keep in valid range

        # ==== NEGATIVE PHASE (Sleep): Sample and Evolve ====
        # 1. Start from buffer states
        q_start = buffer_q
        p_start = buffer_p

        # 2. Random re-initialization (prevents mode collapse)
        # Reset 5% of chains to random noise
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(subkey, reinit_prob, (x_real.shape[0],))

        # Generate fresh noise
        key, subkey = jax.random.split(key)
        q_noise = jax.random.uniform(subkey, x_real.shape, minval=-1, maxval=1)
        key, subkey = jax.random.split(key)
        p_noise = jax.random.normal(subkey, x_real.shape) * 0.1

        # Apply mask: keep buffer where mask=0, use noise where mask=1
        q_start = jnp.where(mask[:, None], q_noise, q_start)
        p_start = jnp.where(mask[:, None], p_noise, p_start)

        # 3. Run physics for k steps
        if sleep_temperature > 0.0:
            # Stochastic evolution with Langevin noise
            # Split key for each particle in batch
            key, *subkeys = jax.random.split(key, x_real.shape[0] + 1)
            subkeys = jnp.array(subkeys)

            def evolve_single(q, p, particle_key):
                """Evolve a single (q, p) state for k steps with noise."""

                def step_fn(carry, _):
                    state, key_state = carry
                    q_s, p_s = state
                    q_next, p_next, new_key = model.stochastic_step(
                        (q_s, p_s),
                        dt=dt,
                        gamma=sleep_friction,
                        temperature=sleep_temperature,
                        key=key_state,
                    )
                    return ((q_next, p_next), new_key), None

                state = (q, p)
                (final_state, _), _ = jax.lax.scan(
                    step_fn, (state, particle_key), None, length=k_steps
                )
                return final_state

            q_fake, p_fake = jax.vmap(evolve_single)(q_start, p_start, subkeys)
        else:
            # Deterministic evolution
            def evolve_single(q, p):
                """Evolve a single (q, p) state for k steps."""

                def step_fn(state, _):
                    return model.step(state, dt=dt, gamma=sleep_friction), None

                state = (q, p)
                final_state, _ = jax.lax.scan(step_fn, state, None, length=k_steps)
                return final_state

            q_fake, p_fake = jax.vmap(evolve_single)(q_start, p_start)

        # 4. Pixel clamping (critical for bounded spaces like images)
        if clamp_outputs:
            q_fake = jnp.clip(q_fake, -1.0, 1.0)

        # ==== CONTRASTIVE DIVERGENCE LOSS ====
        def loss_fn(model):
            # Wake: Minimize energy of NOISY real data
            # "Push this noisy '7' down the hill" - widens the basin
            p_real = jnp.zeros_like(x_real_noisy)  # Zero momentum for static images
            E_real = jax.vmap(model.H)(x_real_noisy, p_real)
            loss_wake = jnp.mean(E_real)

            # Sleep: Maximize energy of fake/dream data
            # "Pull this random noise up the hill"
            # Negative sign because we maximize
            E_fake = jax.vmap(model.H)(q_fake, p_fake)
            loss_sleep = -jnp.mean(E_fake)

            # Energy Regularization: Keep energies in reasonable range [-10, 10]
            # Without this, energy explodes to -8000 and temperature/noise becomes useless.
            # Penalizes the model for outputting massive energy magnitudes.
            loss_reg = 0.01 * (jnp.mean(E_real**2) + jnp.mean(E_fake**2))

            # Total contrastive loss with regularization
            total_loss = loss_wake + energy_weight * loss_sleep + loss_reg

            return total_loss, (loss_wake, loss_sleep)

        (loss, (loss_wake, loss_sleep)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)

        # Update model
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # Return updated buffer states for persistence
        return loss_wake, loss_sleep, model, opt_state, q_fake, p_fake, key

    # Training loop
    print(f"Training generative model for {epochs} epochs...")
    print(f"Buffer capacity: {buffer_capacity}, Batch size: {batch_size}")
    print(
        f"Re-init prob: {reinit_prob}, K-steps: {k_steps}, Friction: {sleep_friction}, Temperature: {sleep_temperature}"
    )
    print(f"Input noise sigma: {input_noise_sigma} (denoising EBM)")

    for _epoch in tqdm(range(epochs), desc="Training (Generative)"):
        k2, subkey = jax.random.split(k2)

        # Sample random batch of real data
        batch_indices = jax.random.randint(
            subkey, shape=(batch_size,), minval=0, maxval=n_samples
        )
        x_batch = data[batch_indices]

        # Sample from buffer (same indices for simplicity, or random sample)
        # For simplicity with buffer_capacity == batch_size, we just use the buffer directly
        # For larger buffers, you'd randomly sample and update specific indices
        k2, subkey = jax.random.split(k2)
        buffer_q, buffer_p, buffer_indices = buffer.sample(subkey, batch_size)

        # Train step
        loss_wake, loss_sleep, model, opt_state, new_q, new_p, k2 = train_step(
            model, opt_state, x_batch, buffer_q, buffer_p, k2, input_noise_sigma
        )

        # CRITICAL: Update persistent buffer with evolved states
        buffer.update((new_q, new_p), buffer_indices)

        # Log losses
        losses["wake"].append(float(loss_wake))
        losses["sleep"].append(float(loss_sleep))
        losses["total"].append(float(loss_wake + energy_weight * loss_sleep))

    # Compute target energy (floor) from real data
    # This is useful for energy-based generation/sampling later
    p_zeros = jnp.zeros_like(data)
    all_energies = jax.vmap(model.H)(data, p_zeros)
    target_energy = float(jnp.percentile(all_energies, 1.0))

    print("\nTraining complete!")
    print(f"Mean real data energy: {jnp.mean(all_energies):.4f}")
    print(f"Target energy (1st percentile): {target_energy:.4f}")

    # Convert losses to arrays
    losses_array = {k: jnp.array(v) for k, v in losses.items()}

    return model, losses_array, target_energy
