"""Experiment C: Generative "Dreaming" on MNIST.

Validates PCD training and generative capability of CHLU.
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.data.mnist import load_mnist_pca
from chlu.training.train import train_chlu
from chlu.utils.checkpoints import save_checkpoint
from chlu.utils.plotting import plot_dreaming_grid


def run_experiment_c(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    pca_dim: Optional[int] = None,
    train_epochs: Optional[int] = None,
    n_samples: Optional[int] = None,
    dream_steps: Optional[int] = None,
    friction: Optional[float] = None,
    dt: Optional[float] = None,
):
    """
    Experiment C: Generative "Dreaming" (MNIST).

    Protocol:
        1. Load MNIST (PCA → 32 dims)
        2. Train CHLU with PCD (wake-sleep)
        3. Initialize random noise (q, p)
        4. Run dissipative dynamics with friction γ=0.01
        5. Show evolution: Noise → Hazy → Crisp Digit

    Expected outcome:
        Grid of images showing progression from random noise
        to recognizable digit shapes.

    Args:
        config: CHLUConfig object (if None, uses defaults)
        save_dir: Directory to save results (overrides config)
        models_dir: Directory to save trained models (defaults to save_dir/models)
        seed: Random seed (overrides config)
        pca_dim: PCA dimensionality (overrides config)
        train_epochs: Training epochs (overrides config)
        n_samples: Number of MNIST samples to use (overrides config)
        dream_steps: Steps to evolve during dreaming (overrides config)
        friction: Friction coefficient for energy dissipation (overrides config)
        dt: Time step (overrides config)
    """
    # Load config with overrides
    if config is None:
        config = get_default_config()

    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed
    if pca_dim is not None:
        config.experiment_c.pca_dim = pca_dim
    if train_epochs is not None:
        config.experiment_c.train_epochs = train_epochs
    if n_samples is not None:
        config.experiment_c.n_samples = n_samples
    if dream_steps is not None:
        config.experiment_c.dream_steps = dream_steps
    if friction is not None:
        config.experiment_c.friction = friction
    if dt is not None:
        config.experiment_c.dt = dt

    # Extract values from config
    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "../models")
    seed = config.project.seed
    pca_dim = config.experiment_c.pca_dim
    train_epochs = config.experiment_c.train_epochs
    n_samples = config.experiment_c.n_samples
    dream_steps = config.experiment_c.dream_steps
    friction = config.experiment_c.friction
    dt = config.experiment_c.dt
    hidden_dim = config.experiment_c.hidden_dim
    n_dreams = config.experiment_c.n_dreams
    p_train_scale = config.experiment_c.p_train_scale
    q_noise_scale = config.experiment_c.q_noise_scale
    p_noise_scale = config.experiment_c.p_noise_scale
    snapshot_steps = config.experiment_c.snapshot_steps

    print("\n" + "=" * 60)
    print("EXPERIMENT C: Generative Dreaming (MNIST)")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)

    # 1. Load MNIST with PCA
    print(f"\n[1/4] Loading MNIST (PCA → {pca_dim} dims, {n_samples} samples)...")
    train_data, test_data, pca = load_mnist_pca(dim=pca_dim, n_samples=n_samples)

    print(f"  Train data: {train_data.shape}")
    if pca is not None:
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        print(f"  PCA: Disabled (using raw {pca_dim}-dim pixel data)")

    # Convert to CHLU format: we'll treat PCA features as position q
    # and initialize momentum p to zero (or small random values)
    # For simplicity, we'll stack q and p: [q, p] where both are pca_dim
    k1, k2 = jax.random.split(key)

    # Create training data in (q, p) format
    # Start with q = PCA features, p = small random momentum
    n_train = len(train_data)
    p_train = jax.random.normal(k1, (n_train, pca_dim)) * p_train_scale
    train_qp = jnp.concatenate([train_data, p_train], axis=-1)  # (n, 2*pca_dim)

    # Reshape for training: add time dimension (treat each sample as single timestep)
    train_qp = train_qp[:, None, :]  # (n, 1, 2*pca_dim)

    # 2. Train CHLU with PCD
    print(f"\n[2/4] Training CHLU ({train_epochs} epochs)...")
    k2, k3 = jax.random.split(k2)

    chlu = CHLU(dim=pca_dim, hidden=hidden_dim, key=k3)
    chlu, losses = train_chlu(
        chlu,
        train_qp,
        key=k3,
        config=config,
    )

    print(f"  Final loss: {losses[-1]:.6f}")

    # Save trained model
    print("\n  Saving trained model...")
    save_checkpoint(chlu, os.path.join(models_dir, "exp_c_chlu.pkl"), 
                   epoch=train_epochs, loss=float(losses[-1]), config=config)
    print(f"    Saved to {models_dir}")

    # 3. Generative Dreaming: Evolving from noise
    print(f"\n[3/4] Dreaming: evolving from noise ({dream_steps} steps)...")

    k3, k4, k5 = jax.random.split(k3, 3)

    # Initialize random noise states (Shared for both experiments)
    # We use the same noise to show the direct effect of friction on the SAME starting point
    q_noise = jax.random.normal(k4, (n_dreams, pca_dim)) * q_noise_scale
    p_noise = jax.random.normal(k4, (n_dreams, pca_dim)) * p_noise_scale

    # Convert config list to a JAX array for indexing
    # Ensure they are within bounds of dream_steps
    snap_indices = jnp.array([t for t in snapshot_steps if t < dream_steps])

    def run_dream_batch(gamma_val, desc):
        print(f"  Running {desc} batch (gamma={gamma_val})...")
        batch_snapshots = []
        batch_final = []

        for i in range(n_dreams):
            q, p = q_noise[i], p_noise[i]

            # 1. Run full physics (Fast JAX loop)
            # trajectory shape: (dream_steps, 2*pca_dim)
            trajectory = chlu(q, p, steps=dream_steps, dt=dt, gamma=gamma_val)

            # 2. Extract Position (q)
            qs = trajectory[:, :pca_dim]

            # 3. Extract Snapshots (Efficient slicing)
            # This replaces your manual "if step in snapshot_steps" check
            snaps = qs[snap_indices]
            batch_snapshots.append(snaps)

            # 4. Extract Final State (Ground State)
            batch_final.append(qs[-1])

        return jnp.array(batch_final), jnp.array(batch_snapshots)

    # Run experiments
    # EXPERIMENT A: The "Ghosts" (Conserved Energy)
    # gamma=0.0 : Physics mode. Particles orbit the concept.
    final_states_ghosts, snaps_ghosts = run_dream_batch(0.0, "Ghost (No Friction)")

    # EXPERIMENT B: The "Ground States" (Annealed)
    # gamma=friction : Denoising mode. Particles fall into the well.
    final_states_annealed, snaps_annealed = run_dream_batch(
        friction, "Annealed (With Friction)"
    )

    # 4. Visualize: Inverse PCA and Plot Side-by-Side
    print("\n[4/4] Creating visualization...")

    def decode_and_plot(states, filename, title):
        # Inverse PCA (or direct reshape if PCA was skipped)
        if pca is not None:
            images = pca.inverse_transform(states)
        else:
            images = np.array(states)  # Already in pixel space
        images = images.reshape(-1, 28, 28)

        # Save
        full_path = os.path.join(save_dir, filename)
        plot_dreaming_grid(images, full_path, n_rows=4, n_cols=8, image_shape=(28, 28))
        print(f"  Saved {title} to {full_path}")

    # Plot both
    decode_and_plot(final_states_ghosts, "exp3_ghosts.png", "Orbital Ghosts")
    decode_and_plot(final_states_annealed, "exp3_annealed.png", "Annealed Digits")

    print("\n" + "=" * 60)
    print("EXPERIMENT C COMPLETE!")
    print("=" * 60 + "\n")
