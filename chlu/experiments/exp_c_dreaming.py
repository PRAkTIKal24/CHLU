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
from chlu.training.train_generative import train_generative
from chlu.utils.checkpoints import load_checkpoint, save_checkpoint
from chlu.utils.plotting import plot_dreaming_grid


def run_experiment_c(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    use_pretrained: Optional[bool] = None,
    seed: Optional[int] = None,
    pca_dim: Optional[int] = None,
    train_epochs: Optional[int] = None,
    n_samples: Optional[int] = None,
    dream_steps: Optional[int] = None,
    friction: Optional[float] = None,
    dt: Optional[float] = None,
    potential_type: Optional[str] = None,
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
        use_pretrained: Load pre-trained model if available (overrides config)
        seed: Random seed (overrides config)
        pca_dim: PCA dimensionality (overrides config)
        train_epochs: Training epochs (overrides config)
        n_samples: Number of MNIST samples to use (overrides config)
        dream_steps: Steps to evolve during dreaming (overrides config)
        friction: Friction coefficient for energy dissipation (overrides config)
        dt: Time step (overrides config)
        potential_type: Potential network type: 'mlp', 'deep_mlp', 'conv' (overrides config)
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
    if use_pretrained is not None:
        config.experiment_c.use_pretrained = use_pretrained
    if potential_type is not None:
        config.experiment_c.potential_type = potential_type

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
    p_train_scale = config.experiment_c.p_train_scale  # noqa: F841
    q_noise_scale = config.experiment_c.q_noise_scale
    p_noise_scale = config.experiment_c.p_noise_scale
    snapshot_steps = config.experiment_c.snapshot_steps
    use_pretrained = config.experiment_c.use_pretrained
    kinetic_mode = config.experiment_c.kinetic_energy_mode
    potential_type = config.experiment_c.potential_type

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

    # For generative training, we only need position data (no trajectory format)
    # train_data is already in the right format: (n_samples, pca_dim)
    # Data should be normalized to [-1, 1] (already done in load_mnist_pca)

    # 2. Initialize model
    k1, k2 = jax.random.split(key)
    print(f"  CHLU kinetic mode: {kinetic_mode}")
    print(f"  Potential type: {potential_type}")
    chlu = CHLU(
        dim=pca_dim,
        hidden=hidden_dim,
        rest_mass=config.model.rest_mass,
        c=config.model.speed_of_causality,
        kinetic_mode=kinetic_mode,
        potential_type=potential_type,
        key=k2,
    )

    # Train or load model
    chlu_path = os.path.join(models_dir, "exp_c_chlu.pkl")
    model_exists = os.path.exists(chlu_path)

    if use_pretrained and model_exists:
        print(f"\n[2/4] Loading pre-trained model from {models_dir}...")
        chlu, _ = load_checkpoint(chlu_path, chlu)
        print("  ✓ Model loaded successfully")
    else:
        if use_pretrained and not model_exists:
            print("\n[2/4] Pre-trained model not found, training from scratch...")
        else:
            print(
                f"\n[2/4] Training CHLU with Generative PCD ({train_epochs} epochs)..."
            )

        k2, k3 = jax.random.split(k2)
        chlu, losses, target_energy = train_generative(
            chlu,
            train_data,
            key=k3,
            config=config,
            epochs=train_epochs,
        )

        print(f"  Final wake loss: {losses['wake'][-1]:.6f}")
        print(f"  Final sleep loss: {losses['sleep'][-1]:.6f}")
        print(f"  Target energy: {target_energy:.6f}")

        # Save trained model
        print("\n  Saving trained model...")
        save_checkpoint(
            chlu,
            chlu_path,
            epoch=train_epochs,
            loss=float(losses["total"][-1]),
            config=config,
        )
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

    # Plot final states
    decode_and_plot(
        final_states_ghosts, "exp3_ghosts_final.png", "Orbital Ghosts (Final)"
    )
    decode_and_plot(
        final_states_annealed, "exp3_annealed_final.png", "Annealed Digits (Final)"
    )

    # Plot evolution grids: 5 samples × time progression
    print(
        f"\n  Creating evolution grids (5 samples × {len(snap_indices)} snapshots)..."
    )
    n_evolution_samples = 5

    def plot_evolution(snapshots, filename, title):
        """Plot evolution of n samples across all snapshot steps.

        Args:
            snapshots: shape (n_dreams, n_snapshots, pca_dim)
            filename: output filename
            title: plot title
        """
        # Take first n_evolution_samples
        evolution_data = snapshots[:n_evolution_samples]  # (5, n_snapshots, pca_dim)

        # Reshape to (5 * n_snapshots, pca_dim) for batch processing
        n_snaps = evolution_data.shape[1]
        evolution_flat = evolution_data.reshape(-1, pca_dim)

        # Decode all at once
        if pca is not None:
            images = pca.inverse_transform(evolution_flat)
        else:
            images = np.array(evolution_flat)
        images = images.reshape(-1, 28, 28)

        # Save with grid: rows=samples, cols=time steps
        full_path = os.path.join(save_dir, filename)
        plot_dreaming_grid(
            images,
            full_path,
            n_rows=n_evolution_samples,
            n_cols=n_snaps,
            image_shape=(28, 28),
        )
        print(f"  Saved {title} to {full_path}")

    plot_evolution(snaps_ghosts, "exp3_ghosts_evolution.png", "Ghosts Evolution")
    plot_evolution(snaps_annealed, "exp3_annealed_evolution.png", "Annealed Evolution")

    # Plot intermediate snapshots (all samples at each timestep)
    print(f"\n  Saving {len(snap_indices)} intermediate snapshots...")
    for snap_idx, step_num in enumerate(snapshot_steps):
        if step_num < dream_steps:  # Only save valid snapshots
            # Extract all dreams at this snapshot step
            # snaps_ghosts shape: (n_dreams, n_snapshots, pca_dim)
            ghosts_at_step = snaps_ghosts[:, snap_idx, :]  # (n_dreams, pca_dim)
            annealed_at_step = snaps_annealed[:, snap_idx, :]  # (n_dreams, pca_dim)

            decode_and_plot(
                ghosts_at_step,
                f"exp3_ghosts_step_{step_num:03d}.png",
                f"Ghosts at step {step_num}",
            )
            decode_and_plot(
                annealed_at_step,
                f"exp3_annealed_step_{step_num:03d}.png",
                f"Annealed at step {step_num}",
            )

    print("\n" + "=" * 60)
    print("EXPERIMENT C COMPLETE!")
    print("=" * 60 + "\n")
