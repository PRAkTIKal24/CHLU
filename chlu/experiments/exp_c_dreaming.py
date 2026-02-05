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
from chlu.core.integrators import get_temperature_schedule
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
    init_mode: Optional[str] = None,
    centroid_noise_scale: Optional[float] = None,
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
        init_mode: Initialization mode: 'random' or 'centroid' (overrides config)
        centroid_noise_scale: Gaussian perturbation scale when using centroid init (overrides config)
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
    if init_mode is not None:
        config.experiment_c.init_mode = init_mode
    if centroid_noise_scale is not None:
        config.experiment_c.centroid_noise_scale = centroid_noise_scale

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

    # Langevin dynamics parameters
    temperature = config.experiment_c.temperature
    temperature_annealing = config.experiment_c.temperature_annealing
    temperature_start = config.experiment_c.temperature_start
    temperature_end = config.experiment_c.temperature_end
    annealing_schedule = config.experiment_c.annealing_schedule
    init_mode = config.experiment_c.init_mode
    centroid_noise_scale = config.experiment_c.centroid_noise_scale

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

    # Compute dataset centroid for initialization (if using centroid mode)
    dataset_centroid = None
    if init_mode == "centroid":
        print("\n  Computing dataset centroid for initialization...")
        # Load raw MNIST to compute pixel-space centroid
        from sklearn.datasets import fetch_openml

        mnist_raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X_raw = np.array(mnist_raw.data[:n_samples], dtype=np.float32)
        X_raw = (X_raw / 127.5) - 1.0  # Normalize to [-1, 1]

        # Compute centroid in pixel space
        centroid_raw = np.mean(X_raw, axis=0)  # Shape: (784,)

        # Transform to PCA space if PCA is enabled
        if pca is not None:
            dataset_centroid = pca.transform(centroid_raw.reshape(1, -1))[
                0
            ]  # Shape: (pca_dim,)
        else:
            dataset_centroid = centroid_raw  # Already in pixel space

        dataset_centroid = jnp.array(dataset_centroid)
        print(f"  Centroid shape: {dataset_centroid.shape}")
        print(
            f"  Centroid stats: mean={float(jnp.mean(dataset_centroid)):.3f}, std={float(jnp.std(dataset_centroid)):.3f}"
        )

        # Visualize the centroid (pure, before noise)
        centroid_img = centroid_raw.reshape(28, 28)
        centroid_path = os.path.join(save_dir, "exp3_centroid.png")
        plot_dreaming_grid(
            np.array([centroid_img]),
            centroid_path,
            n_rows=1,
            n_cols=1,
            image_shape=(28, 28),
        )
        print(f"  Saved pure centroid visualization to {centroid_path}")

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

    # Ensure random key is properly initialized for both training and loading paths
    k2, k3 = jax.random.split(k2)
    k3, k4, k5 = jax.random.split(k3, 3)

    # Initialize states based on initialization mode
    print(f"\n  Initialization mode: {init_mode}")
    if init_mode == "centroid":
        # Centroid initialization: start from dataset mean + Gaussian noise
        print(f"  Using centroid + Gaussian noise (scale={centroid_noise_scale})")
        q_noise = jnp.tile(dataset_centroid, (n_dreams, 1))  # Replicate centroid
        q_noise = (
            q_noise + jax.random.normal(k4, (n_dreams, pca_dim)) * centroid_noise_scale
        )
    else:
        # Random initialization: pure Gaussian noise
        print(f"  Using random Gaussian noise (scale={q_noise_scale})")
        q_noise = jax.random.normal(k4, (n_dreams, pca_dim)) * q_noise_scale

    # Initialize momentum (always random, using separate key)
    p_noise = jax.random.normal(k5, (n_dreams, pca_dim)) * p_noise_scale

    # Convert config list to a JAX array for indexing
    # Ensure they are within bounds of dream_steps
    snap_indices = jnp.array([t for t in snapshot_steps if t < dream_steps])

    def run_dream_batch(gamma_val, temp_val, use_annealing, desc, dream_key):
        """
        Run dreaming with optional Langevin dynamics and temperature annealing.

        Args:
            gamma_val: Friction coefficient
            temp_val: Temperature (constant if not annealing)
            use_annealing: Whether to apply temperature annealing schedule
            desc: Description for logging
            dream_key: JAX random key for stochastic evolution

        Returns:
            (final_states, snapshots): Final states and snapshot evolution
        """
        print(
            f"  Running {desc} batch (gamma={gamma_val}, temp={temp_val}, annealing={use_annealing})..."
        )
        batch_snapshots = []
        batch_final = []

        # Generate temperature schedule if annealing is enabled
        if use_annealing and temp_val > 0:
            temp_schedule = get_temperature_schedule(
                temperature_start, temperature_end, dream_steps, annealing_schedule
            )
        else:
            # Constant temperature
            temp_schedule = None

        for i in range(n_dreams):
            q, p = q_noise[i], p_noise[i]

            # Split key for this dream
            dream_key, subkey = jax.random.split(dream_key)

            # Choose evolution method based on temperature
            if temp_val > 0.0:
                # Stochastic evolution with Langevin noise
                # Use stochastic_rollout which now supports both constant and scheduled temperature
                temp_to_use = temp_schedule if temp_schedule is not None else temp_val
                trajectory = chlu.stochastic_rollout(
                    q, p, dream_steps, dt, gamma_val, temp_to_use, subkey
                )
            else:
                # Deterministic evolution
                trajectory = chlu(q, p, steps=dream_steps, dt=dt, gamma=gamma_val)

            # Extract Position (q)
            qs = trajectory[:, :pca_dim]

            # Extract Snapshots
            snaps = qs[snap_indices]
            batch_snapshots.append(snaps)

            # Extract Final State
            batch_final.append(qs[-1])

        return jnp.array(batch_final), jnp.array(batch_snapshots), dream_key

    # Run experiments
    # EXPERIMENT A: The "Ghosts" (Conserved Energy)
    # gamma=0.0 : Physics mode. Particles orbit the concept.
    final_states_ghosts, snaps_ghosts, k5 = run_dream_batch(
        0.0, 0.0, False, "Ghost (No Friction)", k5
    )

    # EXPERIMENT B: The "Ground States" (Annealed - Deterministic)
    # gamma=friction : Denoising mode. Particles fall into the well.
    final_states_annealed, snaps_annealed, k5 = run_dream_batch(
        friction, 0.0, False, "Annealed (With Friction)", k5
    )

    # EXPERIMENT C: Thermal Exploration (Optional - if temperature > 0)
    # gamma=0, temperature>0: Particles explore via thermal noise without dissipation
    if temperature > 0.0:
        final_states_thermal, snaps_thermal, k5 = run_dream_batch(
            0.0, temperature, False, "Thermal (No Friction, With Noise)", k5
        )

    # EXPERIMENT D: Annealed Thermal (Optional - if temperature > 0 and friction > 0)
    # gamma=friction, temperature with annealing: Simulated annealing to find modes
    if temperature > 0.0 and friction > 0.0:
        final_states_annealed_thermal, snaps_annealed_thermal, k5 = run_dream_batch(
            friction,
            temperature,
            temperature_annealing,
            "Annealed Thermal (Friction + Noise + Cooling)",
            k5,
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

    # Plot thermal modes if enabled
    if temperature > 0.0:
        decode_and_plot(
            final_states_thermal,
            "exp3_thermal_final.png",
            "Thermal Exploration (Final)",
        )
        if friction > 0.0:
            decode_and_plot(
                final_states_annealed_thermal,
                "exp3_annealed_thermal_final.png",
                "Annealed Thermal (Final)",
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

    if temperature > 0.0:
        plot_evolution(snaps_thermal, "exp3_thermal_evolution.png", "Thermal Evolution")
        if friction > 0.0:
            plot_evolution(
                snaps_annealed_thermal,
                "exp3_annealed_thermal_evolution.png",
                "Annealed Thermal Evolution",
            )

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

            # Thermal modes if enabled
            if temperature > 0.0:
                thermal_at_step = snaps_thermal[:, snap_idx, :]
                decode_and_plot(
                    thermal_at_step,
                    f"exp3_thermal_step_{step_num:03d}.png",
                    f"Thermal at step {step_num}",
                )

                if friction > 0.0:
                    annealed_thermal_at_step = snaps_annealed_thermal[:, snap_idx, :]
                    decode_and_plot(
                        annealed_thermal_at_step,
                        f"exp3_annealed_thermal_step_{step_num:03d}.png",
                        f"Annealed Thermal at step {step_num}",
                    )

    print("\n" + "=" * 60)
    print("EXPERIMENT C COMPLETE!")
    print("=" * 60 + "\n")
