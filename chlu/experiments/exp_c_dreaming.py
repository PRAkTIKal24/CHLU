"""Experiment C: Generative "Dreaming" on MNIST.

Validates PCD training and generative capability of CHLU.
"""

import os
import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU
from chlu.data.mnist import load_mnist_pca
from chlu.training.train import train_chlu
from chlu.utils.plotting import plot_dreaming_grid


def run_experiment_c(
    save_dir: str = "results/",
    seed: int = 42,
    pca_dim: int = 32,
    train_epochs: int = 1000,
    n_samples: int = 5000,
    dream_steps: int = 100,
    friction: float = 0.01,
    dt: float = 0.01,
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
        save_dir: Directory to save results
        seed: Random seed
        pca_dim: PCA dimensionality
        train_epochs: Training epochs
        n_samples: Number of MNIST samples to use
        dream_steps: Steps to evolve during dreaming
        friction: Friction coefficient for energy dissipation
        dt: Time step
    """
    print("\n" + "="*60)
    print("EXPERIMENT C: Generative Dreaming (MNIST)")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    
    # 1. Load MNIST with PCA
    print(f"\n[1/4] Loading MNIST (PCA → {pca_dim} dims, {n_samples} samples)...")
    train_data, test_data, pca = load_mnist_pca(dim=pca_dim, n_samples=n_samples)
    
    print(f"  Train data: {train_data.shape}")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Convert to CHLU format: we'll treat PCA features as position q
    # and initialize momentum p to zero (or small random values)
    # For simplicity, we'll stack q and p: [q, p] where both are pca_dim
    k1, k2 = jax.random.split(key)
    
    # Create training data in (q, p) format
    # Start with q = PCA features, p = small random momentum
    n_train = len(train_data)
    p_train = jax.random.normal(k1, (n_train, pca_dim)) * 0.1
    train_qp = jnp.concatenate([train_data, p_train], axis=-1)  # (n, 2*pca_dim)
    
    # Reshape for training: add time dimension (treat each sample as single timestep)
    train_qp = train_qp[:, None, :]  # (n, 1, 2*pca_dim)
    
    # 2. Train CHLU with PCD
    print(f"\n[2/4] Training CHLU ({train_epochs} epochs)...")
    k2, k3 = jax.random.split(k2)
    
    chlu = CHLU(dim=pca_dim, hidden=32, key=k3)
    chlu, losses = train_chlu(
        chlu,
        train_qp,
        key=k3,
        epochs=train_epochs,
        dt=dt,
    )
    
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # 3. Generative dreaming: start from random noise
    print(f"\n[3/4] Dreaming: evolving from noise ({dream_steps} steps)...")
    
    k3, k4 = jax.random.split(k3)
    
    # Number of dream sequences to visualize
    n_dreams = 32  # 4x8 grid
    
    # Initialize random noise states
    q_noise = jax.random.normal(k4, (n_dreams, pca_dim)) * 2.0
    p_noise = jax.random.normal(k4, (n_dreams, pca_dim)) * 0.5
    
    # Evolve each noise state with friction (energy dissipation)
    dream_sequences = []
    
    for i in range(n_dreams):
        q, p = q_noise[i], p_noise[i]
        sequence = []
        
        # Record snapshots at different timesteps
        snapshot_steps = [0, 10, 30, 50, 70, 100]
        current_step = 0
        
        for step in range(dream_steps):
            # Record snapshot
            if step in snapshot_steps:
                sequence.append(q)
                current_step += 1
            
            # Evolve one step
            q, p = chlu.step((q, p), dt)
            
            # Apply friction to momentum (energy dissipation)
            p = (1 - friction) * p
        
        # Final snapshot
        if dream_steps not in snapshot_steps:
            sequence.append(q)
        
        dream_sequences.append(sequence)
    
    # 4. Visualize: inverse PCA to get images
    print("\n[4/4] Creating visualization...")
    
    # Take final states from dreams
    final_states = jnp.array([seq[-1] for seq in dream_sequences])
    
    # Inverse PCA transform
    final_images = pca.inverse_transform(final_states)
    
    # Reshape to 28x28
    final_images = final_images.reshape(-1, 28, 28)
    
    # Plot
    save_path = os.path.join(save_dir, "exp3_dreaming.png")
    plot_dreaming_grid(
        final_images, 
        save_path, 
        n_rows=4, 
        n_cols=8,
        image_shape=(28, 28)
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT C COMPLETE!")
    print(f"Results saved to: {save_path}")
    print("="*60 + "\n")
