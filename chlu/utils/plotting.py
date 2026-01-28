"""Plotting utilities for CHLU experiments."""

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


def plot_three_panel_trajectories(
    trajectories: dict,
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
):
    """
    Plot three-panel figure comparing model trajectories.
    
    Used for Experiment A: Stability comparison.
    
    Args:
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        ground_truth: Ground truth trajectory (T, 4) [x, y, vx, vy]
        titles: List of 3 subplot titles
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot ground truth on all panels (in gray)
    for ax in axes:
        ax.plot(
            ground_truth[:, 0], 
            ground_truth[:, 1], 
            'gray', 
            alpha=0.3, 
            linewidth=2,
            label='Ground Truth'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Plot LSTM (left panel)
    lstm_traj = trajectories["LSTM"]
    axes[0].plot(lstm_traj[:, 0], lstm_traj[:, 1], 'r-', linewidth=1.5, label='LSTM')
    axes[0].set_title(titles[0])
    axes[0].legend()
    
    # Plot NODE (middle panel)
    node_traj = trajectories["NODE"]
    axes[1].plot(node_traj[:, 0], node_traj[:, 1], 'orange', linewidth=1.5, label='NODE')
    axes[1].set_title(titles[1])
    axes[1].legend()
    
    # Plot CHLU (right panel)
    chlu_traj = trajectories["CHLU"]
    axes[2].plot(chlu_traj[:, 0], chlu_traj[:, 1], 'g-', linewidth=1.5, label='CHLU')
    axes[2].set_title(titles[2])
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved three-panel trajectory plot to {save_path}")


def plot_noise_curves(
    sigmas: jnp.ndarray,
    mse_dict: dict,
    save_path: str,
):
    """
    Plot noise robustness curves.
    
    Used for Experiment B: Noise rejection comparison.
    
    Args:
        sigmas: Array of noise levels
        mse_dict: Dict with keys "LSTM", "NODE", "CHLU" and MSE arrays
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Plot curves
    plt.plot(sigmas, mse_dict["LSTM"], 'r-o', linewidth=2, markersize=6, label='LSTM')
    plt.plot(sigmas, mse_dict["NODE"], 'orange', marker='s', linewidth=2, markersize=6, label='NODE')
    plt.plot(sigmas, mse_dict["CHLU"], 'g-^', linewidth=2, markersize=6, label='CHLU')
    
    plt.xlabel('Noise Sigma (σ)', fontsize=12)
    plt.ylabel('Reconstruction MSE', fontsize=12)
    plt.title('Noise Robustness: The Filter Effect', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved noise curve plot to {save_path}")


def plot_dreaming_grid(
    images: jnp.ndarray,
    save_path: str,
    n_rows: int = 4,
    n_cols: int = 8,
    image_shape: tuple = (28, 28),
):
    """
    Plot grid of evolving images for generative dreaming.
    
    Used for Experiment C: MNIST dreaming visualization.
    
    Args:
        images: Array of images (n_images, height * width) or (n_images, height, width)
        save_path: Path to save figure
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        image_shape: Shape to reshape images to (height, width)
    """
    # Reshape images if needed
    if images.ndim == 2:
        images = images.reshape(-1, *image_shape)
    
    n_images = min(len(images), n_rows * n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten()
    
    for i in range(n_images):
        axes[i].imshow(np.array(images[i]), cmap='gray')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        axes[i].axis('off')
    
    plt.suptitle('CHLU Generative Dreaming: Noise → Digit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dreaming grid to {save_path}")
