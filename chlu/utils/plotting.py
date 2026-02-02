"""Plotting utilities for CHLU experiments."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import numpy as np


def plot_three_panel_trajectories(
    trajectories: dict,
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
    steps_per_cycle: int = None,
    n_cycles_to_show: int = 3,
):
    """
    Plot three-panel figure comparing model trajectories.
    
    Used for Experiment A: Stability comparison.
    Shows only the last N cycles for focused comparison.
    
    Args:
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        ground_truth: Ground truth trajectory (T, 4) [x, y, vx, vy]
        titles: List of 3 subplot titles
        save_path: Path to save figure
        steps_per_cycle: If provided, only plot the last n_cycles_to_show cycles
        n_cycles_to_show: Number of final cycles to display (default: 3)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract last N cycles if steps_per_cycle is provided
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        gt_plot = ground_truth[-steps_to_show:]
    else:
        gt_plot = ground_truth
    
    # Plot ground truth on all panels (in gray)
    for ax in axes:
        label = f'Ground Truth (Last {n_cycles_to_show} Cycles)' if steps_per_cycle else 'Ground Truth'
        ax.plot(
            gt_plot[:, 0], 
            gt_plot[:, 1], 
            'gray', 
            alpha=0.3, 
            linewidth=2,
            label=label
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Plot LSTM (left panel) - last N cycles
    lstm_traj = trajectories["LSTM"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        lstm_plot = lstm_traj[-steps_to_show:]
        label = f'LSTM (Last {n_cycles_to_show} Cycles)'
    else:
        lstm_plot = lstm_traj
        label = 'LSTM'
    axes[0].plot(lstm_plot[:, 0], lstm_plot[:, 1], 'r-', linewidth=1.5, label=label)
    axes[0].set_title(titles[0])
    axes[0].legend()
    
    # Plot NODE (middle panel) - last N cycles
    node_traj = trajectories["NODE"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        node_plot = node_traj[-steps_to_show:]
        label = f'NODE (Last {n_cycles_to_show} Cycles)'
    else:
        node_plot = node_traj
        label = 'NODE'
    axes[1].plot(node_plot[:, 0], node_plot[:, 1], 'orange', linewidth=1.5, label=label)
    axes[1].set_title(titles[1])
    axes[1].legend()
    
    # Plot CHLU (right panel) - last N cycles
    chlu_traj = trajectories["CHLU"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        chlu_plot = chlu_traj[-steps_to_show:]
        label = f'CHLU (Last {n_cycles_to_show} Cycles)'
    else:
        chlu_plot = chlu_traj
        label = 'CHLU'
    axes[2].plot(chlu_plot[:, 0], chlu_plot[:, 1], 'g-', linewidth=1.5, label=label)
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
    Automatically unnormalizes images from [-1, 1] to [0, 255] for display.
    
    Args:
        images: Array of images (n_images, height * width) or (n_images, height, width)
                Expected to be in [-1, 1] range (will be unnormalized for display)
        save_path: Path to save figure
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        image_shape: Shape to reshape images to (height, width)
    """
    # Reshape images if needed
    if images.ndim == 2:
        images = images.reshape(-1, *image_shape)
    
    # Unnormalize from [-1, 1] to [0, 255]
    images = np.array(images)
    images = (images + 1.0) * 127.5
    images = np.clip(images, 0, 255).astype(np.uint8)
    
    n_images = min(len(images), n_rows * n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten()
    
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray', vmin=0, vmax=255)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        axes[i].axis('off')
    
    plt.suptitle('CHLU Generative Dreaming: Noise → Digit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dreaming grid to {save_path}")

def plot_trajectory_evolution(
    trajectories: dict,
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
    n_snapshots: int = 10,
    steps_per_cycle: int = None,
    n_cycles_solid: int = 3,
):
    """
    Plot trajectory evolution with transparent intermediate steps and final trajectory.
    
    Shows how each model's trajectory evolves over time with progressive snapshots.
    If steps_per_cycle is provided, shows first cycles very lightly and 
    only the last N cycles in solid color.
    
    Args:
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        ground_truth: Ground truth trajectory (T, 4) [x, y, vx, vy]
        titles: List of 3 subplot titles
        save_path: Path to save figure
        n_snapshots: Number of intermediate snapshots to show
        steps_per_cycle: If provided, plot first cycles lightly, last n_cycles_solid solid
        n_cycles_solid: Number of final cycles to show in solid color (default: 3)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    
    for idx, (ax, model_name, color, title) in enumerate(zip(axes, model_names, colors, titles)):
        # Plot ground truth - first cycles lightly, last N cycles solid
        if steps_per_cycle is not None:
            steps_solid = n_cycles_solid * steps_per_cycle
            # First cycles slightly transparent
            gt_early = ground_truth[:-steps_solid]
            if len(gt_early) > 0:
                ax.plot(
                    gt_early[:, 0], 
                    gt_early[:, 1], 
                    'gray', 
                    alpha=0.15, 
                    linewidth=1,
                    zorder=1
                )
            # Last N cycles solid
            gt_last = ground_truth[-steps_solid:]
            ax.plot(
                gt_last[:, 0], 
                gt_last[:, 1], 
                'gray', 
                alpha=0.5, 
                linewidth=2,
                label=f'Ground Truth (Last {n_cycles_solid} Cycles)',
                zorder=2
            )
        else:
            ax.plot(
                ground_truth[:, 0], 
                ground_truth[:, 1], 
                'gray', 
                alpha=0.3, 
                linewidth=2,
                label='Ground Truth',
                zorder=1
            )
        
        traj = trajectories[model_name]
        n_steps = len(traj)
        
        if steps_per_cycle is not None:
            steps_solid = n_cycles_solid * steps_per_cycle
            # Plot first cycles slightly transparent
            traj_early = traj[:-steps_solid]
            if len(traj_early) > 0:
                ax.plot(
                    traj_early[:, 0], 
                    traj_early[:, 1], 
                    color=color,
                    alpha=0.15,
                    linewidth=1,
                    zorder=3
                )
            
            # Plot last N cycles solid
            traj_last = traj[-steps_solid:]
            ax.plot(
                traj_last[:, 0], 
                traj_last[:, 1], 
                color=color,
                linewidth=2.5,
                label=f'{model_name} (Last {n_cycles_solid} Cycles)',
                zorder=4
            )
        else:
            # Original behavior: intermediate snapshots with increasing transparency
            snapshot_indices = np.linspace(n_steps // n_snapshots, n_steps, n_snapshots, dtype=int)
            
            for i, snap_idx in enumerate(snapshot_indices[:-1]):
                alpha = 0.1 + (i / n_snapshots) * 0.3  # Fade from 0.1 to 0.4
                ax.plot(
                    traj[:snap_idx, 0], 
                    traj[:snap_idx, 1], 
                    color=color,
                    alpha=alpha,
                    linewidth=0.8,
                    zorder=2
                )
            
            # Plot final trajectory with solid line
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=color,
                linewidth=2,
                label=f'{model_name} (final)',
                zorder=3
            )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory evolution plot to {save_path}")


def create_trajectory_animation(
    trajectories: dict,
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
    fps: int = 30,
    n_frames: int = 100,
):
    """
    Create animated GIF showing trajectory evolution over time.
    
    Args:
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        ground_truth: Ground truth trajectory (T, 4) [x, y, vx, vy]
        titles: List of 3 subplot titles
        save_path: Path to save GIF (should end in .gif)
        fps: Frames per second
        n_frames: Number of frames in animation
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    
    # Determine max length
    max_len = max(len(trajectories[name]) for name in model_names)
    frame_indices = np.linspace(10, max_len, n_frames, dtype=int)
    
    # Initialize plots
    lines = {}
    for idx, (ax, model_name, color, title) in enumerate(zip(axes, model_names, colors, titles)):
        # Plot ground truth (static)
        ax.plot(
            ground_truth[:, 0], 
            ground_truth[:, 1], 
            'gray', 
            alpha=0.3, 
            linewidth=2,
            label='Ground Truth'
        )
        
        # Initialize trajectory line
        line, = ax.plot([], [], color=color, linewidth=2, label=model_name)
        lines[model_name] = line
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set axis limits based on data
        all_x = np.concatenate([ground_truth[:, 0], trajectories[model_name][:, 0]])
        all_y = np.concatenate([ground_truth[:, 1], trajectories[model_name][:, 1]])
        margin = 0.1
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    def update(frame_idx):
        """Update function for animation."""
        idx = frame_indices[frame_idx]
        for model_name in model_names:
            traj = trajectories[model_name]
            end_idx = min(idx, len(traj))
            lines[model_name].set_data(traj[:end_idx, 0], traj[:end_idx, 1])
        return list(lines.values())
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000/fps, blit=True
    )
    
    plt.tight_layout()
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Saved trajectory animation to {save_path}")


def plot_sine_wave_comparison(
    clean_data: jnp.ndarray,
    noisy_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    n_examples: int = 3,
    sigma: float = 0.5,
):
    """
    Plot expected vs generated sine waves for each algorithm.
    
    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        noisy_data: Noisy test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        n_examples: Number of example waves to show
        sigma: Noise level used
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    
    for row in range(n_examples):
        clean_seq = clean_data[row]
        noisy_seq = noisy_data[row]
        time_steps = np.arange(len(clean_seq))
        
        for col, (model_name, color) in enumerate(zip(model_names, colors)):
            ax = axes[row, col]
            pred_seq = predictions[model_name][row]
            
            # Plot clean (expected)
            ax.plot(time_steps, clean_seq[:, 0], 'k-', linewidth=2, label='Expected', alpha=0.7)
            
            # Plot noisy input
            ax.scatter(time_steps[::5], noisy_seq[::5, 0], c='gray', s=10, alpha=0.4, label=f'Noisy Input (σ={sigma})')
            
            # Plot prediction
            ax.plot(time_steps, pred_seq[:, 0], color=color, linewidth=2, label=f'{model_name} Output', linestyle='--')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{model_name} - Wave {row + 1}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    plt.suptitle(f'Sine Wave Reconstruction (σ = {sigma})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sine wave comparison to {save_path}")


def plot_phase_space(
    clean_data: jnp.ndarray,
    noisy_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    n_examples: int = 3,
    sigma: float = 0.5,
):
    """
    Plot phase space (q vs p) for sine wave predictions.
    
    Args:
        clean_data: Clean test data (n_waves, steps, 2) where dim 0 = q, dim 1 = p
        noisy_data: Noisy test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        n_examples: Number of example waves to show
        sigma: Noise level used
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    
    for col, (model_name, color, ax) in enumerate(zip(model_names, colors, axes)):
        # Plot all examples for this model
        for row in range(n_examples):
            clean_seq = clean_data[row]
            noisy_seq = noisy_data[row]
            pred_seq = predictions[model_name][row]
            
            # Plot clean trajectory (expected)
            if row == 0:
                ax.plot(clean_seq[:, 0], clean_seq[:, 1], 'k-', linewidth=1.5, 
                       alpha=0.5, label='Expected')
            else:
                ax.plot(clean_seq[:, 0], clean_seq[:, 1], 'k-', linewidth=1.5, alpha=0.5)
            
            # Plot noisy input points
            if row == 0:
                ax.scatter(noisy_seq[::10, 0], noisy_seq[::10, 1], c='gray', 
                          s=15, alpha=0.3, label=f'Noisy (σ={sigma})')
            else:
                ax.scatter(noisy_seq[::10, 0], noisy_seq[::10, 1], c='gray', 
                          s=15, alpha=0.3)
            
            # Plot prediction
            if row == 0:
                ax.plot(pred_seq[:, 0], pred_seq[:, 1], color=color, linewidth=2, 
                       linestyle='--', label=f'{model_name}')
            else:
                ax.plot(pred_seq[:, 0], pred_seq[:, 1], color=color, linewidth=2, 
                       linestyle='--')
        
        ax.set_xlabel('Position (q)')
        ax.set_ylabel('Momentum (p)')
        ax.set_title(f'{model_name} Phase Space')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.suptitle(f'Phase Space Trajectories (σ = {sigma})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved phase space plot to {save_path}")


def plot_multi_noise_grid(
    clean_data: jnp.ndarray,
    noise_levels_data: dict,
    save_path: str,
    example_idx: int = 0,
):
    """
    Plot multi-level noise comparison grid.
    
    Shows predictions at low, medium, and high noise levels for each model.
    Layout: 3 rows (LSTM, NODE, CHLU) x 3 columns (low, medium, high noise)
    
    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        noise_levels_data: Dict with structure:
            {
                'sigmas': [low_sigma, mid_sigma, high_sigma],
                'noisy_inputs': [low_noisy, mid_noisy, high_noisy],
                'predictions': {
                    'LSTM': [low_pred, mid_pred, high_pred],
                    'NODE': [low_pred, mid_pred, high_pred],
                    'CHLU': [low_pred, mid_pred, high_pred]
                }
            }
        save_path: Path to save figure
        example_idx: Which test example to show (default: 0)
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    sigmas = noise_levels_data['sigmas']
    noisy_inputs = noise_levels_data['noisy_inputs']
    predictions = noise_levels_data['predictions']
    
    clean_seq = clean_data[example_idx]
    time_steps = np.arange(len(clean_seq))
    
    # Column titles
    noise_labels = ['Low Noise', 'Medium Noise', 'High Noise']
    
    for row, (model_name, color) in enumerate(zip(model_names, colors)):
        for col, (sigma, noisy_data, noise_label) in enumerate(zip(sigmas, noisy_inputs, noise_labels)):
            ax = axes[row, col]
            
            noisy_seq = noisy_data[example_idx]
            pred_seq = predictions[model_name][col][example_idx]
            
            # Plot clean signal (ground truth)
            ax.plot(time_steps, clean_seq[:, 0], 'k-', linewidth=2.5, 
                   label='Clean Signal', alpha=0.7, zorder=3)
            
            # Plot noisy input (scatter to show noise)
            ax.scatter(time_steps[::3], noisy_seq[::3, 0], c='gray', s=12, 
                      alpha=0.4, label=f'Noisy Input', zorder=1)
            
            # Plot model prediction
            ax.plot(time_steps, pred_seq[:, 0], color=color, linewidth=2, 
                   label=f'{model_name} Prediction', linestyle='--', zorder=2)
            
            # Styling
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Title at top of each column
            if row == 0:
                ax.set_title(f'{noise_label}\n(σ = {sigma:.2f})', fontsize=11, fontweight='bold')
            
            # Y-axis label on left side
            if col == 0:
                ax.text(-0.25, 0.5, model_name, transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', rotation=90, 
                       va='center', ha='center')
            
            # Legend only on first subplot
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle('Multi-Level Noise Comparison: Model Predictions Across Noise Levels', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-noise grid to {save_path}")


def plot_noise_heatmap(
    sigmas: jnp.ndarray,
    temporal_errors: dict,
    save_path: str,
):
    """
    Plot noise level heatmap showing error evolution over time.
    
    Creates a 2D heatmap for each model showing how reconstruction error
    varies across noise levels (y-axis) and time steps (x-axis).
    
    Args:
        sigmas: Array of noise levels (n_sigma,)
        temporal_errors: Dict with structure:
            {
                'LSTM': array of shape (n_sigma, n_steps),
                'NODE': array of shape (n_sigma, n_steps),
                'CHLU': array of shape (n_sigma, n_steps)
            }
            Each entry [i, t] contains the mean squared error at noise level i and timestep t
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = ["LSTM", "NODE", "CHLU"]
    cmaps = ['Reds', 'Oranges', 'Greens']
    
    for ax, model_name, cmap in zip(axes, model_names, cmaps):
        error_matrix = np.array(temporal_errors[model_name])
        n_sigma, n_steps = error_matrix.shape
        
        # Create heatmap
        im = ax.imshow(
            error_matrix,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            extent=[0, n_steps, float(sigmas[0]), float(sigmas[-1])],
            interpolation='bilinear'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Squared Error', fontsize=10)
        
        # Styling
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Noise Level (σ)', fontsize=11)
        ax.set_title(f'{model_name} Error Heatmap', fontsize=12, fontweight='bold')
        ax.grid(False)
    
    plt.suptitle('Temporal Error Evolution Across Noise Levels', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved noise heatmap to {save_path}")
