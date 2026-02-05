"""Plotting utilities for CHLU experiments."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
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
    gt_label = f'Ground Truth (Last {n_cycles_to_show} Cycles)' if steps_per_cycle else 'Ground Truth'
    for ax in axes:
        ax.plot(
            gt_plot[:, 0], 
            gt_plot[:, 1], 
            'gray', 
            alpha=0.3, 
            linewidth=2
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
    else:
        lstm_plot = lstm_traj
    axes[0].plot(lstm_plot[:, 0], lstm_plot[:, 1], 'r-', linewidth=1.5)
    axes[0].set_title(titles[0])
    
    # Plot NODE (middle panel) - last N cycles
    node_traj = trajectories["NODE"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        node_plot = node_traj[-steps_to_show:]
    else:
        node_plot = node_traj
    axes[1].plot(node_plot[:, 0], node_plot[:, 1], 'orange', linewidth=1.5)
    axes[1].set_title(titles[1])
    
    # Plot CHLU (right panel) - last N cycles
    chlu_traj = trajectories["CHLU"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        chlu_plot = chlu_traj[-steps_to_show:]
    else:
        chlu_plot = chlu_traj
    axes[2].plot(chlu_plot[:, 0], chlu_plot[:, 1], 'g-', linewidth=1.5)
    axes[2].set_title(titles[2])
    
    # Create unified legend outside the plot area
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.3, label=gt_label),
        Line2D([0], [0], color='r', linewidth=1.5, label='LSTM'),
        Line2D([0], [0], color='orange', linewidth=1.5, label='NODE'),
        Line2D([0], [0], color='g', linewidth=1.5, label='CHLU')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
              ncol=4, frameon=True, fontsize=10)
    
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
    plt.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
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
    # Handle case where axes is a single Axes object (when n_rows=1 and n_cols=1)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
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
    
    # Collect handles and labels from all subplots for unified legend
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    # Create unified legend outside the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
              ncol=len(labels), frameon=True, fontsize=10)
    
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
        
        # Set axis limits based on data
        all_x = np.concatenate([ground_truth[:, 0], trajectories[model_name][:, 0]])
        all_y = np.concatenate([ground_truth[:, 1], trajectories[model_name][:, 1]])
        margin = 0.1
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Collect handles and labels from first subplot for unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Create unified legend outside the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
              ncol=len(labels), frameon=True, fontsize=10)
    
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
    
    # Collect handles and labels from first row for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    plt.suptitle(f'Sine Wave Reconstruction (σ = {sigma})', fontsize=14, fontweight='bold')
    # Create unified legend outside the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
              ncol=3, frameon=True, fontsize=9)
    
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
            
            # Mark initial condition with blue triangle
            if row == 0:
                ax.scatter(noisy_seq[0, 0], noisy_seq[0, 1], marker='^', 
                          c='blue', s=150, edgecolors='darkblue', linewidths=1.5, 
                          zorder=10, label='Initial Conditions')
            else:
                ax.scatter(noisy_seq[0, 0], noisy_seq[0, 1], marker='^', 
                          c='blue', s=150, edgecolors='darkblue', linewidths=1.5, 
                          zorder=10)
            
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
        ax.set_aspect('equal')
    
    # Collect handles and labels from first subplot for unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    plt.suptitle(f'Phase Space Trajectories (σ = {sigma})', fontsize=14, fontweight='bold')
    # Create unified legend outside the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
              ncol=len(labels), frameon=True, fontsize=10)
    
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
            ax.set_ylim(-3, 3)  # Fixed y-axis range for easy comparison
            ax.grid(True, alpha=0.3)
            
            # Title at top of each column
            if row == 0:
                ax.set_title(f'{noise_label}\n(σ = {sigma:.2f})', fontsize=11, fontweight='bold')
            
            # Y-axis label on left side
            if col == 0:
                ax.text(-0.25, 0.5, model_name, transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', rotation=90, 
                       va='center', ha='center')
            
    # Collect handles and labels from first subplot for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    plt.suptitle('Multi-Level Noise Comparison: Model Predictions Across Noise Levels', 
                fontsize=14, fontweight='bold', y=0.995)
    # Create unified legend outside the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
              ncol=3, frameon=True, fontsize=9)
    
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

def plot_kinetic_energy_vs_time(
    clean_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    chlu_model=None,
    dt: float = 0.01,
    n_examples: int = 3,
    sigma: float = 0.5,
    rest_mass: float = 1.0,
    c: float = 5.0,
):
    """
    Plot Kinetic Energy vs time for all three models (CHLU with learned KE).
    
    LSTM/NODE use relativistic kinetic energy from momentum values.
    CHLU computes kinetic energy from its learned relativistic Hamiltonian.
    
    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        chlu_model: CHLU model instance (to compute learned kinetic energy)
        dt: Time step size
        n_examples: Number of example waves to show
        sigma: Noise level used
        rest_mass: Rest mass for relativistic kinetic energy (default: 1.0)
        c: Speed of causality (default: 5.0)
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    
    def compute_relativistic_kinetic(p, rest_mass, c):
        """Compute relativistic kinetic energy: sqrt(p^2 + (m*c)^2)."""
        return jnp.sqrt(p**2 + (rest_mass * c)**2)
    
    def compute_chlu_kinetic(q, p, chlu_model):
        """Extract kinetic energy from CHLU's Hamiltonian."""
        # Get mass parameters
        M = jax.nn.softplus(chlu_model.log_mass)
        M_inv = 1.0 / (M + 1e-6)
        
        if chlu_model.kinetic_mode == "relativistic":
            p_norm_squared = jnp.sum((p * p) * M_inv)
            rest_energy = (chlu_model.rest_mass * chlu_model.c) ** 2
            return jnp.sqrt(p_norm_squared + rest_energy)
        elif chlu_model.kinetic_mode == "newtonian_learned":
            return 0.5 * jnp.sum((p * p) * M_inv)
        else:  # newtonian_identity
            return 0.5 * jnp.sum(p * p)
    
    for row in range(n_examples):
        clean_seq = clean_data[row]
        
        # Limit to first 50 timesteps
        n_steps = min(50, len(clean_seq))
        clean_seq = clean_seq[:n_steps]
        time_steps = np.arange(n_steps) * dt
        
        # Compute clean trajectory kinetic energy
        clean_KE = np.array([
            compute_relativistic_kinetic(clean_seq[t, 1], rest_mass, c)
            for t in range(len(clean_seq))
        ])
        
        for col, (model_name, color) in enumerate(zip(model_names, colors)):
            ax = axes[row, col]
            pred_seq = predictions[model_name][row][:n_steps]
            
            # Plot clean trajectory kinetic energy
            ax.plot(time_steps, clean_KE, 'k-', linewidth=2, label='Clean Signal', alpha=0.7)
            
            # Compute and plot model prediction kinetic energy
            if model_name == "CHLU" and chlu_model is not None:
                # Use CHLU's learned kinetic energy
                pred_KE = np.array([
                    compute_chlu_kinetic(pred_seq[t, 0:1], pred_seq[t, 1:2], chlu_model)
                    for t in range(len(pred_seq))
                ])
            else:
                # Use relativistic kinetic energy from p for LSTM/NODE
                pred_KE = np.array([
                    compute_relativistic_kinetic(pred_seq[t, 1], rest_mass, c)
                    for t in range(len(pred_seq))
                ])
            
            ax.plot(time_steps, pred_KE, color=color, linewidth=2, 
                   label=f'{model_name} Prediction', linestyle='--')
            
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Kinetic Energy', fontsize=10)
            ax.set_title(f'{model_name} - Wave {row + 1}', fontsize=11)
            ax.grid(True, alpha=0.3)
    
    # Collect handles and labels from first row for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    plt.suptitle(f'Kinetic Energy vs Time (σ = {sigma}) - Relativistic', 
                fontsize=14, fontweight='bold')
    # Create unified legend outside the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
              ncol=2, frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Kinetic Energy vs time (relativistic) plot to {save_path}")


def plot_kinetic_energy_vs_time_unified(
    clean_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    dt: float = 0.01,
    n_examples: int = 3,
    sigma: float = 0.5,
):
    """
    Plot Kinetic Energy vs time using same formula for all models.
    
    All models use Newtonian kinetic energy: KE = 0.5 * v^2
    where v = dq/dt is computed from consecutive position values.
    This provides an apples-to-apples comparison based solely on predicted trajectories.
    
    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        dt: Time step size
        n_examples: Number of example waves to show
        sigma: Noise level used
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    
    def compute_newtonian_kinetic_from_trajectory(q_trajectory, dt):
        """Compute KE = 0.5 * v^2 where v = dq/dt from consecutive positions."""
        # Compute velocity using finite differences
        v = np.gradient(q_trajectory, dt)
        # Compute kinetic energy
        return 0.5 * v**2
    
    for row in range(n_examples):
        clean_seq = clean_data[row]
        
        # Limit to first 50 timesteps
        n_steps = min(50, len(clean_seq))
        clean_seq = clean_seq[:n_steps]
        time_steps = np.arange(n_steps) * dt
        
        # Compute clean trajectory kinetic energy from position gradient
        clean_KE = compute_newtonian_kinetic_from_trajectory(
            np.array(clean_seq[:, 0]), dt
        )
        
        for col, (model_name, color) in enumerate(zip(model_names, colors)):
            ax = axes[row, col]
            pred_seq = predictions[model_name][row][:n_steps]
            
            # Plot clean trajectory kinetic energy
            ax.plot(time_steps, clean_KE, 'k-', linewidth=2, label='Clean Signal', alpha=0.7)
            
            # Compute prediction kinetic energy from position gradient (same for all)
            pred_KE = compute_newtonian_kinetic_from_trajectory(
                np.array(pred_seq[:, 0]), dt
            )
            
            ax.plot(time_steps, pred_KE, color=color, linewidth=2, 
                   label=f'{model_name} Prediction', linestyle='--')
            
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Kinetic Energy', fontsize=10)
            ax.set_title(f'{model_name} - Wave {row + 1}', fontsize=11)
            ax.grid(True, alpha=0.3)
    
    # Collect handles and labels from first row for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    plt.suptitle(f'Kinetic Energy vs Time (σ = {sigma}) - Newtonian (v=dq/dt)', 
                fontsize=14, fontweight='bold')
    # Create unified legend outside the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
              ncol=2, frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Kinetic Energy vs time (Newtonian) plot to {save_path}")


def plot_potential_landscape_2d(
    chlu_model,
    trajectory: jnp.ndarray,
    save_path: str,
    grid_resolution: int = 100,
    trajectory_label: str = "CHLU Trajectory",
):
    """
    Plot 2D potential landscape V(q) with trajectory overlay.
    
    Creates a heatmap/contour plot of the learned potential energy function
    over the position space with the trajectory overlaid on top.
    
    Args:
        chlu_model: Trained CHLU model with learned potential_net
        trajectory: Trajectory array (T, 4) with [x, y, vx, vy]
        save_path: Path to save figure
        grid_resolution: Number of grid points along each axis (default: 100)
        trajectory_label: Label for the trajectory (default: "CHLU Trajectory")
    """
    # Extract position coordinates from trajectory
    x_traj = np.array(trajectory[:, 0])
    y_traj = np.array(trajectory[:, 1])
    
    # Create grid bounds with some margin around trajectory
    margin = 0.3
    x_min, x_max = x_traj.min() - margin, x_traj.max() + margin
    y_min, y_max = y_traj.min() - margin, y_traj.max() + margin
    
    # Create meshgrid
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute potential at each grid point
    V = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            q = jnp.array([X[i, j], Y[i, j]])
            V[i, j] = chlu_model.potential_net(q)
    
    # Create figure with two panels: contour + 3D view
    fig = plt.figure(figsize=(16, 6))
    
    # Left panel: 2D contour plot
    ax1 = fig.add_subplot(121)
    
    # Plot filled contours
    contourf = ax1.contourf(X, Y, V, levels=20, cmap='viridis', alpha=0.8)
    
    # Plot contour lines
    contour = ax1.contour(X, Y, V, levels=10, colors='white', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # Overlay trajectory
    ax1.plot(x_traj, y_traj, 'r-', linewidth=2.5, label=trajectory_label, alpha=0.9)
    ax1.scatter(x_traj[0], y_traj[0], c='lime', s=150, marker='o', 
               edgecolors='darkgreen', linewidths=2, zorder=10, label='Start')
    ax1.scatter(x_traj[-1], y_traj[-1], c='red', s=150, marker='X', 
               edgecolors='darkred', linewidths=2, zorder=10, label='End')
    
    # Formatting
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Learned Potential Landscape V(q)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('Potential Energy V(q)', fontsize=11)
    
    # Right panel: 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot surface
    surf = ax2.plot_surface(X, Y, V, cmap='viridis', alpha=0.7, 
                           edgecolor='none', antialiased=True)
    
    # Plot trajectory on surface
    V_traj = np.array([chlu_model.potential_net(jnp.array([x_traj[i], y_traj[i]])) 
                       for i in range(len(x_traj))])
    ax2.plot(x_traj, y_traj, V_traj, 'r-', linewidth=2.5, label=trajectory_label)
    ax2.scatter(x_traj[0], y_traj[0], V_traj[0], c='lime', s=100, 
               marker='o', edgecolors='darkgreen', linewidths=2, zorder=10)
    ax2.scatter(x_traj[-1], y_traj[-1], V_traj[-1], c='red', s=100, 
               marker='X', edgecolors='darkred', linewidths=2, zorder=10)
    
    # Formatting
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('V(q)', fontsize=11)
    ax2.set_title('3D Potential Surface', fontsize=14, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    
    # Add colorbar
    cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
    cbar2.set_label('V(q)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 2D potential landscape to {save_path}")


def plot_potential_surface_3d(
    chlu_model,
    trajectory: jnp.ndarray,
    save_path: str,
    grid_resolution: int = 50,
):
    """
    Plot 3D potential surface V(q) with trajectory.
    
    Creates a 3D surface plot showing the learned potential energy landscape
    with the trajectory plotted as a path on the surface.
    
    Args:
        chlu_model: Trained CHLU model with learned potential_net
        trajectory: Trajectory array (T, 4) with [x, y, vx, vy]
        save_path: Path to save figure
        grid_resolution: Number of grid points along each axis (default: 50)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract position coordinates from trajectory
    x_traj = np.array(trajectory[:, 0])
    y_traj = np.array(trajectory[:, 1])
    
    # Create grid bounds with some margin around trajectory
    margin = 0.3
    x_min, x_max = x_traj.min() - margin, x_traj.max() + margin
    y_min, y_max = y_traj.min() - margin, y_traj.max() + margin
    
    # Create meshgrid
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute potential at each grid point
    V = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            q = jnp.array([X[i, j], Y[i, j]])
            V[i, j] = chlu_model.potential_net(q)
    
    # Compute potential along trajectory
    V_traj = np.array([chlu_model.potential_net(jnp.array([x_traj[i], y_traj[i]])) 
                       for i in range(len(x_traj))])
    
    # Create figure with multiple viewing angles
    fig = plt.figure(figsize=(18, 6))
    
    for idx, (elev, azim, title_suffix) in enumerate([
        (30, 45, '(View 1)'),
        (20, 135, '(View 2)'),
        (60, 225, '(View 3)')
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Plot surface with gradient coloring
        surf = ax.plot_surface(X, Y, V, cmap='viridis', alpha=0.6, 
                             edgecolor='none', antialiased=True, shade=True)
        
        # Plot trajectory on the surface
        ax.plot(x_traj, y_traj, V_traj, 'r-', linewidth=3, label='CHLU Trajectory', zorder=10)
        
        # Mark start and end points
        ax.scatter(x_traj[0], y_traj[0], V_traj[0], c='lime', s=150, 
                  marker='o', edgecolors='darkgreen', linewidths=2, zorder=15, label='Start')
        ax.scatter(x_traj[-1], y_traj[-1], V_traj[-1], c='red', s=150, 
                  marker='X', edgecolors='darkred', linewidths=2, zorder=15, label='End')
        
        # Formatting
        ax.set_xlabel('x Position', fontsize=10)
        ax.set_ylabel('y Position', fontsize=10)
        ax.set_zlabel('Potential V(q)', fontsize=10)
        ax.set_title(f'Learned Potential Surface {title_suffix}', fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        
        if idx == 0:
            ax.legend(fontsize=9, loc='upper left')
    
    plt.suptitle('3D Potential Energy Landscape - Multiple Views', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 3D potential surface to {save_path}")


def plot_force_field(
    chlu_model,
    trajectory: jnp.ndarray,
    save_path: str,
    grid_resolution: int = 20,
):
    """
    Plot force field F = -∇V(q) with trajectory overlay.
    
    Creates a vector field showing the forces derived from the learned potential,
    with arrows color-coded by magnitude. The trajectory is overlaid to show
    how it flows through the force field.
    
    Args:
        chlu_model: Trained CHLU model with learned potential_net
        trajectory: Trajectory array (T, 4) with [x, y, vx, vy]
        save_path: Path to save figure
        grid_resolution: Number of grid points along each axis (default: 20)
    """
    # Extract position coordinates from trajectory
    x_traj = np.array(trajectory[:, 0])
    y_traj = np.array(trajectory[:, 1])
    
    # Create grid bounds with some margin around trajectory
    margin = 0.3
    x_min, x_max = x_traj.min() - margin, x_traj.max() + margin
    y_min, y_max = y_traj.min() - margin, y_traj.max() + margin
    
    # Create meshgrid for vector field
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute gradient of potential at each grid point: F = -∇V
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    
    # Use JAX's automatic differentiation to compute gradient
    grad_V = jax.grad(chlu_model.potential_net)
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            q = jnp.array([X[i, j], Y[i, j]])
            grad = grad_V(q)
            # Force is negative gradient of potential
            Fx[i, j] = -grad[0]
            Fy[i, j] = -grad[1]
    
    # Compute force magnitude for coloring
    F_mag = np.sqrt(Fx**2 + Fy**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot vector field with color based on magnitude
    quiver = ax.quiver(X, Y, Fx, Fy, F_mag, 
                      cmap='plasma', 
                      scale=20,
                      scale_units='xy',
                      width=0.004,
                      alpha=0.7,
                      pivot='mid')
    
    # Add colorbar for force magnitude
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label('Force Magnitude |F|', fontsize=12)
    
    # Overlay trajectory
    ax.plot(x_traj, y_traj, 'cyan', linewidth=3, label='CHLU Trajectory', 
           alpha=0.9, zorder=10)
    ax.scatter(x_traj[0], y_traj[0], c='lime', s=200, marker='o', 
              edgecolors='darkgreen', linewidths=2.5, zorder=15, label='Start')
    ax.scatter(x_traj[-1], y_traj[-1], c='red', s=200, marker='X', 
              edgecolors='darkred', linewidths=2.5, zorder=15, label='End')
    
    # Formatting
    ax.set_xlabel('x Position', fontsize=13)
    ax.set_ylabel('y Position', fontsize=13)
    ax.set_title('Force Field: F = -∇V(q)', fontsize=15, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved force field plot to {save_path}")


def plot_energy_conservation(
    chlu_model,
    trajectories: dict,
    save_path: str,
    dt: float = 0.01,
    n_steps_to_plot: int = None,
):
    """
    Plot energy conservation comparison across models.
    
    Shows total energy H(q,p), kinetic energy T(p), and potential energy V(q)
    over time for CHLU, NODE, and LSTM. CHLU should show flat energy (conserved),
    while baselines drift.
    
    Args:
        chlu_model: Trained CHLU model (to compute Hamiltonian)
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        save_path: Path to save figure
        dt: Time step size
        n_steps_to_plot: Number of steps to plot (default: all)
    """
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ['red', 'orange', 'green']
    
    # Create figure with 4 panels: Total Energy, Kinetic, Potential, and Combined
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for model_name, color in zip(model_names, colors):
        traj = trajectories[model_name]
        
        # Limit steps if specified
        if n_steps_to_plot is not None:
            traj = traj[:n_steps_to_plot]
        
        n_steps = len(traj)
        time = np.arange(n_steps) * dt
        
        # Extract positions and momenta
        q_seq = traj[:, :2]  # First 2 components are positions
        p_seq = traj[:, 2:]  # Last 2 components are momenta
        
        # Compute energies at each timestep
        H_seq = []
        T_seq = []
        V_seq = []
        
        for i in range(n_steps):
            q = jnp.array(q_seq[i])
            p = jnp.array(p_seq[i])
            
            # Compute Hamiltonian (total energy)
            H = chlu_model.H(q, p)
            H_seq.append(H)
            
            # Compute kinetic energy
            M = jax.nn.softplus(chlu_model.log_mass)
            M_inv = 1.0 / (M + 1e-6)
            
            if chlu_model.kinetic_mode == "relativistic":
                p_norm_squared = jnp.sum((p * p) * M_inv)
                rest_energy = (chlu_model.rest_mass * chlu_model.c) ** 2
                T = chlu_model.c * jnp.sqrt(p_norm_squared + rest_energy)
            elif chlu_model.kinetic_mode == "newtonian_learned":
                T = 0.5 * jnp.sum((p * p) * M_inv)
            else:  # newtonian_identity
                T = 0.5 * jnp.sum(p * p)
            
            T_seq.append(T)
            
            # Compute potential energy
            V = chlu_model.potential_net(q)
            V_seq.append(V)
        
        H_seq = np.array(H_seq)
        T_seq = np.array(T_seq)
        V_seq = np.array(V_seq)
        
        # Panel 1: Total Energy (Hamiltonian)
        axes[0].plot(time, H_seq, color=color, linewidth=2, label=model_name, alpha=0.8)
        
        # Panel 2: Kinetic Energy
        axes[1].plot(time, T_seq, color=color, linewidth=2, label=model_name, alpha=0.8)
        
        # Panel 3: Potential Energy
        axes[2].plot(time, V_seq, color=color, linewidth=2, label=model_name, alpha=0.8)
        
        # Panel 4: Kinetic + Potential (stacked or separate)
        axes[3].plot(time, T_seq, color=color, linewidth=1.5, 
                    linestyle='--', alpha=0.6, label=f'{model_name} T')
        axes[3].plot(time, V_seq, color=color, linewidth=1.5, 
                    linestyle=':', alpha=0.6, label=f'{model_name} V')
    
    # Formatting for each panel
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Total Energy H(q,p)', fontsize=12)
    axes[0].set_title('Total Energy (Hamiltonian)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Kinetic Energy T(p)', fontsize=12)
    axes[1].set_title('Kinetic Energy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Potential Energy V(q)', fontsize=12)
    axes[2].set_title('Potential Energy', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10, loc='best')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].set_ylabel('Energy', fontsize=12)
    axes[3].set_title('Energy Components (T and V)', fontsize=13, fontweight='bold')
    axes[3].legend(fontsize=9, loc='best', ncol=2)
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Energy Conservation Comparison: CHLU vs Baselines', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved energy conservation plot to {save_path}")