"""Experiment A: Eternal Memory Stability Test.

Compares long-horizon stability across CHLU, NODE, and LSTM.
"""

import os
from typing import Optional

import jax

from chlu.config import CHLUConfig, get_default_config
from chlu.core.baselines import LSTMPredictor, NeuralODE
from chlu.core.chlu_unit import CHLU
from chlu.data.figure8 import generate_figure8
from chlu.training.train import train_chlu
from chlu.training.train_baselines import train_lstm, train_neural_ode
from chlu.utils.plotting import (
    create_trajectory_animation,
    plot_three_panel_trajectories,
    plot_trajectory_evolution,
)


def run_experiment_a(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    seed: Optional[int] = None,
    n_train_cycles: Optional[int] = None,
    n_test_cycles: Optional[int] = None,
    train_epochs: Optional[int] = None,
    dt: Optional[float] = None,
):
    """
    Experiment A: "Eternal Memory" Stability Test.

    Protocol:
        1. Generate Figure-8 training data (3 complete cycles)
        2. Train all 3 models on windowed sub-sequences
        3. Generate test data (50 complete cycles)
        4. Run each model autonomously from last training point (free run)
        5. Compare trajectory stability and geometry preservation

    Expected outcomes:
        - LSTM: Drift/explosion (Chaos)
        - NODE: Spiral inward (Dissipation)
        - CHLU: Stable Figure-8 (Conservation)

    Args:
        config: CHLUConfig object (if None, uses defaults)
        save_dir: Directory to save results (overrides config)
        seed: Random seed (overrides config)
        n_train_cycles: Number of cycles to train on (overrides config)
        n_test_cycles: Number of cycles to extrapolate (overrides config)
        train_epochs: Training epochs per model (overrides config)
        dt: Time step size (overrides config)
    """
    # Load config with overrides
    if config is None:
        config = get_default_config()

    # Apply overrides
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed
    if n_train_cycles is not None:
        config.experiment_a.n_train_cycles = n_train_cycles
    if n_test_cycles is not None:
        config.experiment_a.n_test_cycles = n_test_cycles
    if train_epochs is not None:
        config.experiment_a.train_epochs = train_epochs
    if dt is not None:
        config.experiment_a.dt = dt

    # Extract values from config
    save_dir = config.project.save_dir or "results/"
    seed = config.project.seed
    n_train_cycles = config.experiment_a.n_train_cycles
    n_test_cycles = config.experiment_a.n_test_cycles
    train_epochs = config.experiment_a.train_epochs
    dt = config.experiment_a.dt
    window_size = config.experiment_a.window_size
    train_steps = config.experiment_a.train_steps
    test_steps = config.experiment_a.test_steps
    steps_per_cycle = config.experiment_a.steps_per_cycle
    # Figure-8 is always 2D in position space (x,y) -> chlu_dim must be 2
    chlu_dim = 2
    node_dim = config.experiment_a.node_dim
    hidden_dim = config.experiment_a.hidden_dim

    print("\n" + "=" * 60)
    print("EXPERIMENT A: Eternal Memory Stability Test")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)

    # 1. Generate Figure-8 data
    print("\n[1/5] Generating Figure-8 trajectory...")
    print(f"  Training: {n_train_cycles} cycles ({train_steps} steps)")
    print(f"  Testing:  {n_test_cycles} cycles ({test_steps} steps)")
    print(f"  Window size: {window_size} steps")
    print(f"  Steps per cycle: {steps_per_cycle}")

    k1, k2 = jax.random.split(key)

    # Generate training data: 3 complete cycles
    train_data = generate_figure8(k1, n_cycles=n_train_cycles, dt=dt)

    # Generate test data: 50 complete cycles
    k2, k3 = jax.random.split(k2)
    test_data = generate_figure8(k3, n_cycles=n_test_cycles, dt=dt)

    print(f"  Training data shape: {train_data.shape}")
    print(f"  Test data shape: {test_data.shape}")

    # 2. Initialize models
    print("\n[2/5] Initializing models (CHLU, NODE, LSTM)...")
    k3, k4, k5, k6 = jax.random.split(k3, 4)

    chlu = CHLU(dim=chlu_dim, hidden=hidden_dim, key=k4)
    node = NeuralODE(dim=node_dim, hidden=hidden_dim, key=k5)  # 4D: (x, y, vx, vy)
    lstm = LSTMPredictor(dim=node_dim, hidden_size=hidden_dim, key=k6)

    print(f"  CHLU initialized (dim={chlu_dim}, hidden={hidden_dim})")
    print(f"  NeuralODE initialized (dim={node_dim}, hidden={hidden_dim})")
    print(f"  LSTM initialized (dim={node_dim}, hidden={hidden_dim})")

    # 3. Train each model on windowed sub-sequences
    print(f"\n[3/5] Training models ({train_epochs} epochs, window={window_size})...")

    print("  Training CHLU...")
    k6, k7 = jax.random.split(k6)
    chlu, chlu_losses = train_chlu(
        chlu,
        train_data,
        key=k7,
        config=config,
        window_size=window_size,
    )
    print(f"    Final loss: {chlu_losses[-1]:.6f}")

    print("  Training Neural ODE...")
    k7, k8 = jax.random.split(k7)
    node, node_losses = train_neural_ode(
        node,
        train_data,
        key=k8,
        config=config,
        window_size=window_size,
    )
    print(f"    Final loss: {node_losses[-1]:.6f}")

    print("  Training LSTM...")
    k8, k9 = jax.random.split(k8)
    lstm, lstm_losses = train_lstm(
        lstm,
        train_data,
        key=k9,
        config=config,
        window_size=window_size,
    )
    print(f"    Final loss: {lstm_losses[-1]:.6f}")

    # 4. Free run evaluation starting from last training point
    print(
        f"\n[4/5] Free run generation ({test_steps} steps from last training point)..."
    )

    # Initial conditions from LAST point of training data (for extrapolation)
    q_last, p_last = train_data[-1, :2], train_data[-1, 2:]
    z_last = train_data[-1]  # Full state for NODE/LSTM

    print(f"  Initial condition: q={q_last}, p={p_last}")

    print("  CHLU: Free run...")
    chlu_traj = chlu(q_last, p_last, steps=test_steps, dt=dt)

    print("  Neural ODE: Free run...")
    t_span = (0.0, test_steps * dt)
    node_traj = node(z_last, t_span, dt)

    print("  LSTM: Autoregressive generation...")
    lstm_traj = lstm.generate(z_last, steps=test_steps)

    # 5. Plot results
    print("\n[5/5] Creating visualization...")

    trajectories = {
        "LSTM": lstm_traj,
        "NODE": node_traj,
        "CHLU": chlu_traj,
    }

    titles = [
        "LSTM:",
        "NODE:",
        "CHLU:",
    ]

    # Original three-panel plot comparing against test ground truth
    save_path = os.path.join(save_dir, "exp1_stability.png")
    plot_three_panel_trajectories(
        trajectories, test_data, titles, save_path, steps_per_cycle=steps_per_cycle
    )

    # Evolution plot with transparent lines
    save_path_evolution = os.path.join(save_dir, "exp1_stability_evolution.png")
    plot_trajectory_evolution(
        trajectories,
        test_data,
        titles,
        save_path_evolution,
        n_snapshots=10,
        steps_per_cycle=steps_per_cycle,
    )

    # Animated GIF
    save_path_gif = os.path.join(save_dir, "exp1_stability_animation.gif")
    print("  Creating animation (this may take a moment)...")
    create_trajectory_animation(
        trajectories, test_data, titles, save_path_gif, fps=20, n_frames=100
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT A COMPLETE!")
    print("Results saved to:")
    print(f"  - {save_path}")
    print(f"  - {save_path_evolution}")
    print(f"  - {save_path_gif}")
    print("=" * 60 + "\n")
