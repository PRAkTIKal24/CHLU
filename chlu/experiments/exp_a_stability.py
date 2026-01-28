"""Experiment A: Eternal Memory Stability Test.

Compares long-horizon stability across CHLU, NODE, and LSTM.
"""

import os
import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU
from chlu.core.baselines import NeuralODE, LSTMPredictor
from chlu.data.figure8 import generate_figure8
from chlu.training.train import train_chlu
from chlu.training.train_baselines import train_neural_ode, train_lstm
from chlu.utils.plotting import plot_three_panel_trajectories


def run_experiment_a(
    save_dir: str = "results/",
    seed: int = 42,
    train_steps: int = 100,
    test_steps: int = 10000,
    train_epochs: int = 500,
    dt: float = 0.01,
):
    """
    Experiment A: "Eternal Memory" Stability Test.
    
    Protocol:
        1. Generate Figure-8 ground truth (10,000 steps)
        2. Train all 3 models on first T=100 steps
        3. Run each model autonomously for T=10,000 steps (100x extrapolation)
        4. Plot 3 subplots showing trajectory behavior
    
    Expected outcomes:
        - LSTM: Drift/explosion (Chaos)
        - NODE: Spiral inward (Dissipation)
        - CHLU: Stable Figure-8 (Conservation)
    
    Args:
        save_dir: Directory to save results
        seed: Random seed for reproducibility
        train_steps: Number of steps to train on (default: 100)
        test_steps: Number of steps to extrapolate (default: 10,000)
        train_epochs: Training epochs per model (default: 500)
        dt: Time step size
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: Eternal Memory Stability Test")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    
    # 1. Generate Figure-8 data
    print(f"\n[1/5] Generating Figure-8 trajectory ({test_steps} steps)...")
    k1, k2 = jax.random.split(key)
    full_trajectory = generate_figure8(k1, steps=test_steps, dt=dt)
    train_data = full_trajectory[:train_steps]  # First 100 steps
    
    print(f"  Training data: {train_data.shape}")
    print(f"  Full trajectory: {full_trajectory.shape}")
    
    # 2. Initialize models
    print("\n[2/5] Initializing models (CHLU, NODE, LSTM)...")
    k2, k3, k4, k5 = jax.random.split(k2, 4)
    
    chlu = CHLU(dim=2, hidden=32, key=k3)
    node = NeuralODE(dim=4, hidden=32, key=k4)  # 4D: (x, y, vx, vy)
    lstm = LSTMPredictor(dim=4, hidden_size=32, key=k5)
    
    print(f"  CHLU initialized (dim=2)")
    print(f"  NeuralODE initialized (dim=4)")
    print(f"  LSTM initialized (dim=4, hidden=32)")
    
    # 3. Train each model on T=100 steps
    print(f"\n[3/5] Training models ({train_epochs} epochs)...")
    
    print("  Training CHLU...")
    k5, k6 = jax.random.split(k5)
    chlu, chlu_losses = train_chlu(
        chlu, 
        train_data, 
        key=k6,
        epochs=train_epochs,
        dt=dt,
    )
    print(f"    Final loss: {chlu_losses[-1]:.6f}")
    
    print("  Training Neural ODE...")
    k6, k7 = jax.random.split(k6)
    node, node_losses = train_neural_ode(
        node,
        train_data,
        key=k7,
        epochs=train_epochs,
        dt=dt,
    )
    print(f"    Final loss: {node_losses[-1]:.6f}")
    
    print("  Training LSTM...")
    k7, k8 = jax.random.split(k7)
    lstm, lstm_losses = train_lstm(
        lstm,
        train_data,
        key=k8,
        epochs=train_epochs,
    )
    print(f"    Final loss: {lstm_losses[-1]:.6f}")
    
    # 4. Autonomous generation for T=10,000 steps
    print(f"\n[4/5] Generating trajectories ({test_steps} steps)...")
    
    # Initial conditions from training data
    q0, p0 = train_data[0, :2], train_data[0, 2:]
    z0 = train_data[0]  # Full state for NODE/LSTM
    
    print("  CHLU: Generating...")
    chlu_traj = chlu(q0, p0, steps=test_steps, dt=dt)
    
    print("  Neural ODE: Generating...")
    t_span = (0.0, test_steps * dt)
    node_traj = node(z0, t_span, dt)
    
    print("  LSTM: Generating...")
    lstm_traj = lstm.generate(z0, steps=test_steps)
    
    # 5. Plot results
    print("\n[5/5] Creating visualization...")
    
    trajectories = {
        "LSTM": lstm_traj,
        "NODE": node_traj,
        "CHLU": chlu_traj,
    }
    
    titles = [
        "LSTM: Drift/Explosion (Chaos)",
        "NODE: Spiral Inward (Dissipation)",
        "CHLU: Stable Figure-8 (Conservation)",
    ]
    
    save_path = os.path.join(save_dir, "exp1_stability.png")
    plot_three_panel_trajectories(trajectories, full_trajectory, titles, save_path)
    
    print("\n" + "="*60)
    print("EXPERIMENT A COMPLETE!")
    print(f"Results saved to: {save_path}")
    print("="*60 + "\n")
