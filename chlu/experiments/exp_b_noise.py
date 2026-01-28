"""Experiment B: Energy-Based Noise Rejection.

Tests the "noise filter" hypothesis across CHLU, NODE, and LSTM.
"""

import os
import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU
from chlu.core.baselines import NeuralODE, LSTMPredictor
from chlu.data.sine_waves import generate_sine_waves, add_noise
from chlu.training.train import train_chlu
from chlu.training.train_baselines import train_neural_ode, train_lstm
from chlu.utils.plotting import plot_noise_curves
from chlu.utils.metrics import compute_mse


def run_experiment_b(
    save_dir: str = "results/",
    seed: int = 42,
    n_waves: int = 100,
    steps: int = 200,
    train_epochs: int = 500,
    dt: float = 0.01,
    sigma_min: float = 0.1,
    sigma_max: float = 1.0,
    n_sigma: int = 10,
):
    """
    Experiment B: Energy-Based Noise Rejection.
    
    Protocol:
        1. Train all 3 models on clean sine waves
        2. Test on inputs with Gaussian noise σ ∈ [0.1, 1.0]
        3. Measure reconstruction MSE for each noise level
    
    Hypothesis:
        - CHLU treats noise as "high energy" → slides down potential well
        - LSTM/NODE try to fit the noise → error grows with σ
    
    Expected outcomes:
        - LSTM: Steep error rise (tries to fit noise)
        - NODE: Moderate error rise
        - CHLU: Flat/robust error (filters noise)
    
    Args:
        save_dir: Directory to save results
        seed: Random seed
        n_waves: Number of sine waves to generate
        steps: Steps per wave
        train_epochs: Training epochs
        dt: Time step
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        n_sigma: Number of noise levels to test
    """
    print("\n" + "="*60)
    print("EXPERIMENT B: Energy-Based Noise Rejection")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    
    # 1. Generate clean sine waves
    print(f"\n[1/4] Generating {n_waves} sine waves ({steps} steps each)...")
    k1, k2 = jax.random.split(key)
    clean_data = generate_sine_waves(k1, n_waves=n_waves, steps=steps, dt=dt)
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * n_waves)
    train_data = clean_data[:split_idx]
    test_data = clean_data[split_idx:]
    
    print(f"  Train data: {train_data.shape}")
    print(f"  Test data: {test_data.shape}")
    
    # 2. Train all models on clean data
    print(f"\n[2/4] Training models on clean data ({train_epochs} epochs)...")
    
    k2, k3, k4, k5 = jax.random.split(k2, 4)
    
    # CHLU (1D sine wave, but we track [x, dx/dt] so dim=1 for position)
    print("  Training CHLU...")
    chlu = CHLU(dim=1, hidden=32, key=k3)
    chlu, _ = train_chlu(chlu, train_data, key=k3, epochs=train_epochs, dt=dt)
    
    # Neural ODE (2D: [x, dx/dt])
    print("  Training Neural ODE...")
    node = NeuralODE(dim=2, hidden=32, key=k4)
    node, _ = train_neural_ode(node, train_data, key=k4, epochs=train_epochs, dt=dt)
    
    # LSTM
    print("  Training LSTM...")
    lstm = LSTMPredictor(dim=2, hidden_size=32, key=k5)
    lstm, _ = train_lstm(lstm, train_data, key=k5, epochs=train_epochs)
    
    # 3. Test across noise levels
    print(f"\n[3/4] Testing noise robustness ({n_sigma} noise levels)...")
    
    sigmas = jnp.linspace(sigma_min, sigma_max, n_sigma)
    mse_chlu, mse_node, mse_lstm = [], [], []
    
    for i, sigma in enumerate(sigmas):
        print(f"  Noise σ = {sigma:.2f} ({i+1}/{n_sigma})")
        
        # Add noise to test data
        k5, k6 = jax.random.split(k5)
        noisy_test = add_noise(test_data, k6, sigma)
        
        # Evaluate each model
        # For CHLU: run dynamics from noisy initial conditions
        errors_chlu = []
        for j in range(len(noisy_test)):
            # Extract clean and noisy initial conditions
            clean_seq = test_data[j]
            noisy_seq = noisy_test[j]
            
            # CHLU: position and momentum from [x, dx/dt]
            q0_noisy = noisy_seq[0, 0:1]  # x (1D)
            p0_noisy = noisy_seq[0, 1:2]  # dx/dt (1D)
            
            # Run CHLU dynamics
            pred_traj = chlu(q0_noisy, p0_noisy, steps=steps, dt=dt)
            
            # Compare against clean data
            errors_chlu.append(compute_mse(pred_traj, clean_seq))
        
        # Neural ODE
        errors_node = []
        for j in range(len(noisy_test)):
            z0_noisy = noisy_test[j, 0]
            pred_traj = node(z0_noisy, (0.0, steps * dt), dt)
            errors_node.append(compute_mse(pred_traj, test_data[j]))
        
        # LSTM
        errors_lstm = []
        for j in range(len(noisy_test)):
            x0_noisy = noisy_test[j, 0]
            pred_traj = lstm.generate(x0_noisy, steps=steps)
            errors_lstm.append(compute_mse(pred_traj, test_data[j]))
        
        mse_chlu.append(jnp.mean(jnp.array(errors_chlu)))
        mse_node.append(jnp.mean(jnp.array(errors_node)))
        mse_lstm.append(jnp.mean(jnp.array(errors_lstm)))
    
    # 4. Plot results
    print("\n[4/4] Creating visualization...")
    
    mse_dict = {
        "LSTM": jnp.array(mse_lstm),
        "NODE": jnp.array(mse_node),
        "CHLU": jnp.array(mse_chlu),
    }
    
    save_path = os.path.join(save_dir, "exp2_noise_curve.png")
    plot_noise_curves(sigmas, mse_dict, save_path)
    
    print("\n" + "="*60)
    print("EXPERIMENT B COMPLETE!")
    print(f"Results saved to: {save_path}")
    print("="*60 + "\n")
