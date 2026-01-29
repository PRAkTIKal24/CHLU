"""Experiment B: Energy-Based Noise Rejection.

Tests the "noise filter" hypothesis across CHLU, NODE, and LSTM.
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp

from chlu.config import CHLUConfig, get_default_config
from chlu.core.baselines import LSTMPredictor, NeuralODE
from chlu.core.chlu_unit import CHLU
from chlu.data.sine_waves import add_noise, generate_sine_waves
from chlu.training.train import train_chlu
from chlu.training.train_baselines import train_lstm, train_neural_ode
from chlu.utils.metrics import compute_mse
from chlu.utils.plotting import (
    plot_noise_curves,
    plot_sine_wave_comparison,
    plot_phase_space,
)


def run_experiment_b(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    seed: Optional[int] = None,
    n_waves: Optional[int] = None,
    steps: Optional[int] = None,
    train_epochs: Optional[int] = None,
    dt: Optional[float] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    n_sigma: Optional[int] = None,
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
        config: CHLUConfig object (if None, uses defaults)
        save_dir: Directory to save results (overrides config)
        seed: Random seed (overrides config)
        n_waves: Number of sine waves to generate (overrides config)
        steps: Steps per wave (overrides config)
        train_epochs: Training epochs (overrides config)
        dt: Time step (overrides config)
        sigma_min: Minimum noise level (overrides config)
        sigma_max: Maximum noise level (overrides config)
        n_sigma: Number of noise levels to test (overrides config)
    """
    # Load config with overrides
    if config is None:
        config = get_default_config()

    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed
    if n_waves is not None:
        config.experiment_b.n_waves = n_waves
    if steps is not None:
        config.experiment_b.steps = steps
    if train_epochs is not None:
        config.experiment_b.train_epochs = train_epochs
    if dt is not None:
        config.experiment_b.dt = dt
    if sigma_min is not None:
        config.experiment_b.sigma_min = sigma_min
    if sigma_max is not None:
        config.experiment_b.sigma_max = sigma_max
    if n_sigma is not None:
        config.experiment_b.n_sigma = n_sigma

    # Extract values from config
    save_dir = config.project.save_dir or "results/"
    seed = config.project.seed
    n_waves = config.experiment_b.n_waves
    steps = config.experiment_b.steps
    train_epochs = config.experiment_b.train_epochs
    sleep_friction = config.experiment_b.sleep_friction
    friction_ramp = config.experiment_b.friction_ramp
    dt = config.experiment_b.dt
    sigma_min = config.experiment_b.sigma_min
    sigma_max = config.experiment_b.sigma_max
    n_sigma = config.experiment_b.n_sigma
    chlu_dim = config.experiment_b.chlu_dim
    node_dim = config.experiment_b.node_dim
    hidden_dim = config.experiment_b.hidden_dim

    print("\n" + "=" * 60)
    print("EXPERIMENT B: Energy-Based Noise Rejection")
    print("=" * 60)

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
    chlu = CHLU(dim=chlu_dim, hidden=hidden_dim, key=k3)
    chlu, _ = train_chlu(chlu, train_data, key=k3, config=config)

    # Neural ODE (2D: [x, dx/dt])
    print("  Training Neural ODE...")
    node = NeuralODE(dim=node_dim, hidden=hidden_dim, key=k4)
    node, _ = train_neural_ode(node, train_data, key=k4, config=config)

    # LSTM
    print("  Training LSTM...")
    lstm = LSTMPredictor(dim=node_dim, hidden_size=hidden_dim, key=k5)
    lstm, _ = train_lstm(lstm, train_data, key=k5, config=config)

    # 3. Test across noise levels
    print(f"\n[3/4] Testing noise robustness ({n_sigma} noise levels)...")

    sigmas = jnp.linspace(sigma_min, sigma_max, n_sigma)
    mse_chlu, mse_node, mse_lstm = [], [], []
    
    # Store predictions for visualization (at middle noise level)
    mid_sigma_idx = n_sigma // 2
    mid_sigma = sigmas[mid_sigma_idx]
    stored_predictions = None
    stored_noisy_data = None

    for i, sigma in enumerate(sigmas):
        print(f"  Noise σ = {sigma:.2f} ({i + 1}/{n_sigma})")

        # Add noise to test data
        k5, k6 = jax.random.split(k5)
        noisy_test = add_noise(test_data, k6, sigma)

        # Store data for visualization at middle noise level
        should_store = (i == mid_sigma_idx)
        if should_store:
            stored_noisy_data = noisy_test
            stored_predictions = {"LSTM": [], "NODE": [], "CHLU": []}

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

            # Friction Annealing
            ramp_steps = int(steps * friction_ramp)
            coasting_steps = steps - ramp_steps

            # Run CHLU dynamics
            # 1. Friction ramp-up phase
            pred_traj_ramp = chlu(
                q0_noisy,
                p0_noisy,
                steps=ramp_steps,
                dt=dt,
                gamma=sleep_friction,
            )

            q_ramped = pred_traj_ramp[-1, 0:1]
            p_ramped = pred_traj_ramp[-1, 1:2]

            # 2. Coasting phase (gamma=0)
            pred_traj_coast = chlu(
                q_ramped, p_ramped, steps=coasting_steps, dt=dt, gamma=0.0
            )

            pred_traj = jnp.concatenate([pred_traj_ramp, pred_traj_coast], axis=0)

            # Compare against clean data
            errors_chlu.append(compute_mse(pred_traj, clean_seq))
            
            if should_store:
                stored_predictions["CHLU"].append(pred_traj)

        # Neural ODE
        errors_node = []
        for j in range(len(noisy_test)):
            z0_noisy = noisy_test[j, 0]
            pred_traj = node(z0_noisy, (0.0, steps * dt), dt)
            errors_node.append(compute_mse(pred_traj, test_data[j]))
            
            if should_store:
                stored_predictions["NODE"].append(pred_traj)

        # LSTM
        errors_lstm = []
        for j in range(len(noisy_test)):
            x0_noisy = noisy_test[j, 0]
            pred_traj = lstm.generate(x0_noisy, steps=steps)
            errors_lstm.append(compute_mse(pred_traj, test_data[j]))
            
            if should_store:
                stored_predictions["LSTM"].append(pred_traj)

        mse_chlu.append(jnp.mean(jnp.array(errors_chlu)))
        mse_node.append(jnp.mean(jnp.array(errors_node)))
        mse_lstm.append(jnp.mean(jnp.array(errors_lstm)))

    # 4. Plot results
    print("\n[4/4] Creating visualizations...")

    mse_dict = {
        "LSTM": jnp.array(mse_lstm),
        "NODE": jnp.array(mse_node),
        "CHLU": jnp.array(mse_chlu),
    }

    # Noise curve
    save_path = os.path.join(save_dir, "exp2_noise_curve.png")
    plot_noise_curves(sigmas, mse_dict, save_path)
    
    # Sine wave comparison (using stored predictions from middle noise level)
    if stored_predictions is not None:
        # Convert lists to arrays
        for key in stored_predictions:
            stored_predictions[key] = jnp.array(stored_predictions[key])
        
        save_path_waves = os.path.join(save_dir, "exp2_sine_wave_comparison.png")
        plot_sine_wave_comparison(
            test_data,
            stored_noisy_data,
            stored_predictions,
            save_path_waves,
            n_examples=3,
            sigma=float(mid_sigma),
        )
        
        # Phase space plot
        save_path_phase = os.path.join(save_dir, "exp2_phase_space.png")
        plot_phase_space(
            test_data,
            stored_noisy_data,
            stored_predictions,
            save_path_phase,
            n_examples=3,
            sigma=float(mid_sigma),
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT B COMPLETE!")
    print(f"Results saved to:")
    print(f"  - {save_path}")
    if stored_predictions is not None:
        print(f"  - {save_path_waves}")
        print(f"  - {save_path_phase}")
    print("=" * 60 + "\n")
