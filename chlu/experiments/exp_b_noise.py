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
from chlu.utils.checkpoints import load_checkpoint, save_checkpoint
from chlu.utils.metrics import compute_mse
from chlu.utils.plotting import (
    plot_noise_curves,
    plot_phase_space,
    plot_sine_wave_comparison,
    plot_multi_noise_grid,
    plot_noise_heatmap,
)


def run_experiment_b(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    use_pretrained: Optional[bool] = None,
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
        save_dir: Directory to save plots (overrides config)
        models_dir: Directory to save trained models (defaults to save_dir/../models)
        use_pretrained: Load pre-trained models if available (overrides config)
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
    if use_pretrained is not None:
        config.experiment_b.use_pretrained = use_pretrained

    # Extract values from config
    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "../models")
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
    use_pretrained = config.experiment_b.use_pretrained
    kinetic_mode = config.experiment_b.kinetic_energy_mode
    use_governor = config.experiment_b.use_governor
    governor_sensitivity = config.experiment_b.governor_sensitivity

    print("\n" + "=" * 60)
    print("EXPERIMENT B: Energy-Based Noise Rejection")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
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

    # 2. Initialize models
    print(f"  CHLU kinetic mode: {kinetic_mode}")
    k2, k3, k4, k5 = jax.random.split(k2, 4)

    chlu = CHLU(dim=chlu_dim, hidden=hidden_dim, rest_mass=config.model.rest_mass, c=config.model.speed_of_causality, kinetic_mode=kinetic_mode, key=k3)
    node = NeuralODE(dim=node_dim, hidden=hidden_dim, key=k4)
    lstm = LSTMPredictor(dim=node_dim, hidden_size=hidden_dim, key=k5)

    # Train or load models
    chlu_path = os.path.join(models_dir, "exp_b_chlu.pkl")
    node_path = os.path.join(models_dir, "exp_b_node.pkl")
    lstm_path = os.path.join(models_dir, "exp_b_lstm.pkl")

    models_exist = (
        os.path.exists(chlu_path)
        and os.path.exists(node_path)
        and os.path.exists(lstm_path)
    )

    if use_pretrained and models_exist:
        print(f"\n[2/4] Loading pre-trained models from {models_dir}...")
        chlu, chlu_metadata = load_checkpoint(chlu_path, chlu)
        target_energy = chlu_metadata.get("target_energy")
        node, _ = load_checkpoint(node_path, node)
        lstm, _ = load_checkpoint(lstm_path, lstm)
        print("  ✓ Models loaded successfully")
        if target_energy is not None:
            print(f"  ✓ Target energy: {target_energy:.4f}")
    else:
        if use_pretrained and not models_exist:
            print("\n[2/4] Pre-trained models not found, training from scratch...")
        else:
            print(f"\n[2/4] Training models on clean data ({train_epochs} epochs)...")

        # CHLU (1D sine wave, but we track [x, dx/dt] so dim=1 for position)
        print("  Training CHLU...")
        chlu, _, target_energy = train_chlu(chlu, train_data, key=k3, config=config)
        print(f"    Computed target energy: {target_energy:.4f}")

        # Neural ODE (2D: [x, dx/dt])
        print("  Training Neural ODE...")
        node, _ = train_neural_ode(node, train_data, key=k4, config=config)

        # LSTM
        print("  Training LSTM...")
        lstm, lstm_losses = train_lstm(lstm, train_data, key=k5, config=config)

        # Save trained models
        print("\n  Saving trained models...")
        save_checkpoint(chlu, chlu_path, epoch=train_epochs, loss=0.0, config=config, target_energy=target_energy)
        save_checkpoint(node, node_path, epoch=train_epochs, loss=0.0, config=config)
        save_checkpoint(
            lstm,
            lstm_path,
            epoch=train_epochs,
            loss=float(lstm_losses[-1]),
            config=config,
        )
        print(f"    Saved to {models_dir}")

    # 3. Test across noise levels
    print(f"\n[3/4] Testing noise robustness ({n_sigma} noise levels)...")
    if use_governor and target_energy is not None:
        print(f"  Using Active Governor (target_energy={target_energy:.4f}, sensitivity={governor_sensitivity})")
    else:
        print(f"  Using Fixed Friction Annealing (friction={sleep_friction}, ramp={friction_ramp})")

    sigmas = jnp.linspace(sigma_min, sigma_max, n_sigma)
    
    # Store temporal errors for heatmap (per-timestep errors)
    temporal_errors = {
        'LSTM': [],
        'NODE': [],
        'CHLU': []
    }
    mse_chlu, mse_node, mse_lstm = [], [], []

    # Store predictions for visualization (at low, middle, and high noise levels)
    # Choose indices for low (25%), mid (50%), and high (75%) noise levels
    low_sigma_idx = n_sigma // 4
    mid_sigma_idx = n_sigma // 2
    high_sigma_idx = (3 * n_sigma) // 4
    mid_sigma = sigmas[mid_sigma_idx]
    
    # Storage for multi-level grid plot
    multi_level_data = {
        'sigmas': [],
        'noisy_inputs': [],
        'predictions': {'LSTM': [], 'NODE': [], 'CHLU': []}
    }
    
    # Storage for existing single-level plots (middle noise)
    stored_predictions = None
    stored_noisy_data = None

    for i, sigma in enumerate(sigmas):
        print(f"  Noise σ = {sigma:.2f} ({i + 1}/{n_sigma})")

        # Add noise to test data
        k5, k6 = jax.random.split(k5)
        noisy_test = add_noise(test_data, k6, sigma)

        # Store data for visualization at middle noise level (existing plots)
        should_store_mid = i == mid_sigma_idx
        if should_store_mid:
            stored_noisy_data = noisy_test
            stored_predictions = {"LSTM": [], "NODE": [], "CHLU": []}
        
        # Store data for multi-level grid plot (low, mid, high)
        should_store_multi = i in [low_sigma_idx, mid_sigma_idx, high_sigma_idx]
        if should_store_multi:
            multi_level_data['sigmas'].append(float(sigma))
            multi_level_data['noisy_inputs'].append(noisy_test)
            # Initialize prediction lists for this noise level
            for model_name in ['LSTM', 'NODE', 'CHLU']:
                multi_level_data['predictions'][model_name].append([])

        # Evaluate each model
        # For CHLU: run dynamics from noisy initial conditions
        errors_chlu = []
        temporal_errors_chlu = []  # Per-timestep errors for this noise level
        for j in range(len(noisy_test)):
            # Extract clean and noisy initial conditions
            clean_seq = test_data[j]
            noisy_seq = noisy_test[j]

            # CHLU: position and momentum from [x, dx/dt]
            q0_noisy = noisy_seq[0, 0:1]  # x (1D)
            p0_noisy = noisy_seq[0, 1:2]  # dx/dt (1D)

            # Choose inference strategy
            if use_governor and target_energy is not None:
                # Active Governor: Dynamic energy-based friction control
                pred_traj = chlu.governed_rollout(
                    q0_noisy,
                    p0_noisy,
                    steps=steps,
                    dt=dt,
                    target_energy=target_energy,
                    sensitivity=governor_sensitivity,
                )
            else:
                # Legacy: Fixed friction annealing (ramp + coast)
                ramp_steps = int(steps * friction_ramp)
                coasting_steps = steps - ramp_steps

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
            
            # Compute per-timestep squared errors
            timestep_errors = jnp.mean((pred_traj - clean_seq) ** 2, axis=1)
            temporal_errors_chlu.append(timestep_errors)

            if should_store_mid:
                stored_predictions["CHLU"].append(pred_traj)
            
            if should_store_multi:
                # Get the index in multi_level lists (0, 1, or 2)
                multi_idx = [low_sigma_idx, mid_sigma_idx, high_sigma_idx].index(i)
                multi_level_data['predictions']['CHLU'][multi_idx].append(pred_traj)

        # Neural ODE
        errors_node = []
        temporal_errors_node = []
        for j in range(len(noisy_test)):
            z0_noisy = noisy_test[j, 0]
            pred_traj = node(z0_noisy, (0.0, steps * dt), dt)
            errors_node.append(compute_mse(pred_traj, test_data[j]))
            
            # Compute per-timestep squared errors
            timestep_errors = jnp.mean((pred_traj - test_data[j]) ** 2, axis=1)
            temporal_errors_node.append(timestep_errors)

            if should_store_mid:
                stored_predictions["NODE"].append(pred_traj)
            
            if should_store_multi:
                multi_idx = [low_sigma_idx, mid_sigma_idx, high_sigma_idx].index(i)
                multi_level_data['predictions']['NODE'][multi_idx].append(pred_traj)

        # LSTM
        errors_lstm = []
        temporal_errors_lstm = []
        for j in range(len(noisy_test)):
            x0_noisy = noisy_test[j, 0]
            pred_traj = lstm.generate(x0_noisy, steps=steps)
            errors_lstm.append(compute_mse(pred_traj, test_data[j]))
            
            # Compute per-timestep squared errors
            timestep_errors = jnp.mean((pred_traj - test_data[j]) ** 2, axis=1)
            temporal_errors_lstm.append(timestep_errors)

            if should_store_mid:
                stored_predictions["LSTM"].append(pred_traj)
            
            if should_store_multi:
                multi_idx = [low_sigma_idx, mid_sigma_idx, high_sigma_idx].index(i)
                multi_level_data['predictions']['LSTM'][multi_idx].append(pred_traj)

        mse_chlu.append(jnp.mean(jnp.array(errors_chlu)))
        mse_node.append(jnp.mean(jnp.array(errors_node)))
        mse_lstm.append(jnp.mean(jnp.array(errors_lstm)))
        
        # Store mean temporal errors across all test samples for this noise level
        temporal_errors['CHLU'].append(jnp.mean(jnp.array(temporal_errors_chlu), axis=0))
        temporal_errors['NODE'].append(jnp.mean(jnp.array(temporal_errors_node), axis=0))
        temporal_errors['LSTM'].append(jnp.mean(jnp.array(temporal_errors_lstm), axis=0))

    # 4. Plot results
    print("\n[4/4] Creating visualizations...")

    mse_dict = {
        "LSTM": jnp.array(mse_lstm),
        "NODE": jnp.array(mse_node),
        "CHLU": jnp.array(mse_chlu),
    }
    
    # Multi-level noise grid (NEW)
    if len(multi_level_data['sigmas']) == 3:
        # Convert prediction lists to arrays
        for model_name in ['LSTM', 'NODE', 'CHLU']:
            for idx in range(3):
                multi_level_data['predictions'][model_name][idx] = jnp.array(
                    multi_level_data['predictions'][model_name][idx]
                )
        
        save_path_grid = os.path.join(save_dir, "exp2_multi_noise_grid.png")
        plot_multi_noise_grid(test_data, multi_level_data, save_path_grid, example_idx=0)
        print(f"  Saved multi-noise grid: {save_path_grid}")

    # Noise curve
    save_path = os.path.join(save_dir, "exp2_noise_curve.png")
    plot_noise_curves(sigmas, mse_dict, save_path)
    
    # Noise heatmap
    # Convert temporal errors to arrays
    for model_name in ['LSTM', 'NODE', 'CHLU']:
        temporal_errors[model_name] = jnp.array(temporal_errors[model_name])
    
    save_path_heatmap = os.path.join(save_dir, "exp2_noise_heatmap.png")
    plot_noise_heatmap(sigmas, temporal_errors, save_path_heatmap)

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
    print("Results saved to:")
    print(f"  - {save_path}")
    print(f"  - {save_path_heatmap}")
    if len(multi_level_data['sigmas']) == 3:
        print(f"  - {save_path_grid}")
    if stored_predictions is not None:
        print(f"  - {save_path_waves}")
        print(f"  - {save_path_phase}")
    print("=" * 60 + "\n")
    if stored_predictions is not None:
        print(f"  - {save_path_waves}")
        print(f"  - {save_path_phase}")
    print("=" * 60 + "\n")
