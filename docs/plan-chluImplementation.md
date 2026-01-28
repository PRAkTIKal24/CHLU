# CHLU Implementation Plan (DRAFT v2)

## Summary of Changes from v1

| Area | Previous | Updated |
|------|----------|---------|
| Baselines | NeuralODE only | **NeuralODE + LSTM** |
| Exp A Dataset | Harmonic oscillator | **Figure-8 (Lemniscate)** |
| Exp A Protocol | Train T=10, test T=1000 | **Train T=100, test T=10,000** |
| Exp A Plot | 2 models comparison | **3-subplot grid (LSTM, NODE, CHLU)** |
| Exp B Dataset | Periodic trajectory | **Sine waves** |
| Exp B Models | CHLU vs NODE | **All 3: CHLU, NODE, LSTM** |
| Config | Custom | Hardcoded (no Hydra) |

---

# PART 1: Code Implementation Plan

## Overview

Build a JAX/Equinox-based research package implementing CHLU with **two baselines** (Neural ODE, LSTM) and three paper-ready experiments demonstrating the "Double Win."

**Agreed Parameters:**
- MLP: 2 hidden layers × 32 units, tanh activation
- LSTM: Hidden size 32, 1 layer
- PCA dim: 32 for MNIST
- Optimizer: Adam (lr=1e-3)
- Lyapunov λ: 0.01
- Timestep dt: 0.01
- Replay buffer: 1024 samples
- Friction γ: 0.01

---

## Phase 1: Project Structure & Core Modules

### Step 1.1: Directory Structure

```
chlu/
├── __init__.py          # Export public API
├── __main__.py          # CLI entry
├── chlu.py              # CLI (updated)
├── core/
│   ├── __init__.py
│   ├── chlu_unit.py     # CHLU class
│   ├── baselines.py     # NeuralODE + LSTM (combined file)
│   ├── regularization.py
│   ├── potentials.py
│   └── integrators.py
├── training/
│   ├── __init__.py
│   ├── train.py         # PCD training for CHLU
│   ├── train_baselines.py  # Standard training for NODE/LSTM
│   ├── replay_buffer.py
│   └── losses.py
├── data/
│   ├── __init__.py
│   ├── figure8.py       # Lemniscate of Bernoulli generator
│   ├── sine_waves.py    # Clean sine wave generator
│   └── mnist.py
├── experiments/
│   ├── __init__.py
│   ├── exp_a_stability.py   # Renamed from exp1
│   ├── exp_b_noise.py       # Renamed from exp2
│   └── exp_c_dreaming.py    # Renamed from exp3
└── utils/
    ├── __init__.py
    ├── plotting.py
    └── metrics.py

run_experiments.py       # Main runner
results/                 # Output directory
```

### Step 1.2: `chlu/core/potentials.py`

```python
class PotentialMLP(eqx.Module):
    """V(q) - learnable potential energy function"""
    layers: list
    
    def __init__(self, dim: int, hidden: int = 32, key: PRNGKey):
        # Linear(dim→32) → tanh → Linear(32→32) → tanh → Linear(32→1)
        
    def __call__(self, q: jnp.ndarray) -> float:
        # Returns scalar potential energy
```

### Step 1.3: `chlu/core/integrators.py`

```python
def velocity_verlet_step(H_fn, q, p, dt):
    """Symplectic integrator - preserves phase space volume"""
    # 1. p_half = p - 0.5 * dt * ∂H/∂q(q, p)
    # 2. q_next = q + dt * ∂H/∂p(q, p_half)
    # 3. p_next = p_half - 0.5 * dt * ∂H/∂q(q_next, p_half)
```

### Step 1.4: `chlu/core/chlu_unit.py`

```python
class CHLU(eqx.Module):
    potential_net: PotentialMLP
    log_mass: jnp.ndarray
    rest_mass: float = 1.0
    
    def H(self, q, p) -> float:
        """Relativistic Hamiltonian"""
        
    def step(self, state, dt) -> tuple:
        """Single Velocity Verlet step"""
        
    def __call__(self, q0, p0, steps, dt) -> jnp.ndarray:
        """Full trajectory via jax.lax.scan"""
```

### Step 1.5: `chlu/core/baselines.py` (NODE + LSTM combined)

```python
import equinox as eqx
import diffrax

# ============= Neural ODE Baseline =============
class NeuralODEDynamics(eqx.Module):
    """MLP dynamics: dz/dt = f(z, t)"""
    layers: list
    
    def __init__(self, dim: int, hidden: int = 32, key: PRNGKey):
        # Match CHLU parameter count
        # Linear(dim→32) → tanh → Linear(32→32) → tanh → Linear(32→dim)

    def __call__(self, t, z, args):
        # Return dz/dt (ignore t for autonomous system)

class NeuralODE(eqx.Module):
    """Neural ODE wrapper using Diffrax"""
    dynamics: NeuralODEDynamics
    
    def __init__(self, dim: int, hidden: int = 32, key: PRNGKey):
        self.dynamics = NeuralODEDynamics(dim, hidden, key)
    
    def __call__(self, z0: jnp.ndarray, t_span: tuple, dt: float) -> jnp.ndarray:
        """
        Solve ODE using diffrax.Tsit5 (Runge-Kutta)
        Returns: trajectory shape (T, dim)
        """
        term = diffrax.ODETerm(self.dynamics)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=jnp.arange(t_span[0], t_span[1], dt))
        solution = diffrax.diffeqsolve(term, solver, t0=t_span[0], t1=t_span[1],
                                        dt0=dt, y0=z0, saveat=saveat)
        return solution.ys

# ============= LSTM Baseline =============
class LSTMPredictor(eqx.Module):
    """LSTM for next-step prediction: x_t → x_{t+1}"""
    cell: eqx.nn.LSTMCell
    input_proj: eqx.nn.Linear   # Project input to hidden
    output_proj: eqx.nn.Linear  # Project hidden to output
    hidden_size: int = eqx.field(static=True)
    
    def __init__(self, dim: int, hidden_size: int = 32, key: PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(dim, hidden_size, key=k1)
        self.input_proj = eqx.nn.Linear(dim, dim, key=k2)  # Optional identity
        self.output_proj = eqx.nn.Linear(hidden_size, dim, key=k3)
    
    def __call__(self, x_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        Autoregressive prediction over sequence.
        Input: (T, dim) 
        Output: (T, dim) predictions for next timestep
        """
        def scan_fn(carry, x_t):
            h, c = carry
            h_new, c_new = self.cell((h, c), x_t)
            pred = self.output_proj(h_new)
            return (h_new, c_new), pred
        
        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        _, predictions = jax.lax.scan(scan_fn, init_state, x_sequence)
        return predictions
    
    def generate(self, x0: jnp.ndarray, steps: int) -> jnp.ndarray:
        """
        Autonomous generation starting from x0.
        Returns: (steps, dim) trajectory
        """
        def scan_fn(carry, _):
            h, c, x = carry
            h_new, c_new = self.cell((h, c), x)
            x_next = self.output_proj(h_new)
            return (h_new, c_new, x_next), x_next
        
        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size), x0)
        _, trajectory = jax.lax.scan(scan_fn, init_state, None, length=steps)
        return trajectory
```

**Key design:** LSTM has a `generate()` method for autonomous rollout (needed for Experiment A).

### Step 1.6: `chlu/core/regularization.py`

```python
def compute_lyapunov_loss(step_fn, trajectory: jnp.ndarray, n_samples: int = 10) -> float:
    """
    Penalize chaos by regularizing Lyapunov exponents.
    Loss = mean(log(singular_values(Jacobian)))
    """
```

---

## Phase 2: Training Infrastructure

### Step 2.1: `chlu/training/train.py` (CHLU-specific PCD)

```python
def train_chlu(model, data, epochs, lr, lyapunov_lambda, sleep_steps):
    """Wake-Sleep PCD training for CHLU"""
```

### Step 2.2: `chlu/training/train_baselines.py`

Separate training functions for NODE and LSTM:

```python
def train_neural_ode(model: NeuralODE, data: jnp.ndarray, 
                     epochs: int, lr: float = 1e-3) -> tuple:
    """
    Standard supervised training for Neural ODE.
    Loss: MSE between predicted and ground-truth trajectories.
    """
    optimizer = optax.adam(lr)
    
    @eqx.filter_jit
    def loss_fn(model, batch):
        # batch: (B, T, dim) trajectories
        # For each trajectory, predict from t=0 to t=T
        # Return MSE loss
        pass
    
    # Training loop
    # Return trained model, losses

def train_lstm(model: LSTMPredictor, data: jnp.ndarray,
               epochs: int, lr: float = 1e-3) -> tuple:
    """
    Supervised training for LSTM next-step prediction.
    Loss: MSE(x_{t+1}_pred, x_{t+1}_true)
    """
    optimizer = optax.adam(lr)
    
    @eqx.filter_jit
    def loss_fn(model, batch):
        # batch: (B, T, dim)
        # Input: x[:-1], Target: x[1:]
        # Return MSE between predictions and targets
        pass
    
    # Training loop
    # Return trained model, losses
```

### Step 2.3: Other training modules

- `replay_buffer.py` - Capacity 1024
- `losses.py` - MSE, energy loss

---

## Phase 3: Data Generators

### Step 3.1: `chlu/data/figure8.py`

```python
def generate_figure8(
    key: PRNGKey,
    steps: int,
    dt: float = 0.01,
    scale: float = 1.0,
) -> jnp.ndarray:
    """
    Generate Lemniscate of Bernoulli (Figure-8 curve).
    
    Parametric equations:
        x(t) = scale * sin(t) / (1 + cos²(t))
        y(t) = scale * sin(t) * cos(t) / (1 + cos²(t))
    
    Also compute velocities (dx/dt, dy/dt) for momentum.
    
    Returns: (steps, 4) array [x, y, vx, vy] ≡ [q, p]
    """
    t = jnp.linspace(0, 2 * jnp.pi * (steps * dt), steps)
    
    # Position
    denom = 1 + jnp.cos(t) ** 2
    x = scale * jnp.sin(t) / denom
    y = scale * jnp.sin(t) * jnp.cos(t) / denom
    
    # Velocity (analytical derivatives)
    # ... compute dx/dt, dy/dt
    
    return jnp.stack([x, y, vx, vy], axis=-1)
```

### Step 3.2: `chlu/data/sine_waves.py`

```python
def generate_sine_waves(
    key: PRNGKey,
    n_waves: int,
    steps: int,
    dt: float = 0.01,
    freq_range: tuple = (0.5, 2.0),
    amp_range: tuple = (0.5, 1.5),
) -> jnp.ndarray:
    """
    Generate clean sine wave trajectories with random freq/amp.
    
    Returns: (n_waves, steps, 2) array [x, dx/dt] for each wave
    """

def add_noise(data: jnp.ndarray, key: PRNGKey, sigma: float) -> jnp.ndarray:
    """Add Gaussian noise N(0, sigma) to data."""
    return data + sigma * jax.random.normal(key, data.shape)
```

### Step 3.3: `chlu/data/mnist.py`

```python
def load_mnist_pca(dim: int = 32) -> tuple:
    """Load MNIST, apply PCA to 32 dims"""
```

---

## Phase 4: Experiments

### Step 4.1: `chlu/experiments/exp_a_stability.py`

**Experiment A: "Eternal Memory" Stability Test**

```python
def run_experiment_a(save_dir: str = "results/"):
    """
    Goal: Compare infinite-horizon stability across all 3 models.
    
    Dataset: Figure-8 Orbit (Lemniscate of Bernoulli)
    Protocol:
        1. Generate Figure-8 ground truth (10,000 steps)
        2. Train all 3 models on first T=100 steps
        3. Run each model autonomously for T=10,000 steps (100x extrapolation)
        4. Plot 3 subplots showing trajectory behavior
    
    Plot: exp1_stability.png
        - Left: LSTM trajectory (expect drift/explosion → Chaos)
        - Middle: NODE trajectory (expect spiral inward → Dissipation)  
        - Right: CHLU trajectory (expect stable Figure-8 → Conservation)
    """
    key = jax.random.PRNGKey(42)
    
    # 1. Generate data
    full_trajectory = generate_figure8(key, steps=10000, dt=0.01)
    train_data = full_trajectory[:100]  # First 100 steps
    
    # 2. Initialize models
    k1, k2, k3 = jax.random.split(key, 3)
    chlu = CHLU(dim=2, key=k1)
    node = NeuralODE(dim=4, key=k2)  # 4D: (x, y, vx, vy)
    lstm = LSTMPredictor(dim=4, hidden_size=32, key=k3)
    
    # 3. Train each model on T=100 steps
    chlu, _ = train_chlu(chlu, train_data, epochs=500)
    node, _ = train_neural_ode(node, train_data, epochs=500)
    lstm, _ = train_lstm(lstm, train_data, epochs=500)
    
    # 4. Autonomous generation for T=10,000 steps
    q0, p0 = train_data[0, :2], train_data[0, 2:]
    
    chlu_traj = chlu(q0, p0, steps=10000, dt=0.01)
    node_traj = node(train_data[0], t_span=(0, 100), dt=0.01)
    lstm_traj = lstm.generate(train_data[0], steps=10000)
    
    # 5. Plot 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth in gray on all plots
    for ax in axes:
        ax.plot(full_trajectory[:, 0], full_trajectory[:, 1], 
                'gray', alpha=0.3, label='Ground Truth')
    
    axes[0].plot(lstm_traj[:, 0], lstm_traj[:, 1], 'r-', label='LSTM')
    axes[0].set_title('LSTM: Drift/Explosion (Chaos)')
    
    axes[1].plot(node_traj[:, 0], node_traj[:, 1], 'orange', label='NODE')
    axes[1].set_title('NODE: Spiral Inward (Dissipation)')
    
    axes[2].plot(chlu_traj[:, 0], chlu_traj[:, 1], 'g-', label='CHLU')
    axes[2].set_title('CHLU: Stable Figure-8 (Conservation)')
    
    plt.savefig(f"{save_dir}/exp1_stability.png", dpi=150)
```

### Step 4.2: `chlu/experiments/exp_b_noise.py`

**Experiment B: Energy-Based Noise Rejection**

```python
def run_experiment_b(save_dir: str = "results/"):
    """
    Goal: Test the "Noise Filter" hypothesis.
    
    Dataset: Clean Sine Waves
    Protocol:
        1. Train all 3 models on clean sine waves
        2. Test on inputs with Gaussian noise σ ∈ [0.1, 1.0]
        3. Measure reconstruction MSE for each noise level
    
    Hypothesis: 
        - CHLU treats noise as "high energy" → slides down potential well
        - LSTM/NODE try to fit the noise → error grows with σ
    
    Plot: exp2_noise_curve.png
        X-axis: Noise Sigma (σ)
        Y-axis: Reconstruction MSE
        Lines: LSTM (steep rise), NODE (moderate rise), CHLU (flat/robust)
    """
    key = jax.random.PRNGKey(42)
    
    # 1. Generate clean sine waves
    clean_data = generate_sine_waves(key, n_waves=100, steps=200)
    train_data, test_data = clean_data[:80], clean_data[80:]
    
    # 2. Train all models on clean data
    k1, k2, k3 = jax.random.split(key, 3)
    chlu = CHLU(dim=1, key=k1)  # 1D sine wave
    node = NeuralODE(dim=2, key=k2)  # (x, dx/dt)
    lstm = LSTMPredictor(dim=2, hidden_size=32, key=k3)
    
    chlu, _ = train_chlu(chlu, train_data, epochs=500)
    node, _ = train_neural_ode(node, train_data, epochs=500)
    lstm, _ = train_lstm(lstm, train_data, epochs=500)
    
    # 3. Test across noise levels
    sigmas = jnp.linspace(0.1, 1.0, 10)
    mse_chlu, mse_node, mse_lstm = [], [], []
    
    for sigma in sigmas:
        noisy_test = add_noise(test_data, key, sigma)
        
        # Reconstruct from noisy input
        chlu_pred = ...  # Run CHLU on noisy initial conditions
        node_pred = ...
        lstm_pred = ...
        
        # Compute MSE against CLEAN ground truth
        mse_chlu.append(compute_mse(chlu_pred, test_data))
        mse_node.append(compute_mse(node_pred, test_data))
        mse_lstm.append(compute_mse(lstm_pred, test_data))
    
    # 4. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(sigmas, mse_lstm, 'r-o', label='LSTM', linewidth=2)
    plt.plot(sigmas, mse_node, 'orange', marker='s', label='NODE', linewidth=2)
    plt.plot(sigmas, mse_chlu, 'g-^', label='CHLU', linewidth=2)
    plt.xlabel('Noise Sigma (σ)')
    plt.ylabel('Reconstruction MSE')
    plt.title('Noise Robustness: The Filter Effect')
    plt.legend()
    plt.savefig(f"{save_dir}/exp2_noise_curve.png", dpi=150)
```

### Step 4.3: `chlu/experiments/exp_c_dreaming.py`

**Experiment C: Generative "Dreaming" (MNIST)**

```python
def run_experiment_c(save_dir: str = "results/"):
    """
    Goal: Validate generative capability of PCD loop.
    
    Protocol:
        1. Load MNIST (PCA → 32 dims)
        2. Train CHLU with PCD (wake-sleep)
        3. Initialize random noise (q, p)
        4. Run dissipative dynamics with friction γ=0.01
        5. Show evolution: Noise → Hazy → Crisp Digit
    
    Plot: exp3_dreaming.png (grid of evolving images)
    """
```

### Step 4.4: `run_experiments.py`

```python
def main():
    key = jax.random.PRNGKey(42)
    os.makedirs("results", exist_ok=True)
    
    print("=" * 50)
    print("CHLU Experiments for ICLR Workshop Paper")
    print("=" * 50)
    
    print("\n[Experiment A] Eternal Memory Stability Test...")
    print("  Dataset: Figure-8 Orbit | Models: CHLU, NODE, LSTM")
    run_experiment_a()
    
    print("\n[Experiment B] Energy-Based Noise Rejection...")
    print("  Dataset: Sine Waves | Models: CHLU, NODE, LSTM")
    run_experiment_b()
    
    print("\n[Experiment C] Generative Dreaming...")
    print("  Dataset: MNIST | Model: CHLU only")
    run_experiment_c()
    
    print("\n" + "=" * 50)
    print("All experiments complete!")
    print(f"Results saved to: results/")
    print("  - exp1_stability.png")
    print("  - exp2_noise_curve.png")
    print("  - exp3_dreaming.png")
```

---

## Phase 5: Utilities

### Step 5.1: `chlu/utils/plotting.py`

```python
def plot_three_panel_trajectories(
    trajectories: dict,  # {"LSTM": arr, "NODE": arr, "CHLU": arr}
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
):
    """For Experiment A: 3-subplot figure"""

def plot_noise_curves(
    sigmas: jnp.ndarray,
    mse_dict: dict,  # {"LSTM": [...], "NODE": [...], "CHLU": [...]}
    save_path: str,
):
    """For Experiment B: noise robustness curves"""

def plot_dreaming_grid(images: jnp.ndarray, save_path: str):
    """For Experiment C: evolution grid"""
```

---

## Implementation Order

| Step | File | Dependencies | Est. LOC |
|------|------|--------------|----------|
| 1.2 | `core/potentials.py` | — | ~30 |
| 1.3 | `core/integrators.py` | — | ~40 |
| 1.4 | `core/chlu_unit.py` | 1.2, 1.3 | ~80 |
| 1.5 | `core/baselines.py` | — | ~120 |
| 1.6 | `core/regularization.py` | — | ~30 |
| 2.1 | `training/replay_buffer.py` | — | ~40 |
| 2.2 | `training/losses.py` | — | ~20 |
| 2.3 | `training/train.py` | 2.1, 2.2, 1.6 | ~100 |
| 2.4 | `training/train_baselines.py` | 1.5 | ~80 |
| 3.1 | `data/figure8.py` | — | ~40 |
| 3.2 | `data/sine_waves.py` | — | ~40 |
| 3.3 | `data/mnist.py` | — | ~40 |
| 4.1 | `experiments/exp_a_stability.py` | 1.4, 1.5, 3.1, 2.3, 2.4 | ~100 |
| 4.2 | `experiments/exp_b_noise.py` | 1.4, 1.5, 3.2, 2.3, 2.4 | ~90 |
| 4.3 | `experiments/exp_c_dreaming.py` | 1.4, 2.3, 3.3 | ~100 |
| 5.1 | `utils/plotting.py` | — | ~80 |

**Total estimated:** ~1030 lines of code

---

# PART 2: Validation & Testing Plan

## Test Categories

### Category 1: Unit Tests (`tests/test_core.py`)

**1.1 Symplecticity Test** (Critical - proves physics correctness)

```python
def test_symplectic_determinant():
    """Verify det(Jacobian) = 1.0 for CHLU step"""
    model = CHLU(dim=2, key=jax.random.PRNGKey(0))
    state = (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    
    def step_wrapper(flat_state):
        q, p = flat_state[:2], flat_state[2:]
        q_new, p_new = model.step((q, p), dt=0.01)
        return jnp.concatenate([q_new, p_new])
    
    J = jax.jacfwd(step_wrapper)(jnp.concatenate([state[0], state[1]]))
    det = jnp.linalg.det(J)
    assert jnp.isclose(det, 1.0, atol=1e-5), f"Symplecticity violated: det={det}"
```

**1.2 Energy Conservation Test**

```python
def test_energy_conservation():
    """Energy should be approximately conserved over trajectory"""
    model = CHLU(dim=2, key=jax.random.PRNGKey(0))
    q0, p0 = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    
    trajectory = model(q0, p0, steps=1000, dt=0.01)
    
    energies = [model.H(t[:2], t[2:]) for t in trajectory]
    energy_drift = jnp.abs(energies[-1] - energies[0]) / energies[0]
    
    assert energy_drift < 0.01, f"Energy drift {energy_drift:.2%} exceeds 1%"
```

**1.3 Gradient Flow Test**

```python
def test_gradients_flow():
    """Verify gradients flow through CHLU"""
    model = CHLU(dim=2, key=jax.random.PRNGKey(0))
    q0, p0 = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    
    def loss_fn(model):
        traj = model(q0, p0, steps=10, dt=0.01)
        return jnp.sum(traj ** 2)
    
    grads = eqx.filter_grad(loss_fn)(model)
    # Check that at least potential_net has non-zero gradients
    assert any(jnp.any(g != 0) for g in jax.tree_util.tree_leaves(grads))
```

**1.4 Mass Matrix Positivity**

```python
def test_mass_positive_definite():
    """Mass matrix M must always be positive"""
    model = CHLU(dim=4, key=jax.random.PRNGKey(0))
    M = jax.nn.softplus(model.log_mass)
    assert jnp.all(M > 0), "Mass matrix has non-positive elements"
```

**1.5 LSTM Forward Pass**

```python
def test_lstm_forward():
    """LSTM produces valid predictions"""
    model = LSTMPredictor(dim=4, hidden_size=32, key=jax.random.PRNGKey(0))
    x_seq = jnp.ones((10, 4))  # 10 timesteps, 4D
    
    preds = model(x_seq)
    
    assert preds.shape == (10, 4)
    assert jnp.all(jnp.isfinite(preds))
```

**1.6 LSTM Autonomous Generation**

```python
def test_lstm_generate():
    """LSTM generate() produces trajectory without errors"""
    model = LSTMPredictor(dim=4, hidden_size=32, key=jax.random.PRNGKey(0))
    x0 = jnp.array([1.0, 0.0, 0.0, 1.0])
    
    traj = model.generate(x0, steps=100)
    
    assert traj.shape == (100, 4)
    assert jnp.all(jnp.isfinite(traj))
```

**1.7 Parameter Count Matching**

```python
def test_parameter_counts_comparable():
    """All 3 models should have similar parameter counts"""
    chlu = CHLU(dim=2, key=jax.random.PRNGKey(0))
    node = NeuralODE(dim=4, key=jax.random.PRNGKey(1))
    lstm = LSTMPredictor(dim=4, hidden_size=32, key=jax.random.PRNGKey(2))
    
    counts = {
        "CHLU": count_params(chlu),
        "NODE": count_params(node),
        "LSTM": count_params(lstm),
    }
    
    max_c, min_c = max(counts.values()), min(counts.values())
    ratio = max_c / min_c
    
    assert ratio < 2.0, f"Param imbalance: {counts}"
```

### Category 2: Data Generator Tests (`tests/test_data.py`)

**2.1 Figure-8 Data Shape**

```python
def test_figure8_shape():
    """Figure-8 generator produces correct shape"""
    data = generate_figure8(jax.random.PRNGKey(0), steps=1000)
    
    assert data.shape == (1000, 4)  # (x, y, vx, vy)
```

**2.2 Figure-8 Is Closed Loop**

```python
def test_figure8_periodic():
    """Figure-8 should be approximately periodic"""
    data = generate_figure8(jax.random.PRNGKey(0), steps=1000, dt=0.01)
    
    # After one full period, should return close to start
    period_steps = int(2 * jnp.pi / 0.01)
    start = data[0]
    end = data[period_steps % len(data)]
    
    assert jnp.allclose(start, end, atol=0.1)
```

**2.3 Sine Wave Shape**

```python
def test_sine_waves_shape():
    """Sine wave generator produces correct shape"""
    data = generate_sine_waves(jax.random.PRNGKey(0), n_waves=10, steps=100)
    
    assert data.shape == (10, 100, 2)  # (batch, time, [x, dx/dt])
```

**2.4 MNIST Loading**

```python
def test_mnist_pca_loading():
    """MNIST loads and PCA reduces correctly"""
    train, test, pca = load_mnist_pca(dim=32)
    
    assert train.shape[1] == 32
    assert test.shape[1] == 32
    assert pca.n_components_ == 32
```

### Category 3: Training Tests (`tests/test_training.py`)

**3.1 CHLU Training Loop Smoke Test**

```python
def test_chlu_training_runs():
    """CHLU training loop completes without errors"""
    from chlu.training.train import train_chlu
    from chlu.data.figure8 import generate_figure8
    
    data = generate_figure8(jax.random.PRNGKey(0), steps=100)
    model = CHLU(dim=2, key=jax.random.PRNGKey(1))
    
    # Run 5 epochs without crashing
    trained, losses = train_chlu(model, data, epochs=5)
    
    assert len(losses) == 5
    assert all(jnp.isfinite(losses))
```

**3.2 CHLU Loss Decreases**

```python
def test_chlu_loss_decreases():
    """CHLU loss should decrease over training"""
    # ... same setup ...
    trained, losses = train_chlu(model, data, epochs=50)
    
    assert losses[-1] < losses[0], "Loss did not decrease"
```

**3.3 LSTM Training Runs**

```python
def test_lstm_training_runs():
    """LSTM training completes without errors"""
    lstm = LSTMPredictor(dim=4, hidden_size=32, key=jax.random.PRNGKey(0))
    data = jnp.ones((10, 50, 4))  # 10 sequences, 50 timesteps
    
    trained, losses = train_lstm(lstm, data, epochs=5)
    
    assert len(losses) == 5
    assert all(jnp.isfinite(losses))
```

**3.4 NODE Training Runs**

```python
def test_node_training_runs():
    """NODE training completes without errors"""
    node = NeuralODE(dim=4, key=jax.random.PRNGKey(0))
    data = generate_figure8(jax.random.PRNGKey(1), steps=100)
    
    trained, losses = train_neural_ode(node, data[None, :, :], epochs=5)
    
    assert len(losses) == 5
```

**3.5 Replay Buffer Operations**

```python
def test_replay_buffer_operations():
    """Buffer sample/update works correctly"""
    buffer = ReplayBuffer(capacity=100, dim=2)
    
    # Initial buffer should be zeros
    sample = buffer.sample(jax.random.PRNGKey(0), batch_size=10)
    assert sample.shape == (10, 2, 2)  # (batch, [q,p], dim)
    
    # Update and verify
    new_states = jnp.ones((10, 2, 2))
    buffer.update(new_states, jnp.arange(10))
```

### Category 4: Experiment Validation Tests (`tests/test_experiments.py`)

**4.1 Experiment A Outputs**

```python
def test_exp_a_produces_three_panel_plot():
    """Experiment A creates 3-subplot figure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment_a(save_dir=tmpdir)
        
        plot_path = os.path.join(tmpdir, "exp1_stability.png")
        assert os.path.exists(plot_path)
```

**4.2 Experiment B Outputs**

```python
def test_exp_b_produces_plot():
    """Experiment B creates noise curve plot"""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment_b(save_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "exp2_noise_curve.png"))
```

**4.3 Experiment C Outputs**

```python
def test_exp_c_produces_plot():
    """Experiment C creates dreaming grid"""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment_c(save_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "exp3_dreaming.png"))
```

### Category 5: Numerical Stability Tests (`tests/test_numerical.py`)

**5.1 Long Trajectory Stability**

```python
def test_no_nan_long_trajectory():
    """CHLU doesn't produce NaN over long runs"""
    model = CHLU(dim=2, key=jax.random.PRNGKey(0))
    traj = model(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), 
                 steps=10000, dt=0.01)
    
    assert jnp.all(jnp.isfinite(traj)), "NaN/Inf in trajectory"
```

**5.2 Gradients Bounded**

```python
def test_gradients_bounded():
    """Gradients should not explode"""
    # Compute gradients on a few batches and verify no extreme values
    # Max gradient magnitude < 100
```

### Category 6: Reproducibility Tests (`tests/test_reproducibility.py`)

**6.1 Seed Determinism**

```python
def test_deterministic_with_seed():
    """Same seed produces identical results"""
    key = jax.random.PRNGKey(42)
    
    model1 = CHLU(dim=2, key=key)
    model2 = CHLU(dim=2, key=key)
    
    traj1 = model1(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), 10, 0.01)
    traj2 = model2(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), 10, 0.01)
    
    assert jnp.allclose(traj1, traj2)
```

---

## Validation Checklist

### Pre-Implementation Validation

- [ ] Dependencies install correctly (`pip install -e .`)
- [ ] JAX detects GPU/TPU (if available)
- [ ] Equinox version ≥ 0.11.0

### Per-Module Validation

| Module | Tests Pass | Manual Check |
|--------|------------|--------------|
| `potentials.py` | `test_gradients_flow` | Output is scalar |
| `integrators.py` | `test_symplectic_determinant` | det(J) = 1 |
| `chlu_unit.py` | All symplectic/energy tests | Energy drift < 1% |
| `baselines.py` | `test_lstm_forward`, `test_neural_ode_forward` | Both produce finite output |
| `train_baselines.py` | `test_lstm_training_runs`, `test_node_training_runs` | Loss decreases |
| `figure8.py` | `test_figure8_shape`, `test_figure8_periodic` | Visual: looks like ∞ |
| `sine_waves.py` | `test_sine_waves_shape` | Visual: clean oscillations |

### Experiment Validation

| Experiment | Models | Expected Visual | Validation |
|------------|--------|-----------------|------------|
| Exp A | CHLU, NODE, LSTM | 3 panels: chaos/dissipation/conservation | Visual inspection of trajectories |
| Exp B | CHLU, NODE, LSTM | 3 curves: steep/moderate/flat | CHLU curve flattest |
| Exp C | CHLU only | Noise → Digit evolution | Recognizable digits emerge |

### Final Acceptance Criteria

1. **All tests pass:** `pytest tests/ -v`
2. **Coverage > 80%:** `pytest --cov=chlu --cov-report=html`
3. **Experiments reproduce:** Run `run_experiments.py` 3× with seed 42, verify identical plots
4. **Paper-ready plots:** Visual quality sufficient for ICLR workshop submission

---

## Verification Commands

```bash
# Run all tests
pytest tests/ -v

# Run only symplecticity test (most critical)
pytest tests/test_core.py::test_symplectic_determinant -v

# Run baseline-specific tests
pytest tests/test_baselines.py -v

# Run data generator tests
pytest tests/test_data.py -v

# Full coverage
pytest --cov=chlu --cov-report=html tests/

# Run experiments
python run_experiments.py

# Quick smoke test (reduced epochs)
python run_experiments.py --quick
```
