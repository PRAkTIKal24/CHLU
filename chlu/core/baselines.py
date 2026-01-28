"""Baseline models: Neural ODE and LSTM."""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax


# ============= Neural ODE Baseline =============

class NeuralODEDynamics(eqx.Module):
    """
    MLP dynamics for Neural ODE: dz/dt = f(z, t).
    
    Architecture matches CHLU parameter count for fair comparison.
    """
    
    layers: list
    
    def __init__(self, dim: int, hidden: int = 32, key: jax.random.PRNGKey = None):
        """
        Initialize dynamics network.
        
        Args:
            dim: State dimension
            hidden: Hidden units (default: 32)
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        keys = jax.random.split(key, 3)
        
        # Linear(dim→32) → tanh → Linear(32→32) → tanh → Linear(32→dim)
        self.layers = [
            eqx.nn.Linear(dim, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, dim, key=keys[2]),
        ]
    
    def __call__(self, t, z, args):
        """
        Compute dz/dt.
        
        Args:
            t: Time (ignored for autonomous system)
            z: State (dim,)
            args: Additional arguments (unused)
        
        Returns:
            dz/dt
        """
        x = z
        # First layer + activation
        x = self.layers[0](x)
        x = jnp.tanh(x)
        
        # Second layer + activation
        x = self.layers[1](x)
        x = jnp.tanh(x)
        
        # Output layer
        x = self.layers[2](x)
        
        return x


class NeuralODE(eqx.Module):
    """
    Neural ODE wrapper using Diffrax.
    
    Represents dissipative continuous-time systems.
    Uses Runge-Kutta integrator (Tsit5).
    """
    
    dynamics: NeuralODEDynamics
    
    def __init__(self, dim: int, hidden: int = 32, key: jax.random.PRNGKey = None):
        """
        Initialize Neural ODE.
        
        Args:
            dim: State dimension
            hidden: Hidden units (default: 32)
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        self.dynamics = NeuralODEDynamics(dim, hidden, key)
    
    def __call__(
        self, 
        z0: jnp.ndarray, 
        t_span: tuple, 
        dt: float
    ) -> jnp.ndarray:
        """
        Solve ODE using diffrax.Tsit5 (Runge-Kutta).
        
        Args:
            z0: Initial state (dim,)
            t_span: (t0, t1) time span
            dt: Time step for output
        
        Returns:
            Trajectory of shape (T, dim)
        """
        term = diffrax.ODETerm(self.dynamics)
        solver = diffrax.Tsit5()
        
        # Create time points for output
        ts = jnp.arange(t_span[0], t_span[1], dt)
        saveat = diffrax.SaveAt(ts=ts)
        
        # Solve ODE with sufficient max_steps for long trajectories
        # Use 10x the number of output points to be safe
        max_steps_needed = len(ts) * 10
        
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt,
            y0=z0,
            saveat=saveat,
            max_steps=max_steps_needed,
        )
        
        return solution.ys


# ============= LSTM Baseline =============

class LSTMPredictor(eqx.Module):
    """
    LSTM for next-step prediction: x_t → x_{t+1}.
    
    Represents discrete sequence models that lack conservation laws.
    """
    
    cell: eqx.nn.LSTMCell
    output_proj: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    
    def __init__(
        self, 
        dim: int, 
        hidden_size: int = 32, 
        key: jax.random.PRNGKey = None
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            dim: Input/output dimension
            hidden_size: LSTM hidden size (default: 32)
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        k1, k2 = jax.random.split(key, 2)
        
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(dim, hidden_size, key=k1)
        self.output_proj = eqx.nn.Linear(hidden_size, dim, key=k2)
    
    def __call__(self, x_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        Autoregressive prediction over sequence.
        
        Args:
            x_sequence: Input sequence (T, dim)
        
        Returns:
            Predictions for next timestep (T, dim)
        """
        def scan_fn(carry, x_t):
            h, c = carry
            (h_new, c_new) = self.cell(x_t, (h, c))
            pred = self.output_proj(h_new)
            return (h_new, c_new), pred
        
        # Initialize hidden state
        init_state = (
            jnp.zeros(self.hidden_size), 
            jnp.zeros(self.hidden_size)
        )
        
        _, predictions = jax.lax.scan(scan_fn, init_state, x_sequence)
        
        return predictions
    
    def generate(self, x0: jnp.ndarray, steps: int) -> jnp.ndarray:
        """
        Autonomous generation starting from x0.
        
        This is crucial for Experiment A where we need to extrapolate
        beyond the training horizon.
        
        Args:
            x0: Initial state (dim,)
            steps: Number of steps to generate
        
        Returns:
            Generated trajectory (steps, dim)
        """
        def scan_fn(carry, _):
            h, c, x = carry
            (h_new, c_new) = self.cell(x, (h, c))
            x_next = self.output_proj(h_new)
            return (h_new, c_new, x_next), x_next
        
        # Initialize
        init_state = (
            jnp.zeros(self.hidden_size),
            jnp.zeros(self.hidden_size),
            x0
        )
        
        _, trajectory = jax.lax.scan(scan_fn, init_state, None, length=steps)
        
        return trajectory
