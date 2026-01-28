Research Engineering Task: Causal Hamiltonian Learning Unit (CHLU)Role: You are a Principal Research Engineer assisting in the development of a novel AI primitive called the Causal Hamiltonian Learning Unit (CHLU).Goal: Create a reproducible JAX-based software package to run experiments for an ICLR Workshop paper (AI & PDE).Framework: JAX, Equinox (for modularity), Optax (optimizers), Diffrax (ODEs).1. Mathematical Primitives & TheoryThe CHLU is a dynamical system unit grounded in Symplectic Mechanics, designed to solve the memory-stability trade-off inherent in RNNs and Neural ODEs.The Physics (Hamiltonian Dynamics)Unlike standard RNNs, the state $z = (q, p)$ (Position, Momentum) evolves according to Hamilton's Equations, which strictly conserve phase space volume (Liouville's Theorem):$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = - \frac{\partial H}{\partial q}$$The Hamiltonian ($H$)We utilize a Relativistic Hamiltonian to ensure causal bounds (speed of light limit) and energy stability.$$H(q, p) = \underbrace{\sqrt{p^T M p + m^2}}_{\text{Relativistic Kinetic Energy}} + \underbrace{V(q)}_{\text{Potential Energy (MLP)}}$$$M$: A learnable, positive-definite mass matrix (can be diagonal for simplicity).$m$: Rest mass constant (hyperparameter, e.g., 1.0).$V(q)$: A learnable non-linear potential function parameterized by a neural network.2. Implementation SpecificationsPlease structure the code as a Python package.File: src/chlu.py (The Core Unit)Library: equinoxKey Constraint: You must use a Symplectic Integrator. Do not use a generic Runge-Kutta solver (like diffrax.dopri5) for the CHLU, as it does not conserve energy long-term.Class Structure:import jax
import jax.numpy as jnp
import equinox as eqx

class CHLU(eqx.Module):
    potential_net: eqx.Module
    log_mass: jnp.ndarray  # Learnable mass matrix diagonal

    def __init__(self, key, dim):
        # Initialize potential_net (MLP) and log_mass
        pass

    def H(self, q, p):
        # Calculate Total Energy
        # M = jax.nn.softplus(self.log_mass)
        # Kinetic = sqrt(p * M * p + 1.0)
        # Potential = self.potential_net(q)
        # Return Kinetic + Potential
        pass

    def step(self, state, dt):
        q, p = state
        # IMPLEMENT VELOCITY VERLET or LEAPFROG INTEGRATOR
        # 1. p_half = p - 0.5 * dt * dH/dq
        # 2. q_next = q + dt * dH/dp_half
        # 3. p_next = p_half - 0.5 * dt * dH/dq_next
        # Use jax.grad(self.H, argnums=...) to get derivatives
        return (q_next, p_next)

    def __call__(self, q0, p0, steps, dt):
        # Use jax.lax.scan to unroll the 'step' function efficiently
        # Return trajectory: (T, dim)
        pass
File: src/baseline.py (Neural ODE)Implement a standard Neural ODE for comparison using diffrax.Dynamics: $\dot{z} = f(z, t)$ where $f$ is an MLP.Solver: diffrax.Tsit5 or Dopri5.Constraint: Ensure the number of parameters is roughly equivalent to the CHLU for a fair comparison.File: src/regularization.py (Lyapunov Stability)To kill chaos, we penalize the Lyapunov exponents.Function: compute_lyapunov_loss(step_fn, trajectory)Logic: Compute the Jacobian $J$ of the update step at sampled points along the trajectory.Loss: $\mathcal{L}_{reg} = \text{mean}(\log(\text{singular\_values}(J)))$.Note: Use jax.jacfwd or jax.jacrev to compute $J$.3. Training Loop: src/train.pyImplement a Persistent Contrastive Divergence (PCD) loop (Wake-Sleep algorithm).Wake Phase (Supervised):Input: Time-series data (e.g., trajectory of a spiral).Objective: Minimize MSE between CHLU prediction and ground truth.Add lambda * lyapunov_loss.Sleep Phase (Unsupervised/Energy Minimization):Maintain a ReplayBuffer of states $(q, p)$.Sample states from the buffer.Run CHLU dynamics freely for $k$ steps.Objective: Minimize the Energy $H(q_T, p_T)$ of the final states (relaxing them into the "Energy Valley").Update the buffer with the new states.4. Experiments (The Paper Deliverables)Create a runner script run_experiments.py that executes the following 3 experiments and saves plots to results/.Experiment 1: Long-Horizon StabilityGoal: Demonstrate that CHLU does not "forget" or explode like Neural ODEs.Data: A simple 2D harmonic oscillator or damped spiral.Train: On the first $T=10$ timesteps.Test: Extrapolate to $T=1000$ timesteps.Metric: Mean Squared Error (MSE) over time.Plot: exp1_stability.png. Two lines (CHLU vs NeuralODE) showing error growing over time. CHLU should remain flat/bounded.Experiment 2: Noise Robustness (The "Filter" Effect)Goal: Show that CHLU mechanics naturally filter out Gaussian noise.Setup: Train both models on clean periodic data.Test: Feed inputs with added Gaussian noise $\mathcal{N}(0, \sigma)$.Plot: exp2_phase_space.png. A Phase Space plot ($q$ vs $p$).Neural ODE: Should show a jittery, distorted trajectory.CHLU: Should show a smooth trajectory confined to the "Energy Shell" (the noise is interpreted as energy fluctuations but the geometry constrains the path).Experiment 3: Generative "Dreaming" (MNIST)Goal: Validate the PCD (Wake-Sleep) training.Data: MNIST (flattened or reduced dimension via PCA).Setup: Train CHLU as a generative model where the "potential well" $V(q)$ learns the manifold of digits.Procedure:Initialize state $(q, p)$ with pure random noise.Run Hamiltonian dynamics with a small "friction" term (energy dissipation) to simulate cooling.Observe the state settling into a valid digit.Plot: exp3_dreaming.png. A grid of images showing the evolution: Noise $\to$ Hazy Shape $\to$ Crisp Digit.5. Requirements & LogisticsDependencies: jax, jaxlib, equinox, optax, diffrax, matplotlib, numpy.Reproducibility: Set a global random seed using jax.random.PRNGKey(42).Tests: Create tests/test_symplectic.py. Use jax.jacfwd to verify that the determinant of the Jacobian of the CHLU step function is exactly $1.0$ (up to float precision), confirming symplecticity.Action: Generate the full directory structure and the code files described above.