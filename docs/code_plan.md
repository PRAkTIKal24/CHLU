Research Engineering Task: Causal Hamiltonian Learning Unit (CHLU)Role: You are a Principal Research Engineer assisting in the development of a novel AI primitive called the Causal Hamiltonian Learning Unit (CHLU).Goal: Create a reproducible JAX-based software package to run experiments for an ICLR Workshop paper (AI & PDE).Framework: JAX, Equinox (for modularity), Optax (optimizers), Diffrax (ODEs).1. Mathematical Primitives & TheoryThe CHLU is a dynamical system unit grounded in Symplectic Mechanics, designed to solve the memory-stability trade-off inherent in RNNs and Neural ODEs.The Physics (Hamiltonian Dynamics)Unlike standard RNNs, the state $z = (q, p)$ (Position, Momentum) evolves according to Hamilton's Equations, which strictly conserve phase space volume (Liouville's Theorem):$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = - \frac{\partial H}{\partial q}$$The Hamiltonian ($H$)We utilize a Relativistic Hamiltonian to ensure causal bounds (speed of light limit) and energy stability.$$H(q, p) = \underbrace{\sqrt{p^T M p + m^2}}_{\text{Relativistic Kinetic Energy}} + \underbrace{V(q)}_{\text{Potential Energy (MLP)}}$$$M$: A learnable, positive-definite mass matrix (can be diagonal for simplicity).$m$: Rest mass constant (hyperparameter, e.g., 1.0).$V(q)$: A learnable non-linear potential function parameterized by a neural network.2. Implementation SpecificationsStructure the code as a Python package.File: src/chlu.py (The Core Unit)Library: equinoxKey Constraint: You must use a Symplectic Integrator. Do not use a generic Runge-Kutta solver for the CHLU.Class Structure:import jax
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
File: src/baselines.py (NODE & LSTM)We need two baselines to demonstrate the "Double Win": defeating LSTMs on stability and NODEs on conservation.1. Neural ODE (NODE)Library: diffraxDynamics: $\dot{z} = f(z, t)$ where $f$ is an MLP.Solver: diffrax.Tsit5 (Runge-Kutta).Note: Represents dissipative continuous systems.2. LSTM (The Anchor)Library: equinoxType: Standard discrete sequence model.Architecture: eqx.nn.LSTMCell wrapped to process a sequence autoregressively.Task: Given $x_t$, predict $x_{t+1}$.Note: Represents discrete systems that lack conservation laws.File: src/regularization.py (Lyapunov Stability)To kill chaos and enforce the "Energy Valley":Function: compute_lyapunov_loss(step_fn, trajectory)Logic: Compute the Jacobian $J$ of the update step using jax.jacfwd.Loss: $\mathcal{L}_{reg} = \text{mean}(\log(\text{singular\_values}(J)))$.3. Training Loop: src/train.pyImplement a Persistent Contrastive Divergence (PCD) loop (Wake-Sleep algorithm).Wake Phase (Supervised):Input: Time-series data (e.g., Figure-8 orbit).Objective: Minimize MSE between prediction and ground truth.Add lambda * lyapunov_loss.Sleep Phase (Unsupervised/Energy Minimization):Maintain a ReplayBuffer of states $(q, p)$.Run CHLU dynamics freely for $k$ steps.Objective: Minimize the Energy $H(q_T, p_T)$ of the final states.Interpretation: This relaxes noise into the low-energy manifold (the "data valley").4. Experiments (The Paper Deliverables)Create a runner script run_experiments.py that executes these experiments and saves plots to results/.Experiment A: The "Eternal Memory" Stability TestGoal: Compare infinite-horizon stability.Dataset: Figure-8 Orbit (Lemniscate of Bernoulli).Generate synthetic ground truth data for a stable Figure-8 cycle in 2D space.Protocol:Train all 3 models (CHLU, NODE, LSTM) on $T=100$ steps of the orbit.Run models autonomously for $T=10,000$ steps (100x extrapolation).Plot: exp1_stability.png (3 Subplots in a row):Left (LSTM): Show trajectory drift/explosion (Chaos).Middle (NODE): Show trajectory spiraling inward (Dissipation).Right (CHLU): Show a solid, stable Figure-8 track (Conservation).Experiment B: Energy-Based Noise RejectionGoal: Test the "Noise Filter" hypothesis.Dataset: Clean Sine Waves.Protocol:Train models on clean sine waves.Test on inputs corrupted by Gaussian noise $\mathcal{N}(0, \sigma)$ with $\sigma \in [0.1, 1.0]$.Hypothesis: CHLU treats noise as "high energy" and slides it down the potential well, recovering the signal. LSTM tries to fit the noise.Plot: exp2_noise_curve.png.X-axis: Noise Sigma ($\sigma$).Y-axis: Reconstruction MSE.Lines: LSTM (Steep rising error) vs CHLU (Flat/Robust error).Experiment C: Generative "Dreaming" (MNIST)Goal: Validate the Generative Capability of the PCD loop.Data: MNIST (compressed to lower dim via PCA/Autoencoder if needed).Protocol:Train CHLU as a generative model.Initialize with pure noise.Run dynamics with friction to settle into "digit valleys".Plot: exp3_dreaming.png. Evolution from Noise $\to$ Digit.5. Requirements & LogisticsDependencies: jax, jaxlib, equinox, optax, diffrax, matplotlib, numpy, hydra-core.Reproducibility: Global seed jax.random.PRNGKey(42).Tests: tests/test_symplectic.py (Verify $\det(J) == 1.0$).Action: Generate the full directory structure and the code files described above.