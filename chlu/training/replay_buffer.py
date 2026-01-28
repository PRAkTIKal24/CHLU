"""Replay buffer for PCD sleep phase."""

import jax
import jax.numpy as jnp


class ReplayBuffer:
    """
    Replay buffer for Persistent Contrastive Divergence (PCD).
    
    Stores (q, p) states that are evolved during the sleep phase
    for energy minimization.
    """
    
    def __init__(self, capacity: int = 1024, dim: int = 2):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of states to store
            dim: Dimensionality of q and p
        """
        self.capacity = capacity
        self.dim = dim
        # Buffer shape: (capacity, 2, dim) for (q, p) pairs
        self.buffer = jnp.zeros((capacity, 2, dim))
        self.initialized = False
    
    def sample(self, key: jax.random.PRNGKey, batch_size: int) -> tuple:
        """
        Sample random (q, p) pairs from buffer.
        
        Args:
            key: JAX random key
            batch_size: Number of samples
        
        Returns:
            (q_batch, p_batch): Sampled states, each of shape (batch_size, dim)
        """
        indices = jax.random.randint(
            key, 
            shape=(batch_size,), 
            minval=0, 
            maxval=self.capacity
        )
        
        samples = self.buffer[indices]  # (batch_size, 2, dim)
        q_batch = samples[:, 0, :]  # (batch_size, dim)
        p_batch = samples[:, 1, :]  # (batch_size, dim)
        
        return q_batch, p_batch, indices
    
    def update(self, new_states: tuple, indices: jnp.ndarray):
        """
        Update buffer at given indices with new states.
        
        Args:
            new_states: (q, p) tuple of arrays (batch_size, dim)
            indices: Indices to update (batch_size,)
        
        Returns:
            Updated buffer
        """
        q, p = new_states
        # Stack q and p: (batch_size, 2, dim)
        stacked = jnp.stack([q, p], axis=1)
        
        # Update buffer
        self.buffer = self.buffer.at[indices].set(stacked)
        self.initialized = True
        
        return self.buffer
    
    def initialize_random(self, key: jax.random.PRNGKey, scale: float = 1.0):
        """
        Initialize buffer with random values.
        
        Args:
            key: JAX random key
            scale: Scale of random initialization
        """
        self.buffer = jax.random.normal(key, (self.capacity, 2, self.dim)) * scale
        self.initialized = True
