"""MNIST data loader with PCA dimensionality reduction."""

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml


def load_mnist_pca(dim: int = 32, n_samples: int = None) -> tuple:
    """
    Load MNIST dataset and apply PCA for dimensionality reduction.
    
    This reduces 28×28 = 784 dimensional images to a lower dimensional
    representation suitable for CHLU training.
    
    Args:
        dim: Target dimensionality after PCA (default: 32)
        n_samples: Number of samples to use (None = all)
    
    Returns:
        (train_data, test_data, pca_model):
            - train_data: Training data (n_train, dim)
            - test_data: Test data (n_test, dim)
            - pca_model: Fitted PCA object for inverse transform
    """
    # Load MNIST
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X = mnist.data.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Optionally subsample
    if n_samples is not None and n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    
    # Fit PCA on training data
    print(f"Applying PCA: {X_train.shape[1]} → {dim} dimensions...")
    pca = PCA(n_components=dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Convert to JAX arrays
    train_data = jnp.array(X_train_pca)
    test_data = jnp.array(X_test_pca)
    
    return train_data, test_data, pca
