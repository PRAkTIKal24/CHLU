"""Data generators for CHLU experiments."""

from chlu.data.figure8 import generate_figure8
from chlu.data.sine_waves import generate_sine_waves, add_noise
from chlu.data.mnist import load_mnist_pca

__all__ = ["generate_figure8", "generate_sine_waves", "add_noise", "load_mnist_pca"]
