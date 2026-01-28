"""Main experiment runner for CHLU research paper.

Runs all three experiments:
    A. Eternal Memory Stability Test
    B. Energy-Based Noise Rejection
    C. Generative Dreaming (MNIST)
"""

import argparse
import jax

from chlu.experiments.exp_a_stability import run_experiment_a
from chlu.experiments.exp_b_noise import run_experiment_b
from chlu.experiments.exp_c_dreaming import run_experiment_c


def main():
    """Run all CHLU experiments."""
    parser = argparse.ArgumentParser(
        description="CHLU Experiments for ICLR Workshop Paper"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["a", "b", "c", "all"],
        default="all",
        help="Which experiment to run (default: all)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/",
        help="Directory to save results (default: results/)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: reduced epochs for testing"
    )
    
    args = parser.parse_args()
    
    # Adjust epochs for quick mode
    if args.quick:
        train_epochs = 50
        print("\n[QUICK MODE] Using reduced epochs (50) for testing\n")
    else:
        train_epochs = 500
    
    # Set JAX to use CPU if no GPU available
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Random seed: {args.seed}")
    print(f"Save directory: {args.save_dir}\n")
    
    # Print header
    print("=" * 70)
    print(" " * 15 + "CHLU EXPERIMENTS FOR ICLR WORKSHOP")
    print(" " * 10 + "Causal Hamiltonian Learning Unit - Research Paper")
    print("=" * 70)
    
    # Run experiments
    if args.experiment in ["a", "all"]:
        run_experiment_a(
            save_dir=args.save_dir,
            seed=args.seed,
            train_epochs=train_epochs,
        )
    
    if args.experiment in ["b", "all"]:
        run_experiment_b(
            save_dir=args.save_dir,
            seed=args.seed,
            train_epochs=train_epochs,
        )
    
    if args.experiment in ["c", "all"]:
        run_experiment_c(
            save_dir=args.save_dir,
            seed=args.seed,
            train_epochs=train_epochs * 2,  # MNIST needs more epochs
        )
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {args.save_dir}")
    print("\nGenerated files:")
    print(f"  • {args.save_dir}/exp1_stability.png")
    print(f"  • {args.save_dir}/exp2_noise_curve.png")
    print(f"  • {args.save_dir}/exp3_dreaming.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
