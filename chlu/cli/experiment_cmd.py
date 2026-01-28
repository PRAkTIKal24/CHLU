"""Experiment execution CLI commands."""

from pathlib import Path
from rich.console import Console
from ..project import ProjectManager
from ..experiments.exp_a_stability import run_experiment_a
from ..experiments.exp_b_noise import run_experiment_b
from ..experiments.exp_c_dreaming import run_experiment_c

console = Console()


def setup_experiment_parsers(subparsers):
    """Set up experiment subcommand parsers."""
    
    # exp-a
    exp_a_parser = subparsers.add_parser(
        'exp-a',
        help='Run Experiment A: Stability (100x extrapolation)'
    )
    exp_a_parser.add_argument('--project', help='Project name to use')
    exp_a_parser.add_argument('--seed', type=int, help='Random seed')
    exp_a_parser.add_argument('--quick', action='store_true', help='Quick mode (50 epochs)')
    exp_a_parser.set_defaults(func=cmd_exp_a)
    
    # exp-b
    exp_b_parser = subparsers.add_parser(
        'exp-b',
        help='Run Experiment B: Noise Rejection'
    )
    exp_b_parser.add_argument('--project', help='Project name to use')
    exp_b_parser.add_argument('--seed', type=int, help='Random seed')
    exp_b_parser.add_argument('--quick', action='store_true', help='Quick mode (50 epochs)')
    exp_b_parser.set_defaults(func=cmd_exp_b)
    
    # exp-c
    exp_c_parser = subparsers.add_parser(
        'exp-c',
        help='Run Experiment C: Dreaming/Generation'
    )
    exp_c_parser.add_argument('--project', help='Project name to use')
    exp_c_parser.add_argument('--seed', type=int, help='Random seed')
    exp_c_parser.add_argument('--quick', action='store_true', help='Quick mode (100 epochs)')
    exp_c_parser.set_defaults(func=cmd_exp_c)
    
    # all-experiments
    all_parser = subparsers.add_parser(
        'all-experiments',
        help='Run all experiments sequentially'
    )
    all_parser.add_argument('--project', help='Project name to use')
    all_parser.add_argument('--seed', type=int, help='Random seed')
    all_parser.add_argument('--quick', action='store_true', help='Quick mode')
    all_parser.set_defaults(func=cmd_all_experiments)


def _get_config_and_paths(args):
    """Get configuration and paths from project or defaults."""
    pm = ProjectManager()
    
    if args.project:
        try:
            config = pm.load(args.project)
            paths = pm.get_paths(args.project)
            pm.update_last_run(args.project)
        except ValueError as e:
            console.print(f"✗ Error loading project: {e}", style="bold red")
            return None, None
    else:
        from ..config import get_default_config
        config = get_default_config()
        # Use current results directory if no project specified
        paths = {
            'plots': Path('results'),
            'results': Path('results'),
            'models': Path('results')
        }
        paths['plots'].mkdir(exist_ok=True)
    
    # Override seed if provided
    if args.seed is not None:
        config.project.seed = args.seed
    
    # Adjust epochs for quick mode
    if args.quick:
        if hasattr(config, 'experiment_a'):
            config.experiment_a.train_epochs = 50
        if hasattr(config, 'experiment_b'):
            config.experiment_b.train_epochs = 50
        if hasattr(config, 'experiment_c'):
            config.experiment_c.train_epochs = 100
    
    return config, paths


def cmd_exp_a(args):
    """Run Experiment A."""
    console.print("[bold cyan]Running Experiment A: Stability Test[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    try:
        run_experiment_a(
            seed=config.project.seed,
            train_steps=config.experiment_a.train_steps,
            test_steps=config.experiment_a.test_steps,
            train_epochs=config.experiment_a.train_epochs,
            dt=config.experiment_a.dt,
            save_dir=str(paths['plots'])
        )
        console.print("✓ Experiment A completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_exp_b(args):
    """Run Experiment B."""
    console.print("[bold cyan]Running Experiment B: Noise Rejection[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    try:
        run_experiment_b(
            seed=config.project.seed,
            n_waves=config.experiment_b.n_waves,
            steps=config.experiment_b.steps,
            train_epochs=config.experiment_b.train_epochs,
            dt=config.experiment_b.dt,
            sigma_min=config.experiment_b.sigma_min,
            sigma_max=config.experiment_b.sigma_max,
            n_sigma=config.experiment_b.n_sigma,
            save_dir=str(paths['plots'])
        )
        console.print("✓ Experiment B completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_exp_c(args):
    """Run Experiment C."""
    console.print("[bold cyan]Running Experiment C: Dreaming[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    try:
        run_experiment_c(
            seed=config.project.seed,
            pca_dim=config.experiment_c.pca_dim,
            train_epochs=config.experiment_c.train_epochs,
            n_samples=config.experiment_c.n_samples,
            dream_steps=config.experiment_c.dream_steps,
            friction=config.experiment_c.friction,
            dt=config.experiment_c.dt,
            save_dir=str(paths['plots'])
        )
        console.print("✓ Experiment C completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_all_experiments(args):
    """Run all experiments."""
    console.print("[bold cyan]Running All Experiments[/bold cyan]")
    
    experiments = [
        ('A: Stability', cmd_exp_a),
        ('B: Noise Rejection', cmd_exp_b),
        ('C: Dreaming', cmd_exp_c)
    ]
    
    for name, func in experiments:
        console.print(f"\n[bold]Starting {name}...[/bold]")
        result = func(args)
        if result != 0:
            console.print(f"✗ Failed at experiment {name}", style="bold red")
            return result
    
    console.print("\n✓ All experiments completed successfully!", style="bold green")
    return 0
