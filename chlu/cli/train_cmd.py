"""Training CLI commands."""

import argparse
from pathlib import Path
from rich.console import Console
from ..project import ProjectManager

console = Console()


def setup_train_parser(subparsers):
    """Set up the 'train' subcommand parser."""
    train_parser = subparsers.add_parser(
        'train',
        help='Train models'
    )
    
    train_subparsers = train_parser.add_subparsers(
        dest='model',
        help='Model to train'
    )
    
    # train chlu
    chlu_parser = train_subparsers.add_parser(
        'chlu',
        help='Train CHLU model'
    )
    chlu_parser.add_argument('--project', help='Project name to use')
    chlu_parser.add_argument('--data', choices=['figure8', 'sine', 'mnist'], default='figure8',
                            help='Dataset to train on')
    chlu_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    chlu_parser.add_argument('--lr', type=float, help='Learning rate')
    chlu_parser.set_defaults(func=cmd_train_chlu)
    
    # train node
    node_parser = train_subparsers.add_parser(
        'node',
        help='Train Neural ODE baseline'
    )
    node_parser.add_argument('--project', help='Project name to use')
    node_parser.add_argument('--data', choices=['figure8', 'sine'], default='figure8',
                            help='Dataset to train on')
    node_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    node_parser.add_argument('--lr', type=float, help='Learning rate')
    node_parser.set_defaults(func=cmd_train_node)
    
    # train lstm
    lstm_parser = train_subparsers.add_parser(
        'lstm',
        help='Train LSTM baseline'
    )
    lstm_parser.add_argument('--project', help='Project name to use')
    lstm_parser.add_argument('--data', choices=['figure8', 'sine'], default='figure8',
                            help='Dataset to train on')
    lstm_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    lstm_parser.add_argument('--lr', type=float, help='Learning rate')
    lstm_parser.set_defaults(func=cmd_train_lstm)


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
        paths = {
            'models': Path('results'),
            'plots': Path('results')
        }
        paths['models'].mkdir(exist_ok=True)
    
    return config, paths


def cmd_train_chlu(args):
    """Train CHLU model."""
    console.print(f"[bold cyan]Training CHLU on {args.data} dataset[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    # Override config with CLI args if provided
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    console.print(f"  Epochs: {config.training.epochs}")
    console.print(f"  Learning rate: {config.training.learning_rate}")
    console.print(f"  Model save path: {paths['models']}")
    
    try:
        # TODO: Implement actual training call once train.py is updated
        console.print("⚠ Training implementation pending - will be connected after config integration", 
                     style="yellow")
        # from ..training.train import train_chlu
        # model = train_chlu(data, config=config)
        # Save model to paths['models']
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_train_node(args):
    """Train Neural ODE model."""
    console.print(f"[bold cyan]Training Neural ODE on {args.data} dataset[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    console.print(f"  Epochs: {config.training.epochs}")
    console.print(f"  Model save path: {paths['models']}")
    
    try:
        console.print("⚠ Training implementation pending", style="yellow")
        # from ..training.train_baselines import train_neural_ode
        # model = train_neural_ode(data, config=config)
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_train_lstm(args):
    """Train LSTM model."""
    console.print(f"[bold cyan]Training LSTM on {args.data} dataset[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    console.print(f"  Epochs: {config.training.epochs}")
    console.print(f"  Model save path: {paths['models']}")
    
    try:
        console.print("⚠ Training implementation pending", style="yellow")
        # from ..training.train_baselines import train_lstm
        # model = train_lstm(data, config=config)
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0
