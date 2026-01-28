"""Training CLI commands."""

import argparse
from pathlib import Path
from rich.console import Console
import jax
import jax.numpy as jnp

from ..project import ProjectManager
from ..core.chlu_unit import CHLU
from ..core.baselines import NeuralODE, LSTMPredictor
from ..training.train import train_chlu as train_chlu_fn
from ..training.train_baselines import train_neural_ode as train_node_fn
from ..training.train_baselines import train_lstm as train_lstm_fn
from ..data.figure8 import generate_figure8
from ..data.sine_waves import generate_sine_waves
from ..data.mnist import load_mnist_pca
from ..utils.checkpoints import save_checkpoint, create_checkpoint_name

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
        # Generate or load data
        key = jax.random.PRNGKey(config.project.seed)
        k1, k2 = jax.random.split(key)
        
        console.print(f"  Generating {args.data} data...")
        if args.data == 'figure8':
            train_data = generate_figure8(k1, steps=500, dt=config.training.dt)
            dim = 2
        elif args.data == 'sine':
            train_data = generate_sine_waves(k1, n_waves=100, steps=200, dt=config.training.dt)
            dim = 1
        elif args.data == 'mnist':
            train_imgs, _, _ = load_mnist_pca(dim=config.experiment_c.pca_dim)
            # Convert to (q, p) format
            n_train = len(train_imgs)
            p_train = jax.random.normal(k1, (n_train, config.experiment_c.pca_dim)) * 0.1
            train_data = jnp.concatenate([train_imgs, p_train], axis=-1)
            train_data = train_data[:, None, :]  # Add time dimension
            dim = config.experiment_c.pca_dim
        else:
            console.print(f"[red]Unknown dataset: {args.data}[/red]")
            return 1
        
        console.print(f"  Training data shape: {train_data.shape}")
        
        # Initialize model
        chlu = CHLU(dim=dim, hidden=config.model.hidden_dim, key=k2)
        
        # Train model
        console.print("  Training...")
        trained_model, losses = train_chlu_fn(
            chlu,
            train_data,
            key=k2,
            config=config
        )
        
        console.print(f"  Final loss: {losses[-1]:.6f}")
        
        # Save model
        model_name = create_checkpoint_name(
            f"chlu_{args.data}",
            epoch=config.training.epochs,
            loss=losses[-1]
        )
        model_path = paths['models'] / model_name
        
        save_checkpoint(
            trained_model,
            model_path,
            epoch=config.training.epochs,
            loss=losses[-1],
            config=config,
            dataset=args.data
        )
        
        console.print(f"[green]✓ Model saved to {model_path}[/green]")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        import traceback
        traceback.print_exc()
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
        # Generate data
        key = jax.random.PRNGKey(config.project.seed)
        k1, k2 = jax.random.split(key)
        
        console.print(f"  Generating {args.data} data...")
        if args.data == 'figure8':
            train_data = generate_figure8(k1, steps=500, dt=config.training.dt)
            dim = 4  # (x, y, vx, vy)
        elif args.data == 'sine':
            train_data = generate_sine_waves(k1, n_waves=100, steps=200, dt=config.training.dt)
            dim = 2  # (x, dx/dt)
        else:
            console.print(f"[red]Neural ODE doesn't support {args.data} dataset[/red]")
            return 1
        
        console.print(f"  Training data shape: {train_data.shape}")
        
        # Initialize and train
        node = NeuralODE(dim=dim, hidden=config.model.hidden_dim, key=k2)
        
        console.print("  Training...")
        trained_model, losses = train_node_fn(
            node,
            train_data,
            key=k2,
            config=config
        )
        # Generate data
        key = jax.random.PRNGKey(config.project.seed)
        k1, k2 = jax.random.split(key)
        
        console.print(f"  Generating {args.data} data...")
        if args.data == 'figure8':
            train_data = generate_figure8(k1, steps=500, dt=config.training.dt)
            dim = 4  # (x, y, vx, vy)
        elif args.data == 'sine':
            train_data = generate_sine_waves(k1, n_waves=100, steps=200, dt=config.training.dt)
            dim = 2  # (x, dx/dt)
        else:
            console.print(f"[red]LSTM doesn't support {args.data} dataset[/red]")
            return 1
        
        console.print(f"  Training data shape: {train_data.shape}")
        
        # Initialize and train
        lstm = LSTMPredictor(dim=dim, hidden_size=config.model.hidden_dim, key=k2)
        
        console.print("  Training...")
        trained_model, losses = train_lstm_fn(
            lstm,
            train_data,
            key=k2,
            config=config
        )
        
        console.print(f"  Final loss: {losses[-1]:.6f}")
        
        # Save model
        model_name = create_checkpoint_name(
            f"lstm_{args.data}",
            epoch=config.training.epochs,
            loss=losses[-1]
        )
        model_path = paths['models'] / model_name
        
        save_checkpoint(
            trained_model,
            model_path,
            epoch=config.training.epochs,
            loss=losses[-1],
            config=config,
            dataset=args.data
        )
        
        console.print(f"[green]✓ Model saved to {model_path}[/green]")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        import traceback
        traceback.print_exc(
            f"node_{args.data}",
            epoch=config.training.epochs,
            loss=losses[-1]
        )
        model_path = paths['models'] / model_name
        
        save_checkpoint(
            trained_model,
            model_path,
            epoch=config.training.epochs,
            loss=losses[-1],
            config=config,
            dataset=args.data
        )
        
        console.print(f"[green]✓ Model saved to {model_path}[/green]")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        import traceback
        traceback.print_exc()
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
