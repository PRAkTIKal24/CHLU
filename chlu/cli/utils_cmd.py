"""Utility CLI commands."""

import argparse
from rich.console import Console
from rich.table import Table
import sys

console = Console()


def setup_utils_parsers(subparsers):
    """Set up utility subcommand parsers."""
    
    # info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show system and package information'
    )
    info_parser.set_defaults(func=cmd_info)
    
    # config show command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration utilities'
    )
    config_subparsers = config_parser.add_subparsers(dest='config_command')
    
    show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    show_parser.add_argument('--project', help='Project name')
    show_parser.set_defaults(func=cmd_config_show)


def cmd_info(args):
    """Show system information."""
    console.print("[bold cyan]CHLU System Information[/bold cyan]\n")
    
    # Python version
    console.print(f"Python: {sys.version.split()[0]}")
    
    # Package versions
    try:
        import jax
        console.print(f"JAX: {jax.__version__}")
    except Exception:
        console.print("JAX: [red]Not available[/red]")
    
    try:
        import equinox
        console.print(f"Equinox: {equinox.__version__}")
    except Exception:
        console.print("Equinox: [red]Not available[/red]")
    
    try:
        import numpy as np
        console.print(f"NumPy: {np.__version__}")
    except Exception:
        console.print("NumPy: [red]Not available[/red]")
    
    # JAX devices
    try:
        import jax
        devices = jax.devices()
        console.print(f"\nJAX Devices: {len(devices)}")
        for i, device in enumerate(devices):
            console.print(f"  [{i}] {device.device_kind}: {device}")
    except Exception as e:
        console.print(f"\nJAX Devices: [red]Error - {e}[/red]")
    
    return 0


def cmd_config_show(args):
    """Show configuration."""
    from ..config import get_default_config
    from ..project import ProjectManager
    import yaml
    
    console.print("[bold cyan]CHLU Configuration[/bold cyan]\n")
    
    if args.project:
        pm = ProjectManager()
        try:
            config = pm.load(args.project)
            console.print(f"Project: [bold]{args.project}[/bold]\n")
        except ValueError as e:
            console.print(f"✗ Error: {e}", style="bold red")
            return 1
    else:
        config = get_default_config()
        console.print("Using default configuration\n")
    
    # Convert to dict and display
    from dataclasses import asdict
    config_dict = {
        'model': asdict(config.model),
        'training': asdict(config.training),
        'experiment_a': asdict(config.experiment_a),
        'experiment_b': asdict(config.experiment_b),
        'experiment_c': asdict(config.experiment_c),
        'data': asdict(config.data),
        'project': asdict(config.project)
    }
    
    yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    console.print(yaml_str)
    
    return 0
