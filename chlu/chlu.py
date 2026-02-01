"""
CHLU - Causal Hamiltonian Learning Unit

Main entry point for the CHLU package.
"""

import argparse
import sys

from art import text2art
from rich.console import Console

from .cli import (
    setup_data_parser,
    setup_experiment_parsers,
    setup_project_parser,
    setup_train_parser,
    setup_utils_parsers,
)

console = Console()


def main():
    """Main entry point for CHLU CLI."""
    parser = argparse.ArgumentParser(
        description="CHLU - Causal Hamiltonian Learning Unit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run chlu project create myexp
  uv run chlu exp-a --project myexp
  uv run chlu train chlu --data figure8
  uv run chlu info
        """,
    )

    # Global options
    parser.add_argument("--version", action="version", version="CHLU 0.2.3")

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Set up all command parsers
    setup_project_parser(subparsers)
    setup_experiment_parsers(subparsers)
    setup_train_parser(subparsers)
    setup_data_parser(subparsers)
    setup_utils_parsers(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command provided, show help
    if args.command is None:
        banner = text2art("CHLU", font="varsity")
        console.print(f"[bold cyan]{banner}[/bold cyan]")
        console.print("[bold green]Causal Hamiltonian Learning Unit[/bold green]")
        console.print("\n[dim]Run with --help to see available commands[/dim]\n")
        parser.print_help()
        return 0

    # Execute the command
    if hasattr(args, "func"):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            return 130
        except Exception as e:
            console.print(f"[bold red]Unexpected error: {e}[/bold red]")
            import traceback

            traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
