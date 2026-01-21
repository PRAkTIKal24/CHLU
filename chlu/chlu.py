"""
CHLU - Causal Hamiltonian Learning Unit

Main entry point for the CHLU package.
"""

import argparse
import tomllib
from pathlib import Path
from rich import print as rprint
from art import text2art


def get_version():
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def main():
    """Main entry point for CHLU."""
    version = get_version()
    
    parser = argparse.ArgumentParser(
        description="CHLU - Causal Hamiltonian Learning Unit"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CHLU v{version}"
    )
    
    args = parser.parse_args()
    
    # Display welcome banner
    banner = text2art("CHLU", font="block")
    rprint(f"[bold cyan]{banner}[/bold cyan]")
    rprint("[bold green]Causal Hamiltonian Learning Unit[/bold green]")
    rprint(f"[yellow]Version {version}[/yellow]")
    rprint("\n[dim]Ready to build CHLU functionality...[/dim]\n")


if __name__ == "__main__":
    main()
