"""
CHLU - Causal Hamiltonian Learning Unit

Main entry point for the CHLU package.
"""

import argparse
from rich import print as rprint
from art import text2art


def main():
    """Main entry point for CHLU."""
    parser = argparse.ArgumentParser(
        description="CHLU - Causal Hamiltonian Learning Unit"
    )
    
    args = parser.parse_args()
    
    # Display welcome banner
    banner = text2art("CHLU", font="block")
    rprint(f"[bold cyan]{banner}[/bold cyan]")
    rprint("[bold green]Causal Hamiltonian Learning Unit[/bold green]")
    rprint("\n[dim]Ready to build CHLU functionality...[/dim]\n")


if __name__ == "__main__":
    main()
