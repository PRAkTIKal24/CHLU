"""Project management CLI commands."""

import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from ..project import ProjectManager

console = Console()


def setup_project_parser(subparsers):
    """Set up the 'project' subcommand parser."""
    project_parser = subparsers.add_parser(
        'project',
        help='Manage CHLU projects'
    )
    
    project_subparsers = project_parser.add_subparsers(
        dest='project_command',
        help='Project command'
    )
    
    # project create
    create_parser = project_subparsers.add_parser(
        'create',
        help='Create a new project'
    )
    create_parser.add_argument(
        'name',
        help='Project name'
    )
    create_parser.add_argument(
        '--description',
        default='',
        help='Project description'
    )
    create_parser.set_defaults(func=cmd_project_create)
    
    # project list
    list_parser = project_subparsers.add_parser(
        'list',
        help='List all projects'
    )
    list_parser.set_defaults(func=cmd_project_list)
    
    # project delete
    delete_parser = project_subparsers.add_parser(
        'delete',
        help='Delete a project'
    )
    delete_parser.add_argument(
        'name',
        help='Project name to delete'
    )
    delete_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation'
    )
    delete_parser.set_defaults(func=cmd_project_delete)


def cmd_project_create(args):
    """Create a new project."""
    pm = ProjectManager()
    
    try:
        project_path = pm.create(args.name, args.description)
        console.print(f"✓ Created project: [bold green]{args.name}[/bold green]", style="green")
        console.print(f"  Location: {project_path}")
        console.print("\n  Directory structure:")
        console.print("    • config/   - Configuration files")
        console.print("    • plots/    - Generated visualizations")
        console.print("    • results/  - Experiment results")
        console.print("    • models/   - Trained models")
        console.print(f"\n  Next: [bold]uv run chlu exp-a --project {args.name}[/bold]")
    except ValueError as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_project_list(args):
    """List all projects."""
    pm = ProjectManager()
    projects = pm.list_all()
    
    if not projects:
        console.print("No projects found. Create one with: [bold]uv run chlu project create <name>[/bold]")
        return 0
    
    table = Table(title="CHLU Projects")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Created", style="green")
    table.add_column("Last Run", style="yellow")
    
    for project in projects:
        created = project['created'].split('T')[0] if project['created'] else 'N/A'
        last_run = project['last_run'].split('T')[0] if project.get('last_run') else 'Never'
        table.add_row(
            project['name'],
            project.get('description', ''),
            created,
            last_run
        )
    
    console.print(table)
    return 0


def cmd_project_delete(args):
    """Delete a project."""
    pm = ProjectManager()
    
    try:
        pm.delete(args.name, force=args.force)
        console.print(f"✓ Deleted project: [bold]{args.name}[/bold]", style="green")
    except ValueError as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0
