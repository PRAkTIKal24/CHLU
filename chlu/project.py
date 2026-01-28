"""
Project management for CHLU.

Handles creation, management, and organization of CHLU projects with
standardized directory structures for configs, plots, results, and models.
"""

from pathlib import Path
from typing import Optional, List, Dict
import json
from datetime import datetime
from .config import CHLUConfig, get_default_config, save_config, load_config


class ProjectManager:
    """Manages CHLU project directories and metadata."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize project manager.
        
        Args:
            base_dir: Base directory for all projects. Defaults to ./projects
        """
        if base_dir is None:
            base_dir = Path.cwd() / "projects"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create(self, name: str, description: str = "") -> Path:
        """
        Create a new project with standard directory structure.
        
        Args:
            name: Project name (used as directory name)
            description: Optional project description
            
        Returns:
            Path to the created project directory
            
        Raises:
            ValueError: If project already exists
        """
        project_path = self.base_dir / name
        
        if project_path.exists():
            raise ValueError(f"Project '{name}' already exists at {project_path}")
        
        # Create directory structure
        project_path.mkdir(parents=True)
        (project_path / "config").mkdir()
        (project_path / "plots").mkdir()
        (project_path / "results").mkdir()
        (project_path / "models").mkdir()
        
        # Create default config
        default_config = get_default_config()
        default_config.project.name = name
        config_path = project_path / "config" / "config.yaml"
        save_config(default_config, config_path)
        
        # Create project metadata
        metadata = {
            "name": name,
            "description": description,
            "created": datetime.now().isoformat(),
            "last_run": None,
            "version": "0.1.0"
        }
        metadata_path = project_path / "project.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        readme_content = f"""# {name}

{description}

## Directory Structure

- `config/` - Configuration files (edit config.yaml to customize parameters)
- `plots/` - Generated plots and visualizations
- `results/` - Experiment results (numpy arrays, metrics, etc.)
- `models/` - Trained model checkpoints

## Created

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        (project_path / "README.md").write_text(readme_content)
        
        return project_path
    
    def list_all(self) -> List[Dict[str, str]]:
        """
        List all existing projects.
        
        Returns:
            List of dictionaries containing project metadata
        """
        projects = []
        for project_dir in sorted(self.base_dir.iterdir()):
            if project_dir.is_dir() and (project_dir / "project.json").exists():
                metadata_path = project_dir / "project.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                projects.append(metadata)
        return projects
    
    def delete(self, name: str, force: bool = False) -> None:
        """
        Delete a project.
        
        Args:
            name: Project name to delete
            force: If True, skip confirmation
            
        Raises:
            ValueError: If project doesn't exist
        """
        project_path = self.base_dir / name
        
        if not project_path.exists():
            raise ValueError(f"Project '{name}' does not exist")
        
        if not force:
            try:
                from rich.prompt import Confirm
                if not Confirm.ask(f"Delete project '{name}'?", default=False):
                    return
            except ImportError:
                # Fallback if rich not available
                response = input(f"Delete project '{name}'? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    return
        
        # Delete the directory
        import shutil
        shutil.rmtree(project_path)
    
    def load(self, name: str) -> CHLUConfig:
        """
        Load configuration from a project.
        
        Args:
            name: Project name
            
        Returns:
            CHLUConfig object with project settings
            
        Raises:
            ValueError: If project doesn't exist
        """
        project_path = self.base_dir / name
        
        if not project_path.exists():
            raise ValueError(f"Project '{name}' does not exist")
        
        config_path = project_path / "config" / "config.yaml"
        if not config_path.exists():
            raise ValueError(f"Config file not found in project '{name}'")
        
        config = load_config(config_path)
        
        # Set project-specific paths
        config.project.name = name
        if config.project.save_dir is None:
            config.project.save_dir = str(project_path)
        
        return config
    
    def get_project_path(self, name: str) -> Path:
        """
        Get the full path to a project directory.
        
        Args:
            name: Project name
            
        Returns:
            Path to project directory
        """
        return self.base_dir / name
    
    def update_last_run(self, name: str) -> None:
        """
        Update the last_run timestamp in project metadata.
        
        Args:
            name: Project name
        """
        project_path = self.base_dir / name
        metadata_path = project_path / "project.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['last_run'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def get_paths(self, name: str) -> Dict[str, Path]:
        """
        Get all standard paths for a project.
        
        Args:
            name: Project name
            
        Returns:
            Dictionary with 'root', 'config', 'plots', 'results', 'models' paths
        """
        project_path = self.base_dir / name
        return {
            'root': project_path,
            'config': project_path / "config",
            'plots': project_path / "plots",
            'results': project_path / "results",
            'models': project_path / "models"
        }
