"""CLI module for CHLU."""

from .project_cmd import setup_project_parser
from .experiment_cmd import setup_experiment_parsers
from .train_cmd import setup_train_parser
from .data_cmd import setup_data_parser
from .utils_cmd import setup_utils_parsers


__all__ = [
    'setup_project_parser',
    'setup_experiment_parsers',
    'setup_train_parser',
    'setup_data_parser',
    'setup_utils_parsers'
]
