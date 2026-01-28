"""
Basic tests for CHLU package.
"""

from chlu import chlu


def test_import():
    """Test that the chlu module can be imported."""
    assert chlu is not None


def test_main_exists():
    """Test that the main function exists."""
    assert hasattr(chlu, 'main')
    assert callable(chlu.main)
