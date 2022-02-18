"""
Test a package version is exported.
"""
from numpy_hirschberg import __version__


def test_version():
    """
    Compare the current package version.
    """
    assert __version__ == "0.1.0"
