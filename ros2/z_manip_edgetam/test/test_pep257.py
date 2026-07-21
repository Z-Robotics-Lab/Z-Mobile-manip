"""Run ROS 2 docstring checks."""

from ament_pep257.main import main
import pytest


@pytest.mark.linter
def test_pep257() -> None:
    """Require public Python symbols to have useful docstrings."""
    assert main(argv=['.']) == 0
