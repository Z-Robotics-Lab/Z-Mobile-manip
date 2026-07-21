"""Run ROS 2 Python style checks."""

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.linter
def test_flake8() -> None:
    """Require the package to satisfy ament flake8."""
    rc, errors = main_with_errors(argv=[])
    assert rc == 0, 'Found %d style errors:\n' % len(errors) + '\n'.join(errors)
