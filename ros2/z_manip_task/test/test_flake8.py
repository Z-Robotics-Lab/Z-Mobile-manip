"""Run ament flake8 when available."""

import pytest


flake8 = pytest.importorskip('ament_flake8.main')


def test_flake8():
    """Check Python style."""
    rc, errors = flake8.main_with_errors(argv=[])
    assert rc == 0, 'Found %d code style errors:\n' % len(errors) + '\n'.join(errors)
