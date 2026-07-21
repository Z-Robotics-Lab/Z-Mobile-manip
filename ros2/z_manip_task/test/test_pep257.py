"""Run ament pep257 when available."""

import pytest


pep257 = pytest.importorskip('ament_pep257.main')


def test_pep257():
    """Check docstrings."""
    rc = pep257.main(argv=['.', 'test'])
    assert rc == 0
