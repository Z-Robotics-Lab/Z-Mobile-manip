from ament_pep257.main import main
import pytest


@pytest.mark.linter
def test_pep257() -> None:
    assert main(argv=['.']) == 0
