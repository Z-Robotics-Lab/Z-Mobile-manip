"""Make the source package and repository core importable without colcon."""

from pathlib import Path
import sys


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(REPOSITORY))
