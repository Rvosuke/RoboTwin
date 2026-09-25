"""Compatibility import for WLA launchers using the original script directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_render import Sapien_TEST


if __name__ == "__main__":
    Sapien_TEST()
