"""Entrypoint for report figure and table generation."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.report.plots import main

if __name__ == "__main__":
    sys.exit(main())

