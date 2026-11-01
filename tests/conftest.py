"""Pytest configuration and environment fixtures for GeneWeaver test suite."""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure CUDA environment paths and hooks are initialized before any tests collect/run
try:
    import alignment.cuda_alignment  # noqa: F401
except Exception:
    pass
