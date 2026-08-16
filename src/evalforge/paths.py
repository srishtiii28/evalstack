"""Where EvalForge keeps its state by default.

Everything lives under one directory so a project's evaluation history is easy
to inspect, back up, or delete.
"""

from __future__ import annotations

from pathlib import Path

STATE_DIR = Path(".evalforge")
DEFAULT_DATABASE = STATE_DIR / "runs.db"
DEFAULT_TRAJECTORY_DIR = STATE_DIR / "trajectories"
DEFAULT_DATASETS_ROOT = Path("datasets")

#: Local secrets, git-ignored. See .env.example for the shape.
DEFAULT_ENV_FILE = Path(".env")

#: Cached model responses. Keeping them under the state directory means a
#: repeated run costs no quota, and deleting the directory forces fresh calls.
DEFAULT_CACHE_DIR = STATE_DIR / "model-cache"
