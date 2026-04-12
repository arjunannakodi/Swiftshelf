# server/app.py
# ---------------------------------------------------------------------------
# OpenEnv multi-mode deployment entry point.
# The canonical app lives at the repo root (app.py); this module re-exports
# it so the validator's "server/app.py" presence check passes while the
# Dockerfile continues to start `app:app` from /app.
# ---------------------------------------------------------------------------

import sys
import os

# Ensure the repo root is on sys.path so the root `app` module is importable
# regardless of where Python is invoked from.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app import app  # noqa: F401  – re-export for `server.app:app`

__all__ = ["app"]


def main():
    """Entry point for multi-mode deployment (uv run / openenv serve)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
