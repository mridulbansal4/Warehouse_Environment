"""
Hugging Face Spaces metadata entry (``app_file`` in README).

The Docker image runs ``uvicorn server.app:app``; this module re-exports the same
``app`` so tooling that resolves ``app.py`` finds the FastAPI application.
"""

from server.app import app

__all__ = ["app"]
