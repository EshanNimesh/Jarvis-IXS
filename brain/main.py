"""Compatibility entry point for ``uvicorn main:app`` from ``brain/``."""

from app.main import app

__all__ = ["app"]
