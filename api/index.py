"""Vercel Python entrypoint -- re-exports the Flask app defined in webapp/app.py.

Vercel looks for a top-level `app` (WSGI/ASGI) variable in api/*.py files; this file just
points it at the real app so the Flask/template/static code lives in one place (webapp/)
and works identically for local dev (`python webapp/app.py`) and Vercel.
"""

from webapp.app import app

__all__ = ["app"]
