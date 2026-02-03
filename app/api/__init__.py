"""API endpoints."""

from app.api.rest import router as rest_router
from app.api.websocket import router as ws_router

__all__ = ["rest_router", "ws_router"]
