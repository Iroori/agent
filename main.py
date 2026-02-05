"""Main entry point for the High-Performance Multi-Agent System."""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from loguru import logger

from app.agents.pool import get_agent_pool
from app.api.rest import router as rest_router
from app.api.websocket import router as ws_router
from app.core.settings import settings
from app.loaders.base import BaseAgentLoader
from app.loaders.file_loader import FileAgentLoader
from app.loaders.api_loader import SeedAIAPILoader
import app.tools  # noqa: F401 - Import to register mandatory tools (math)


def create_agent_loader() -> BaseAgentLoader:
    """Create agent loader based on configuration.

    Returns:
        BaseAgentLoader instance based on AGENT_CONFIG_LOAD_TYPE setting
    """
    if settings.agent_config_load_type == "seedai-api":
        logger.debug("Using SeedAI API loader")
        return SeedAIAPILoader()
    else:
        logger.debug(f"Using file loader from {settings.agents_config_dir}")
        return FileAgentLoader()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.debug("Starting High-Performance Multi-Agent System...")
    logger.debug(f"Debug mode: {settings.debug}")
    logger.debug(f"Config load type: {settings.agent_config_load_type}")

    # Initialize agent loader based on config type
    loader = create_agent_loader()

    # Start file watching if using file loader
    if isinstance(loader, FileAgentLoader):
        logger.debug(f"Agent config directory: {settings.agents_config_dir}")
        await loader.watch_changes()

    # Initialize agent pool
    pool = get_agent_pool()
    pool.set_loader(loader)
    await pool.start()

    logger.debug("System startup complete")

    yield

    # Shutdown
    logger.debug("Shutting down...")

    # Stop file watching if applicable
    if isinstance(loader, FileAgentLoader):
        await loader.stop_watching()

    # Shutdown agent pool (gracefully stops all agents)
    await pool.shutdown()

    logger.debug("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="High-Performance Multi-Agent System",
        description="Python 3.14 Free-threaded Multi-Agent Platform",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Include routers
    app.include_router(rest_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")

    return app


app = create_app()


def handle_shutdown(signum: int, frame: object) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    # The actual shutdown is handled by uvicorn's signal handling
    # which will trigger the lifespan context manager's cleanup


def main() -> None:
    """Run the application."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    logger.info(f"Starting server on {settings.host}:{settings.port}")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    main()
