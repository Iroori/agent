"""Thread-safe singleton agent pool."""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from app.agents.base_agent import AgentStatus, BaseAgent
from app.agents.builder import AgentBuilder, get_agent_builder
from app.core.settings import settings
from app.loaders.base import AgentInfo, BaseAgentLoader


class AgentPool:
    """Thread-safe singleton pool for managing agent instances.

    Features:
    - Singleton pattern with thread-safe initialization
    - Lazy agent creation on demand
    - Automatic cleanup of idle agents
    - Graceful shutdown support
    """

    _instance: "AgentPool | None" = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "AgentPool":
        """Create singleton instance with thread-safe double-checked locking."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize pool (only runs once due to singleton)."""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            self._agents: dict[str, BaseAgent] = {}
            self._agent_lock = threading.Lock()
            self._loader: BaseAgentLoader | None = None
            self._builder: AgentBuilder = get_agent_builder()
            self._max_size = settings.agent_pool_max_size
            self._idle_timeout = timedelta(seconds=settings.agent_idle_timeout_seconds)
            self._cleanup_task: asyncio.Task[None] | None = None
            self._running = False

            self._initialized = True
            logger.info(
                f"AgentPool initialized: max_size={self._max_size}, "
                f"idle_timeout={self._idle_timeout}"
            )

    def set_loader(self, loader: BaseAgentLoader) -> None:
        """Set the agent loader.

        Args:
            loader: BaseAgentLoader implementation
        """
        self._loader = loader
        logger.info(f"Agent loader set: {type(loader).__name__}")

    @property
    def size(self) -> int:
        """Get current pool size."""
        with self._agent_lock:
            return len(self._agents)

    @property
    def is_full(self) -> bool:
        """Check if pool is at capacity."""
        return self.size >= self._max_size

    def get(self, uuid: str) -> BaseAgent | None:
        """Get an agent by UUID if it exists.

        Args:
            uuid: Agent UUID

        Returns:
            Agent if found, None otherwise
        """
        with self._agent_lock:
            return self._agents.get(uuid)

    async def get_or_create(self, uuid: str) -> BaseAgent:
        """Get existing agent or create new one.

        Args:
            uuid: Agent UUID

        Returns:
            Agent instance

        Raises:
            ValueError: If loader not set or agent config not found
            RuntimeError: If pool is full
        """
        # Check existing first (fast path)
        agent = self.get(uuid)
        if agent and agent.status != AgentStatus.SHUTDOWN:
            return agent

        # Need to create - acquire lock
        with self._agent_lock:
            # Double-check after acquiring lock
            if uuid in self._agents:
                agent = self._agents[uuid]
                if agent.status != AgentStatus.SHUTDOWN:
                    return agent

            # Check pool capacity
            if len(self._agents) >= self._max_size:
                # Try to evict idle agents
                evicted = await self._evict_idle_agents()
                if not evicted and len(self._agents) >= self._max_size:
                    raise RuntimeError(
                        f"Agent pool full ({self._max_size}), cannot create agent"
                    )

            # Create new agent
            if not self._loader:
                raise ValueError("Agent loader not set")

            agent_info = await self._loader.load_agent_info(uuid)
            if not agent_info:
                raise ValueError(f"Agent configuration not found: {uuid}")

            agent = await self._builder.build(agent_info)
            self._agents[uuid] = agent
            logger.info(f"Agent added to pool: {uuid} (pool size: {len(self._agents)})")

            return agent

    async def create_from_info(self, agent_info: AgentInfo) -> BaseAgent:
        """Create an agent directly from AgentInfo.

        Args:
            agent_info: Agent configuration

        Returns:
            Agent instance
        """
        with self._agent_lock:
            if agent_info.uuid in self._agents:
                existing = self._agents[agent_info.uuid]
                if existing.status != AgentStatus.SHUTDOWN:
                    return existing

            if len(self._agents) >= self._max_size:
                await self._evict_idle_agents()
                if len(self._agents) >= self._max_size:
                    raise RuntimeError("Agent pool full")

            agent = await self._builder.build(agent_info)
            self._agents[agent_info.uuid] = agent
            logger.info(
                f"Agent created from info: {agent_info.uuid} "
                f"(pool size: {len(self._agents)})"
            )

            return agent

    async def remove(self, uuid: str) -> bool:
        """Remove an agent from the pool.

        Args:
            uuid: Agent UUID

        Returns:
            True if removed
        """
        with self._agent_lock:
            agent = self._agents.pop(uuid, None)

        if agent:
            await agent.shutdown()
            logger.info(f"Agent removed from pool: {uuid}")
            return True
        return False

    async def reload(self, uuid: str) -> BaseAgent | None:
        """Reload an agent (rebuild from config).

        Args:
            uuid: Agent UUID

        Returns:
            Reloaded agent or None if not found
        """
        if not self._loader:
            raise ValueError("Agent loader not set")

        # Reload config
        agent_info = await self._loader.reload_agent(uuid)
        if not agent_info:
            return None

        with self._agent_lock:
            old_agent = self._agents.get(uuid)
            if old_agent:
                await old_agent.shutdown()

            new_agent = await self._builder.build(agent_info)
            self._agents[uuid] = new_agent
            logger.info(f"Agent reloaded: {uuid}")

            return new_agent

    def list_agents(self) -> list[str]:
        """List all agent UUIDs in pool."""
        with self._agent_lock:
            return list(self._agents.keys())

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all agents."""
        with self._agent_lock:
            return {uuid: agent.stats for uuid, agent in self._agents.items()}

    async def _evict_idle_agents(self) -> int:
        """Evict idle agents that haven't been used recently.

        Returns:
            Number of agents evicted
        """
        now = datetime.utcnow()
        to_evict = []

        # Find idle agents (don't hold lock during shutdown)
        with self._agent_lock:
            for uuid, agent in self._agents.items():
                if agent.status == AgentStatus.READY:
                    last_used = agent.stats.get("last_used_at")
                    if last_used:
                        last_used_dt = datetime.fromisoformat(last_used)
                        if now - last_used_dt > self._idle_timeout:
                            to_evict.append(uuid)

        # Evict outside lock
        for uuid in to_evict:
            await self.remove(uuid)

        if to_evict:
            logger.info(f"Evicted {len(to_evict)} idle agents")

        return len(to_evict)

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._evict_idle_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def start(self) -> None:
        """Start the agent pool (background tasks)."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("AgentPool started")

    async def shutdown(self) -> None:
        """Gracefully shutdown all agents and cleanup."""
        logger.info("AgentPool shutting down...")
        self._running = False

        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Shutdown all agents
        with self._agent_lock:
            agents = list(self._agents.values())
            self._agents.clear()

        for agent in agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent {agent.uuid}: {e}")

        # Cleanup builder resources
        await self._builder.cleanup()

        logger.info("AgentPool shutdown complete")


# Global pool instance
_pool: AgentPool | None = None
_pool_lock = threading.Lock()


def get_agent_pool() -> AgentPool:
    """Get global agent pool instance."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = AgentPool()
        return _pool
