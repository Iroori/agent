"""API-based agent loader implementation for SeedAI API integration."""

import asyncio
from typing import Any

import httpx
from loguru import logger

from app.core.settings import settings
from app.loaders.base import AgentInfo, BaseAgentLoader


class SeedAIAPILoader(BaseAgentLoader):
    """Load agent configurations from SeedAI API.

    Fetches agent configurations from a remote API endpoint.
    Supports caching and periodic refresh.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        cache_ttl: int = 300,
    ) -> None:
        """Initialize API loader.

        Args:
            api_url: SeedAI API base URL (defaults to settings)
            api_key: API authentication key (defaults to settings)
            cache_ttl: Cache time-to-live in seconds
        """
        self._api_url = (api_url or settings.seedai_api_url).rstrip("/")
        self._api_key = api_key or settings.seedai_api_key
        self._cache: dict[str, tuple[AgentInfo, float]] = {}
        self._cache_ttl = cache_ttl
        self._agent_list_cache: tuple[list[str], float] | None = None

        if not self._api_url:
            logger.warning("SeedAI API URL not configured")

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _is_cache_valid(self, cache_time: float) -> bool:
        """Check if cached data is still valid."""
        import time
        return (time.time() - cache_time) < self._cache_ttl

    async def load_agent_info(self, uuid: str) -> AgentInfo | None:
        """Load agent configuration from API.

        Args:
            uuid: Agent UUID

        Returns:
            AgentInfo if found, None otherwise
        """
        import time

        # Check cache first
        if uuid in self._cache:
            agent_info, cache_time = self._cache[uuid]
            if self._is_cache_valid(cache_time):
                logger.debug(f"Using cached agent config: {uuid}")
                return agent_info

        if not self._api_url:
            logger.error("SeedAI API URL not configured")
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self._api_url}/agents/{uuid}",
                    headers=self._get_headers(),
                )

                if response.status_code == 404:
                    logger.debug(f"Agent not found in API: {uuid}")
                    return None

                response.raise_for_status()
                data = response.json()

                # Handle nested data structure if present
                if "data" in data:
                    data = data["data"]

                # Ensure UUID is set
                data["uuid"] = uuid
                agent_info = AgentInfo(**data)

                # Cache the result
                self._cache[uuid] = (agent_info, time.time())
                logger.debug(f"Loaded agent config from API: {uuid}")

                return agent_info

        except httpx.HTTPStatusError as e:
            logger.error(f"API error loading agent {uuid}: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Network error loading agent {uuid}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load agent {uuid} from API: {e}")
            return None

    async def list_agents(self) -> list[str]:
        """List all available agent UUIDs from API."""
        import time

        # Check cache
        if self._agent_list_cache:
            agent_list, cache_time = self._agent_list_cache
            if self._is_cache_valid(cache_time):
                return agent_list

        if not self._api_url:
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self._api_url}/agents",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()

                # Handle different response formats
                if isinstance(data, list):
                    agent_ids = data
                elif "data" in data:
                    agent_ids = data["data"]
                elif "agents" in data:
                    agent_ids = data["agents"]
                else:
                    agent_ids = []

                # Extract UUIDs if objects are returned
                if agent_ids and isinstance(agent_ids[0], dict):
                    agent_ids = [a.get("uuid") or a.get("id") for a in agent_ids if a]

                self._agent_list_cache = (agent_ids, time.time())
                return agent_ids

        except Exception as e:
            logger.error(f"Failed to list agents from API: {e}")
            return []

    async def save_agent_info(self, agent_info: AgentInfo) -> bool:
        """Save agent configuration to API.

        Args:
            agent_info: Agent configuration

        Returns:
            True if successful
        """
        import time

        if not self._api_url:
            logger.error("SeedAI API URL not configured")
            return False

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                data = agent_info.model_dump(exclude_none=True)

                # Try PUT first (update)
                response = await client.put(
                    f"{self._api_url}/agents/{agent_info.uuid}",
                    headers=self._get_headers(),
                    json=data,
                )

                if response.status_code == 404:
                    # Agent doesn't exist, try POST (create)
                    response = await client.post(
                        f"{self._api_url}/agents",
                        headers=self._get_headers(),
                        json=data,
                    )

                response.raise_for_status()

                # Update cache
                self._cache[agent_info.uuid] = (agent_info, time.time())
                self._agent_list_cache = None  # Invalidate list cache

                logger.debug(f"Saved agent config to API: {agent_info.uuid}")
                return True

        except Exception as e:
            logger.error(f"Failed to save agent {agent_info.uuid} to API: {e}")
            return False

    async def delete_agent(self, uuid: str) -> bool:
        """Delete agent configuration from API.

        Args:
            uuid: Agent UUID

        Returns:
            True if deleted
        """
        if not self._api_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self._api_url}/agents/{uuid}",
                    headers=self._get_headers(),
                )

                if response.status_code == 404:
                    return False

                response.raise_for_status()

                # Clear from cache
                self._cache.pop(uuid, None)
                self._agent_list_cache = None

                logger.debug(f"Deleted agent from API: {uuid}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete agent {uuid} from API: {e}")
            return False

    async def reload_agent(self, uuid: str) -> AgentInfo | None:
        """Force reload agent configuration from API.

        Bypasses cache to get fresh data.
        """
        # Clear cache entry
        self._cache.pop(uuid, None)
        return await self.load_agent_info(uuid)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._agent_list_cache = None
        logger.debug("API loader cache cleared")

    async def health_check(self) -> bool:
        """Check if API is reachable.

        Returns:
            True if API is healthy
        """
        if not self._api_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._api_url}/health",
                    headers=self._get_headers(),
                )
                return response.status_code == 200
        except Exception:
            return False
