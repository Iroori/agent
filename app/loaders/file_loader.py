"""File-based agent loader implementation."""

import asyncio
import json
from pathlib import Path
from typing import Any

import aiofiles
import yaml
from loguru import logger

from app.core.settings import settings
from app.loaders.base import AgentInfo, BaseAgentLoader


class FileAgentLoader(BaseAgentLoader):
    """Load agent configurations from local JSON/YAML files.

    Directory structure:
        agents_config/
            agent-uuid-1.json
            agent-uuid-2.yaml
            agent-uuid-3.yml
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """Initialize file loader.

        Args:
            config_dir: Directory containing agent config files.
                       Defaults to settings.agents_config_dir
        """
        self._config_dir = Path(config_dir or settings.agents_config_dir)
        self._cache: dict[str, AgentInfo] = {}
        self._watch_task: asyncio.Task[None] | None = None
        self._watching = False

    @property
    def config_dir(self) -> Path:
        """Get configuration directory path."""
        return self._config_dir

    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def _find_agent_file(self, uuid: str) -> Path | None:
        """Find agent config file by UUID."""
        for ext in (".json", ".yaml", ".yml"):
            path = self._config_dir / f"{uuid}{ext}"
            if path.exists():
                return path
        return None

    async def _read_file(self, path: Path) -> dict[str, Any]:
        """Read and parse config file."""
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()

        if path.suffix == ".json":
            return json.loads(content)
        else:
            return yaml.safe_load(content)

    async def _write_file(self, path: Path, data: dict[str, Any]) -> None:
        """Write config data to file."""
        if path.suffix == ".json":
            content = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            content = yaml.dump(data, allow_unicode=True, default_flow_style=False)

        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)

    async def load_agent_info(self, uuid: str) -> AgentInfo | None:
        """Load agent configuration by UUID."""
        # Check cache first
        if uuid in self._cache:
            return self._cache[uuid]

        file_path = self._find_agent_file(uuid)
        if not file_path:
            logger.debug(f"Agent config not found: {uuid}")
            return None

        try:
            data = await self._read_file(file_path)
            # Ensure UUID matches
            data["uuid"] = uuid
            agent_info = AgentInfo(**data)
            self._cache[uuid] = agent_info
            logger.info(f"Loaded agent config: {uuid} from {file_path}")
            return agent_info
        except Exception as e:
            logger.error(f"Failed to load agent config {uuid}: {e}")
            return None

    async def list_agents(self) -> list[str]:
        """List all available agent UUIDs."""
        if not self._config_dir.exists():
            return []

        agents = []
        for ext in (".json", ".yaml", ".yml"):
            for path in self._config_dir.glob(f"*{ext}"):
                uuid = path.stem
                if uuid not in agents:
                    agents.append(uuid)

        return sorted(agents)

    async def save_agent_info(self, agent_info: AgentInfo) -> bool:
        """Save agent configuration to file."""
        self._ensure_config_dir()

        # Use JSON by default for new files
        file_path = self._find_agent_file(agent_info.uuid)
        if not file_path:
            file_path = self._config_dir / f"{agent_info.uuid}.json"

        try:
            data = agent_info.model_dump(exclude_none=True)
            await self._write_file(file_path, data)
            self._cache[agent_info.uuid] = agent_info
            logger.info(f"Saved agent config: {agent_info.uuid} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save agent config {agent_info.uuid}: {e}")
            return False

    async def delete_agent(self, uuid: str) -> bool:
        """Delete agent configuration file."""
        file_path = self._find_agent_file(uuid)
        if not file_path:
            return False

        try:
            file_path.unlink()
            self._cache.pop(uuid, None)
            logger.info(f"Deleted agent config: {uuid}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete agent config {uuid}: {e}")
            return False

    async def reload_agent(self, uuid: str) -> AgentInfo | None:
        """Force reload agent configuration, bypassing cache."""
        self._cache.pop(uuid, None)
        return await self.load_agent_info(uuid)

    async def watch_changes(self) -> None:
        """Watch for configuration file changes using watchfiles."""
        try:
            from watchfiles import awatch
        except ImportError:
            logger.warning("watchfiles not installed, file watching disabled")
            return

        if not self._config_dir.exists():
            self._ensure_config_dir()

        self._watching = True
        logger.info(f"Starting file watcher for {self._config_dir}")

        async def _watch() -> None:
            try:
                async for changes in awatch(self._config_dir):
                    if not self._watching:
                        break
                    for change_type, path in changes:
                        path = Path(path)
                        if path.suffix in (".json", ".yaml", ".yml"):
                            uuid = path.stem
                            logger.info(f"Config change detected: {change_type} {uuid}")
                            # Invalidate cache
                            self._cache.pop(uuid, None)
            except asyncio.CancelledError:
                pass

        self._watch_task = asyncio.create_task(_watch())

    async def stop_watching(self) -> None:
        """Stop file watcher."""
        self._watching = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
        logger.info("File watcher stopped")

    def clear_cache(self) -> None:
        """Clear the agent config cache."""
        self._cache.clear()
        logger.debug("Agent config cache cleared")
