"""Agent builder with tool binding and retry logic."""

import asyncio
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from langchain_openai import ChatOpenAI
from loguru import logger

from app.agents.base_agent import BaseAgent
from app.core.settings import settings
from app.loaders.base import AgentInfo
from app.tools.mcp_client import MCPClient
from app.tools.registry import get_tool_registry

P = ParamSpec("P")
R = TypeVar("R")


def retry_with_backoff(
    max_retries: int | None = None,
    base_delay: float | None = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum retry attempts (defaults to settings)
        base_delay: Base delay in seconds (defaults to settings)
        exceptions: Exceptions to catch and retry

    Returns:
        Decorated function with retry logic
    """
    _max_retries = max_retries or settings.max_retries
    _base_delay = base_delay or settings.retry_base_delay

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            for attempt in range(_max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < _max_retries - 1:
                        delay = _base_delay * (2**attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{_max_retries} after {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {_max_retries} retries failed: {e}")
            raise last_exception  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            for attempt in range(_max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < _max_retries - 1:
                        delay = _base_delay * (2**attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{_max_retries} after {delay}s: {e}"
                        )
                        import time

                        time.sleep(delay)
                    else:
                        logger.error(f"All {_max_retries} retries failed: {e}")
            raise last_exception  # type: ignore

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class AgentBuilder:
    """Build agents from AgentInfo with tool binding and LLM initialization.

    Handles:
    - LLM creation with retry logic
    - Tool binding from registry
    - MCP server connection and tool import
    """

    def __init__(self) -> None:
        self._mcp_clients: dict[str, MCPClient] = {}

    @retry_with_backoff(exceptions=(Exception,))
    async def _create_llm(self, agent_info: AgentInfo) -> ChatOpenAI:
        """Create LLM instance with retry logic.

        Args:
            agent_info: Agent configuration

        Returns:
            ChatOpenAI instance
        """
        logger.debug(f"Creating LLM for agent {agent_info.uuid}: {agent_info.model}")

        return ChatOpenAI(
            model=agent_info.model,
            temperature=agent_info.temperature,
            max_tokens=agent_info.max_tokens,
            api_key=settings.openai_api_key,
            streaming=True,
        )

    async def _bind_tools(self, agent_info: AgentInfo) -> list[Any]:
        """Bind tools from registry and MCP servers.

        Args:
            agent_info: Agent configuration

        Returns:
            List of LangChain tools
        """
        tools = []
        registry = get_tool_registry()

        # Bind tools from registry
        for tool_name in agent_info.tools:
            tool = registry.get(tool_name)
            if tool:
                tools.append(tool)
                logger.debug(f"Bound tool from registry: {tool_name}")
            else:
                logger.warning(f"Tool not found in registry: {tool_name}")

        # Handle detailed tool configs
        for tool_config in agent_info.tool_configs:
            if not tool_config.enabled:
                continue
            tool = registry.get(tool_config.name)
            if tool:
                # Could apply config to tool here if needed
                tools.append(tool)
                logger.debug(f"Bound configured tool: {tool_config.name}")

        # Connect to MCP servers and import tools
        for mcp_config in agent_info.mcp_servers:
            try:
                client = MCPClient(mcp_config.url, mcp_config.auth_token)
                if await client.connect():
                    mcp_tools = await client.get_tools(mcp_config.tools or None)
                    tools.extend(mcp_tools)
                    self._mcp_clients[mcp_config.url] = client
                    logger.info(
                        f"Connected to MCP server {mcp_config.url}, "
                        f"imported {len(mcp_tools)} tools"
                    )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {mcp_config.url}: {e}")

        return tools

    async def build(self, agent_info: AgentInfo) -> BaseAgent:
        """Build an agent from configuration.

        Args:
            agent_info: Agent configuration

        Returns:
            Initialized BaseAgent
        """
        logger.info(f"Building agent: {agent_info.uuid} ({agent_info.name})")

        # Create LLM with retry
        llm = await self._create_llm(agent_info)

        # Bind tools
        tools = await self._bind_tools(agent_info)

        # Create agent
        agent = BaseAgent(agent_info, llm, tools)

        logger.info(
            f"Agent built: {agent_info.uuid} with {len(tools)} tools, "
            f"model={agent_info.model}"
        )

        return agent

    async def rebuild(self, agent: BaseAgent) -> BaseAgent:
        """Rebuild an agent (e.g., after config change).

        Args:
            agent: Existing agent to rebuild

        Returns:
            New agent instance
        """
        await agent.shutdown()
        return await self.build(agent.info)

    async def cleanup(self) -> None:
        """Clean up resources (MCP connections, etc.)."""
        for url, client in self._mcp_clients.items():
            try:
                await client.disconnect()
                logger.debug(f"Disconnected MCP client: {url}")
            except Exception as e:
                logger.error(f"Error disconnecting MCP client {url}: {e}")
        self._mcp_clients.clear()


# Global builder instance
_builder: AgentBuilder | None = None


def get_agent_builder() -> AgentBuilder:
    """Get global agent builder instance."""
    global _builder
    if _builder is None:
        _builder = AgentBuilder()
    return _builder
