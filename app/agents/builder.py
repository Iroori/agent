"""Agent builder with tool binding, retry logic, and multi-LLM support."""

import asyncio
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from langchain_core.language_models import BaseChatModel
from loguru import logger

from app.agents.base_agent import AgentType, BaseAgent
from app.agents.model_factory import create_model_from_agent_info
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
    """Build agents from AgentInfo with tool binding and multi-LLM support.

    Handles:
    - Multi-provider LLM creation (GPT, Claude, Ollama, Gemini, Grok, Friendli, Luxia)
    - LLM creation with retry logic
    - Tool binding from registry
    - MCP server connection and tool import
    - Sub-agent tool creation
    """

    def __init__(self) -> None:
        self._mcp_clients: dict[str, MCPClient] = {}
        self._sub_agents: dict[str, BaseAgent] = {}

    @retry_with_backoff(exceptions=(Exception,))
    async def _create_llm(self, agent_info: AgentInfo) -> BaseChatModel:
        """Create LLM instance with retry logic using Model Factory.

        Args:
            agent_info: Agent configuration

        Returns:
            BaseChatModel instance (OpenAI, Claude, Ollama, Gemini, Grok, etc.)
        """
        logger.debug(f"Creating LLM for agent {agent_info.uuid}: {agent_info.model}")
        return create_model_from_agent_info(agent_info, streaming=True)

    async def _bind_tools(
        self,
        agent_info: AgentInfo,
        agent_loader: Any = None,
    ) -> list[Any]:
        """Bind tools from registry, MCP servers, and sub-agents.

        Args:
            agent_info: Agent configuration
            agent_loader: Optional agent loader for sub-agent resolution

        Returns:
            List of LangChain tools
        """
        from app.tools.sub_agent import create_sub_agent_tool

        tools = []
        registry = get_tool_registry()

        # Bind tools from simple tools list
        for tool_name in agent_info.tools:
            tool = registry.get(tool_name)
            if tool:
                tools.append(tool)
                logger.debug(f"Bound tool from registry: {tool_name}")
            else:
                logger.debug(f"Tool not found in registry: {tool_name}")

        # Handle bind_tools with type specification
        for bind_config in agent_info.bind_tools:
            if not bind_config.enabled:
                continue

            if bind_config.type == "registry":
                tool = registry.get(bind_config.name)
                if tool:
                    tools.append(tool)
                    logger.debug(f"Bound registry tool: {bind_config.name}")
            elif bind_config.type == "mcp":
                # MCP tools are handled below when connecting to servers
                logger.debug(f"MCP tool will be bound: {bind_config.name}")

        # Handle detailed tool configs
        for tool_config in agent_info.tool_configs:
            if not tool_config.enabled:
                continue
            tool = registry.get(tool_config.name)
            if tool:
                tools.append(tool)
                logger.debug(f"Bound configured tool: {tool_config.name}")

        # Connect to MCP servers and import tools
        for mcp_config in agent_info.mcp_servers:
            try:
                client = MCPClient(
                    server_url=mcp_config.url,
                    auth_token=mcp_config.auth_token,
                    transport=mcp_config.transport,
                )
                if await client.connect():
                    # Filter tools if specified
                    if mcp_config.tools:
                        mcp_tools = client.get_filtered_tools(mcp_config.tools)
                    else:
                        mcp_tools = await client.get_tools()
                    tools.extend(mcp_tools)
                    self._mcp_clients[mcp_config.url] = client
                    logger.debug(
                        f"Connected to MCP server {mcp_config.url} via {mcp_config.transport}, "
                        f"imported {len(mcp_tools)} tools"
                    )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {mcp_config.url}: {e}")

        # Create sub-agent tools
        for sub_agent_id in agent_info.sub_agent_ids:
            if sub_agent_id in self._sub_agents:
                # Sub-agent already built
                sub_agent = self._sub_agents[sub_agent_id]
                sub_agent_tool = create_sub_agent_tool(sub_agent)
                tools.append(sub_agent_tool)
                logger.debug(f"Bound existing sub-agent tool: {sub_agent_id}")
            elif agent_loader:
                # Try to build sub-agent from loader
                try:
                    sub_agent_info = await agent_loader.load_agent_info(sub_agent_id)
                    if sub_agent_info:
                        sub_agent = await self.build_sub_agent(sub_agent_info)
                        self._sub_agents[sub_agent_id] = sub_agent
                        sub_agent_tool = create_sub_agent_tool(sub_agent)
                        tools.append(sub_agent_tool)
                        logger.debug(f"Built and bound sub-agent tool: {sub_agent_id}")
                except Exception as e:
                    logger.error(f"Failed to build sub-agent {sub_agent_id}: {e}")
            else:
                logger.debug(f"Sub-agent {sub_agent_id} not available (no loader)")

        return tools

    async def build(
        self,
        agent_info: AgentInfo,
        agent_type: AgentType = AgentType.MAIN,
        agent_loader: Any = None,
    ) -> BaseAgent:
        """Build an agent from configuration.

        Args:
            agent_info: Agent configuration
            agent_type: Agent type (MAIN or SUB)
            agent_loader: Optional loader for resolving sub-agents

        Returns:
            Initialized BaseAgent
        """
        logger.debug(f"Building agent: {agent_info.uuid} ({agent_info.name})")

        # Create LLM with retry
        llm = await self._create_llm(agent_info)

        # Bind tools (with agent_loader for sub-agent resolution)
        tools = await self._bind_tools(agent_info, agent_loader)

        # Create agent
        agent = BaseAgent(agent_info, llm, tools, agent_type)

        logger.debug(
            f"Agent built: {agent_info.uuid} with {len(tools)} tools, "
            f"model={agent_info.model}, type={agent_type.value}"
        )

        return agent

    async def build_sub_agent(self, agent_info: AgentInfo) -> BaseAgent:
        """Build a sub-agent.

        Args:
            agent_info: Sub-agent configuration

        Returns:
            Initialized sub-agent
        """
        # Sub-agents don't have nested sub-agents (no loader passed)
        return await self.build(agent_info, AgentType.SUB, None)

    def register_sub_agent(self, agent: BaseAgent) -> None:
        """Register an existing agent as a sub-agent.

        Args:
            agent: Agent to register
        """
        self._sub_agents[agent.uuid] = agent
        logger.debug(f"Registered sub-agent: {agent.uuid}")

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
