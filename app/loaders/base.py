"""Abstract base class for agent loaders."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Tool configuration for an agent."""

    name: str = Field(description="Tool name from registry")
    enabled: bool = Field(default=True)
    config: dict[str, Any] = Field(default_factory=dict)


class BindToolConfig(BaseModel):
    """Tool binding configuration with type specification."""

    name: str = Field(description="Tool name")
    type: Literal["registry", "mcp"] = Field(
        default="registry",
        description="Tool source type: registry (local) or mcp (from MCP server)",
    )
    enabled: bool = Field(default=True)


class MCPServerConfig(BaseModel):
    """MCP server configuration."""

    url: str = Field(description="MCP server URL or command for stdio")
    transport: Literal["sse", "http", "streamable_http", "stdio"] = Field(
        default="sse",
        description="Transport type for MCP connection",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names to import (empty = all)",
    )
    auth_token: str | None = Field(default=None, description="Authentication token")


class AgentInfo(BaseModel):
    """Agent configuration and metadata."""

    uuid: str = Field(description="Unique agent identifier")
    name: str = Field(description="Human-readable agent name")
    model: str = Field(default="gpt-4o", description="LLM model to use")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt for the agent",
    )
    tools: list[str] = Field(
        default_factory=list, description="Tool names to bind to this agent"
    )
    bind_tools: list[BindToolConfig] = Field(
        default_factory=list,
        description="Detailed tool bindings with type specification",
    )
    tool_configs: list[ToolConfig] = Field(
        default_factory=list, description="Detailed tool configurations"
    )
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list, description="MCP servers to connect"
    )
    sub_agent_ids: list[str] = Field(
        default_factory=list,
        description="Sub-agent UUIDs to bind as tools",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        extra = "allow"


class BaseAgentLoader(ABC):
    """Abstract base class for loading agent configurations.

    Implementations can load agent info from various sources:
    - Local files (JSON/YAML)
    - Database
    - Remote configuration service
    - Environment variables
    """

    @abstractmethod
    async def load_agent_info(self, uuid: str) -> AgentInfo | None:
        """Load agent configuration by UUID.

        Args:
            uuid: Unique agent identifier

        Returns:
            AgentInfo if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_agents(self) -> list[str]:
        """List all available agent UUIDs.

        Returns:
            List of agent UUIDs
        """
        pass

    @abstractmethod
    async def save_agent_info(self, agent_info: AgentInfo) -> bool:
        """Save or update agent configuration.

        Args:
            agent_info: Agent configuration to save

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_agent(self, uuid: str) -> bool:
        """Delete an agent configuration.

        Args:
            uuid: Agent UUID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    async def reload_agent(self, uuid: str) -> AgentInfo | None:
        """Force reload agent configuration from source.

        Default implementation just calls load_agent_info.
        Subclasses can override to clear caches.
        """
        return await self.load_agent_info(uuid)

    async def watch_changes(self) -> None:
        """Watch for configuration changes.

        Default implementation does nothing.
        Subclasses can override to implement file watching, etc.
        """
        pass

    async def stop_watching(self) -> None:
        """Stop watching for changes.

        Default implementation does nothing.
        """
        pass
