"""MCP (Model Context Protocol) client using langchain-mcp-adapters."""

import asyncio
import threading
from typing import Any, Literal

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field


class MCPToolSpec(BaseModel):
    """Specification for an MCP tool."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool inputs"
    )


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    url: str
    transport: Literal["sse", "http", "streamable_http", "stdio"] = "sse"
    name: str = ""
    version: str = ""
    tools: list[MCPToolSpec] = Field(default_factory=list)
    connected: bool = False


class MCPClient:
    """Client for connecting to MCP servers using langchain-mcp-adapters.

    Supports multiple transport types:
    - sse: Server-Sent Events
    - http: HTTP transport
    - streamable_http: Streamable HTTP transport
    - stdio: Standard I/O (for local processes)
    """

    def __init__(
        self,
        server_url: str,
        auth_token: str | None = None,
        transport: Literal["sse", "http", "streamable_http", "stdio"] = "sse",
        timeout: float = 30.0,
    ) -> None:
        """Initialize MCP client.

        Args:
            server_url: MCP server URL or command for stdio
            auth_token: Optional authentication token
            transport: Transport type (sse, http, streamable_http, stdio)
            timeout: Request timeout in seconds
        """
        self._server_url = server_url
        self._auth_token = auth_token
        self._transport = transport
        self._timeout = timeout
        self._server_info: MCPServerInfo | None = None
        self._tools: dict[str, BaseTool] = {}
        self._lock = threading.Lock()
        self._connected = False
        self._mcp_client: Any = None
        self._session: Any = None

    @property
    def server_url(self) -> str:
        """Get server URL."""
        return self._server_url

    @property
    def transport(self) -> str:
        """Get transport type."""
        return self._transport

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    async def connect(self) -> bool:
        """Connect to MCP server and discover tools.

        Returns:
            True if connection successful
        """
        try:
            logger.debug(f"Connecting to MCP server: {self._server_url} via {self._transport}")

            if self._transport == "sse":
                await self._connect_sse()
            elif self._transport in ("http", "streamable_http"):
                await self._connect_http()
            elif self._transport == "stdio":
                await self._connect_stdio()
            else:
                raise ValueError(f"Unknown transport: {self._transport}")

            self._connected = True
            logger.debug(f"Connected to MCP server: {self._server_url}")
            return True

        except ImportError as e:
            logger.error(f"MCP adapter not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self._connected = False
            return False

    async def _connect_sse(self) -> None:
        """Connect using SSE transport."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            headers = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"

            self._mcp_client = MultiServerMCPClient(
                {
                    "mcp_server": {
                        "url": self._server_url,
                        "transport": "sse",
                        "headers": headers if headers else None,
                    }
                }
            )
            await self._mcp_client.__aenter__()
            self._tools = {tool.name: tool for tool in self._mcp_client.get_tools()}

        except ImportError:
            # Fallback to basic SSE client
            from langchain_mcp_adapters.tools import load_mcp_tools
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            headers = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"

            async with sse_client(self._server_url, headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    self._session = session
                    await session.initialize()

                    tools = await load_mcp_tools(session)
                    self._tools = {tool.name: tool for tool in tools}

    async def _connect_http(self) -> None:
        """Connect using HTTP transport."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        transport_type = "streamable_http" if self._transport == "streamable_http" else "http"

        headers = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        self._mcp_client = MultiServerMCPClient(
            {
                "mcp_server": {
                    "url": self._server_url,
                    "transport": transport_type,
                    "headers": headers if headers else None,
                }
            }
        )
        await self._mcp_client.__aenter__()
        self._tools = {tool.name: tool for tool in self._mcp_client.get_tools()}

    async def _connect_stdio(self) -> None:
        """Connect using stdio transport."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # For stdio, server_url should be the command to run
        command_parts = self._server_url.split()
        command = command_parts[0]
        args = command_parts[1:] if len(command_parts) > 1 else []

        self._mcp_client = MultiServerMCPClient(
            {
                "mcp_server": {
                    "command": command,
                    "args": args,
                    "transport": "stdio",
                }
            }
        )
        await self._mcp_client.__aenter__()
        self._tools = {tool.name: tool for tool in self._mcp_client.get_tools()}

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if self._mcp_client:
            try:
                await self._mcp_client.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error during MCP client cleanup: {e}")

        self._connected = False
        self._mcp_client = None
        self._session = None
        with self._lock:
            self._tools.clear()
        logger.debug(f"Disconnected from MCP server: {self._server_url}")

    async def list_tools(self) -> list[MCPToolSpec]:
        """List available tools from server.

        Returns:
            List of tool specifications
        """
        if not self._connected:
            return []

        specs = []
        with self._lock:
            for name, tool in self._tools.items():
                specs.append(
                    MCPToolSpec(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.args_schema.model_json_schema() if tool.args_schema else {},
                    )
                )
        return specs

    async def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            LangChain tool if found
        """
        with self._lock:
            return self._tools.get(name)

    async def get_tools(self, names: list[str] | None = None) -> list[BaseTool]:
        """Get multiple tools.

        Args:
            names: Tool names to get, or None for all

        Returns:
            List of LangChain tools
        """
        if not self._connected:
            return []

        with self._lock:
            if names:
                return [self._tools[name] for name in names if name in self._tools]
            return list(self._tools.values())

    def get_filtered_tools(self, tool_names: list[str]) -> list[BaseTool]:
        """Get tools filtered by name list.

        Args:
            tool_names: List of tool names to include

        Returns:
            Filtered list of tools
        """
        with self._lock:
            return [
                tool for name, tool in self._tools.items()
                if name in tool_names
            ]


class MCPManager:
    """Manage multiple MCP server connections."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._lock = threading.Lock()

    async def add_server(
        self,
        server_url: str,
        auth_token: str | None = None,
        transport: Literal["sse", "http", "streamable_http", "stdio"] = "sse",
        auto_connect: bool = True,
    ) -> MCPClient:
        """Add and optionally connect to an MCP server.

        Args:
            server_url: Server URL or command
            auth_token: Authentication token
            transport: Transport type
            auto_connect: Whether to connect immediately

        Returns:
            MCPClient instance
        """
        client = MCPClient(server_url, auth_token, transport)

        with self._lock:
            self._clients[server_url] = client

        if auto_connect:
            await client.connect()

        return client

    async def remove_server(self, server_url: str) -> bool:
        """Remove and disconnect from an MCP server.

        Args:
            server_url: Server URL

        Returns:
            True if removed
        """
        with self._lock:
            client = self._clients.pop(server_url, None)

        if client:
            await client.disconnect()
            return True
        return False

    def get_client(self, server_url: str) -> MCPClient | None:
        """Get client for a server URL."""
        with self._lock:
            return self._clients.get(server_url)

    async def get_all_tools(self) -> list[BaseTool]:
        """Get all tools from all connected servers."""
        all_tools = []
        with self._lock:
            clients = list(self._clients.values())

        for client in clients:
            if client.is_connected:
                tools = await client.get_tools()
                all_tools.extend(tools)

        return all_tools

    async def shutdown(self) -> None:
        """Disconnect from all servers."""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()

        for client in clients:
            await client.disconnect()


# Global MCP manager instance
_mcp_manager: MCPManager | None = None
_manager_lock = threading.Lock()


def get_mcp_manager() -> MCPManager:
    """Get global MCP manager instance."""
    global _mcp_manager
    with _manager_lock:
        if _mcp_manager is None:
            _mcp_manager = MCPManager()
        return _mcp_manager
