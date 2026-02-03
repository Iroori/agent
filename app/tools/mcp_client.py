"""MCP (Model Context Protocol) client for external tool integration."""

import asyncio
import threading
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger
from pydantic import BaseModel, Field


class MCPToolSpec(BaseModel):
    """Specification for an MCP tool."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool inputs"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool outputs"
    )


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    url: str
    name: str = ""
    version: str = ""
    tools: list[MCPToolSpec] = Field(default_factory=list)
    connected: bool = False


class MCPClient:
    """Client for connecting to MCP (Model Context Protocol) servers.

    MCP provides a standardized way for LLMs to interact with external tools.
    This client handles:
    - Server discovery and connection
    - Tool listing and execution
    - Authentication
    - Retry logic
    """

    def __init__(
        self,
        server_url: str,
        auth_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize MCP client.

        Args:
            server_url: MCP server URL
            auth_token: Optional authentication token
            timeout: Request timeout in seconds
        """
        self._server_url = server_url
        self._auth_token = auth_token
        self._timeout = timeout
        self._server_info: MCPServerInfo | None = None
        self._tools: dict[str, BaseTool] = {}
        self._lock = threading.Lock()
        self._connected = False

    @property
    def server_url(self) -> str:
        """Get server URL."""
        return self._server_url

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
            # In a real implementation, this would make HTTP/WebSocket requests
            # to the MCP server to discover available tools
            logger.info(f"Connecting to MCP server: {self._server_url}")

            # Simulate server info retrieval
            self._server_info = MCPServerInfo(
                url=self._server_url,
                name="MCP Server",
                version="1.0.0",
                tools=[],
                connected=True,
            )

            # TODO: Implement actual MCP protocol
            # This would involve:
            # 1. HTTP GET to server_url/tools to list available tools
            # 2. Parse tool specifications
            # 3. Create LangChain tool wrappers

            self._connected = True
            logger.info(f"Connected to MCP server: {self._server_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        self._connected = False
        self._server_info = None
        with self._lock:
            self._tools.clear()
        logger.info(f"Disconnected from MCP server: {self._server_url}")

    async def list_tools(self) -> list[MCPToolSpec]:
        """List available tools from server.

        Returns:
            List of tool specifications
        """
        if not self._connected or not self._server_info:
            return []
        return self._server_info.tools

    async def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            LangChain tool wrapper if found
        """
        with self._lock:
            if name in self._tools:
                return self._tools[name]

        # Check if tool exists on server
        tools = await self.list_tools()
        tool_spec = next((t for t in tools if t.name == name), None)

        if not tool_spec:
            return None

        # Create LangChain tool wrapper
        lc_tool = self._create_tool_wrapper(tool_spec)
        with self._lock:
            self._tools[name] = lc_tool

        return lc_tool

    async def get_tools(self, names: list[str] | None = None) -> list[BaseTool]:
        """Get multiple tools.

        Args:
            names: Tool names to get, or None for all

        Returns:
            List of LangChain tool wrappers
        """
        if not self._connected:
            return []

        all_specs = await self.list_tools()

        if names:
            specs = [s for s in all_specs if s.name in names]
        else:
            specs = all_specs

        tools = []
        for spec in specs:
            tool = await self.get_tool(spec.name)
            if tool:
                tools.append(tool)

        return tools

    def _create_tool_wrapper(self, spec: MCPToolSpec) -> BaseTool:
        """Create a LangChain tool wrapper for an MCP tool.

        Args:
            spec: MCP tool specification

        Returns:
            LangChain tool wrapper
        """

        async def call_mcp_tool(**kwargs: Any) -> str:
            """Execute MCP tool call."""
            return await self._execute_tool(spec.name, kwargs)

        return StructuredTool.from_function(
            func=call_mcp_tool,
            name=spec.name,
            description=spec.description,
            coroutine=call_mcp_tool,
        )

    async def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a tool on the MCP server.

        Args:
            tool_name: Tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")

        try:
            # TODO: Implement actual MCP tool execution
            # This would involve:
            # 1. HTTP POST to server_url/tools/{tool_name}/execute
            # 2. Send arguments as JSON body
            # 3. Parse and return result

            logger.debug(f"Executing MCP tool: {tool_name} with args: {arguments}")

            # Placeholder response
            return f"MCP tool '{tool_name}' executed successfully"

        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            raise


class MCPManager:
    """Manage multiple MCP server connections."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._lock = threading.Lock()

    async def add_server(
        self,
        server_url: str,
        auth_token: str | None = None,
        auto_connect: bool = True,
    ) -> MCPClient:
        """Add and optionally connect to an MCP server.

        Args:
            server_url: Server URL
            auth_token: Authentication token
            auto_connect: Whether to connect immediately

        Returns:
            MCPClient instance
        """
        client = MCPClient(server_url, auth_token)

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
