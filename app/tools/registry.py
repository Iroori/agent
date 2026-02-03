"""Async tool registry with decorator-based registration."""

import asyncio
import inspect
import threading
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

P = ParamSpec("P")
R = TypeVar("R")


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""

    name: str
    description: str
    is_async: bool = False
    category: str = "general"
    requires_confirmation: bool = False


class ToolRegistry:
    """Thread-safe registry for async tools.

    Supports both decorator-based and programmatic registration.
    """

    _instance: "ToolRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ToolRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._tools: dict[str, BaseTool] = {}
                cls._instance._metadata: dict[str, ToolMetadata] = {}
                cls._instance._tool_lock = threading.Lock()
        return cls._instance

    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        category: str = "general",
        requires_confirmation: bool = False,
        args_schema: type[BaseModel] | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to register a function as a tool.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            category: Tool category for organization
            requires_confirmation: Whether tool requires user confirmation
            args_schema: Pydantic model for argument validation

        Example:
            @registry.register(name="search", description="Search the web")
            async def search_web(query: str) -> str:
                ...
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or "No description"
            is_async = asyncio.iscoroutinefunction(func)

            # Wrap sync functions to be async-compatible
            if not is_async:

                @wraps(func)
                async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, lambda: func(*args, **kwargs)
                    )

                tool_func = async_wrapper
            else:
                tool_func = func

            # Create LangChain tool
            lc_tool = StructuredTool.from_function(
                func=tool_func,
                name=tool_name,
                description=tool_description,
                args_schema=args_schema,
                coroutine=tool_func if is_async else None,
            )

            # Store metadata
            metadata = ToolMetadata(
                name=tool_name,
                description=tool_description,
                is_async=is_async,
                category=category,
                requires_confirmation=requires_confirmation,
            )

            with self._tool_lock:
                self._tools[tool_name] = lc_tool
                self._metadata[tool_name] = metadata

            logger.debug(f"Registered tool: {tool_name} (async={is_async})")
            return func

        return decorator

    def register_tool(
        self,
        tool: BaseTool,
        category: str = "general",
        requires_confirmation: bool = False,
    ) -> None:
        """Register a pre-built LangChain tool.

        Args:
            tool: LangChain BaseTool instance
            category: Tool category
            requires_confirmation: Whether tool requires confirmation
        """
        metadata = ToolMetadata(
            name=tool.name,
            description=tool.description,
            is_async=asyncio.iscoroutinefunction(tool._run),
            category=category,
            requires_confirmation=requires_confirmation,
        )

        with self._tool_lock:
            self._tools[tool.name] = tool
            self._metadata[tool.name] = metadata

        logger.debug(f"Registered external tool: {tool.name}")

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        with self._tool_lock:
            return self._tools.get(name)

    def get_tools(self, names: list[str]) -> list[BaseTool]:
        """Get multiple tools by name."""
        tools = []
        with self._tool_lock:
            for name in names:
                if name in self._tools:
                    tools.append(self._tools[name])
                else:
                    logger.warning(f"Tool not found: {name}")
        return tools

    def get_all(self) -> list[BaseTool]:
        """Get all registered tools."""
        with self._tool_lock:
            return list(self._tools.values())

    def get_by_category(self, category: str) -> list[BaseTool]:
        """Get tools by category."""
        tools = []
        with self._tool_lock:
            for name, metadata in self._metadata.items():
                if metadata.category == category:
                    tools.append(self._tools[name])
        return tools

    def get_metadata(self, name: str) -> ToolMetadata | None:
        """Get tool metadata."""
        with self._tool_lock:
            return self._metadata.get(name)

    def list_tools(self) -> list[str]:
        """List all tool names."""
        with self._tool_lock:
            return list(self._tools.keys())

    def list_categories(self) -> list[str]:
        """List all tool categories."""
        with self._tool_lock:
            categories = set(m.category for m in self._metadata.values())
        return sorted(categories)

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        with self._tool_lock:
            if name in self._tools:
                del self._tools[name]
                del self._metadata[name]
                logger.debug(f"Unregistered tool: {name}")
                return True
            return False

    def clear(self) -> None:
        """Clear all registered tools."""
        with self._tool_lock:
            self._tools.clear()
            self._metadata.clear()
        logger.debug("Cleared all tools")


# Global registry instance
_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ToolRegistry()
        return _registry


# Convenience decorator using global registry
def tool(
    name: str | None = None,
    description: str | None = None,
    category: str = "general",
    requires_confirmation: bool = False,
    args_schema: type[BaseModel] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to register a tool with the global registry.

    Example:
        @tool(name="calculator", description="Perform calculations")
        async def calculate(expression: str) -> str:
            return str(eval(expression))
    """
    return get_tool_registry().register(
        name=name,
        description=description,
        category=category,
        requires_confirmation=requires_confirmation,
        args_schema=args_schema,
    )
