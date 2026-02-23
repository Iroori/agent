"""Base agent entity wrapping LangGraph ReAct agent."""

import asyncio
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, Field

from app.core.callback_handler import UniversalToolCallbackHandler
from app.core.logger import get_agent_logger, token_tracker
from app.loaders.base import AgentInfo
from app.memory.base import Message, get_memory_storage


class AgentStatus(str, Enum):
    """Agent lifecycle status."""

    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentResponse(BaseModel):
    """Response from agent invocation."""

    content: str
    session_id: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentType(str, Enum):
    """Agent type classification."""

    MAIN = "main"
    SUB = "sub"


class BaseAgent:
    """LangGraph ReAct agent wrapper with observability and thread-safety.

    Provides:
    - Async invocation with streaming support via LangGraph
    - Token usage tracking
    - Session-based conversation memory
    - Structured logging
    - ReAct pattern for tool usage
    """

    def __init__(
        self,
        agent_info: AgentInfo,
        llm: Any,
        tools: list[BaseTool] | None = None,
        agent_type: AgentType = AgentType.MAIN,
    ) -> None:
        """Initialize base agent.

        Args:
            agent_info: Agent configuration
            llm: LangChain LLM instance
            tools: List of bound tools
            agent_type: Agent type (main or sub)
        """
        self._info = agent_info
        self._llm = llm
        self._tools = tools or []
        self._agent_type = agent_type
        self._status = AgentStatus.INITIALIZING
        self._lock = threading.Lock()
        self._active_sessions: set[str] = set()
        self._created_at = datetime.utcnow()
        self._last_used_at: datetime | None = None
        self._invocation_count = 0
        self._total_tokens = TokenUsage()

        # Create LangGraph ReAct agent
        self._graph = self._create_react_graph()

        self._status = AgentStatus.READY
        self._logger = get_agent_logger(self.uuid)
        self._logger.info(f"Agent initialized: {self._info.name} (type={agent_type.value})")

    def _create_react_graph(self) -> Any:
        """Create LangGraph ReAct agent.

        Returns:
            Compiled LangGraph graph
        """
        if self._tools:
            # Create ReAct agent with tools using LangGraph
            return create_react_agent(
                model=self._llm,
                tools=self._tools,
                prompt=self._info.system_prompt,
            )
        else:
            # No tools - create simple ReAct agent without tools
            return create_react_agent(
                model=self._llm,
                tools=[],
                prompt=self._info.system_prompt,
            )

    @property
    def uuid(self) -> str:
        """Get agent UUID."""
        return self._info.uuid

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._info.name

    @property
    def info(self) -> AgentInfo:
        """Get agent info."""
        return self._info

    @property
    def agent_type(self) -> AgentType:
        """Get agent type."""
        return self._agent_type

    @property
    def status(self) -> AgentStatus:
        """Get current status."""
        with self._lock:
            return self._status

    @property
    def is_ready(self) -> bool:
        """Check if agent is ready for invocation."""
        return self.status == AgentStatus.READY

    @property
    def tools(self) -> list[BaseTool]:
        """Get agent tools."""
        return self._tools

    @property
    def stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        with self._lock:
            return {
                "uuid": self.uuid,
                "name": self.name,
                "status": self._status.value,
                "agent_type": self._agent_type.value,
                "created_at": self._created_at.isoformat(),
                "last_used_at": (
                    self._last_used_at.isoformat() if self._last_used_at else None
                ),
                "invocation_count": self._invocation_count,
                "active_sessions": len(self._active_sessions),
                "total_tokens": self._total_tokens.model_dump(),
                "tools_count": len(self._tools),
            }

    def _convert_to_langchain_messages(
        self, messages: list[Message]
    ) -> list[BaseMessage]:
        """Convert internal messages to LangChain format."""
        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
        return lc_messages

    async def invoke(
        self,
        input_text: str,
        session_id: str | None = None,
    ) -> AgentResponse:
        """Invoke agent with input text using LangGraph ReAct pattern.

        Args:
            input_text: User input
            session_id: Session ID for conversation continuity

        Returns:
            AgentResponse with result
        """
        session_id = session_id or str(uuid.uuid4())
        agent_logger = get_agent_logger(self.uuid, session_id)

        with self._lock:
            if self._status != AgentStatus.READY:
                raise RuntimeError(f"Agent not ready: {self._status}")
            self._status = AgentStatus.BUSY
            self._active_sessions.add(session_id)

        try:
            agent_logger.debug(f"Invoking with input: {input_text[:100]}...")

            # Load conversation history
            memory = get_memory_storage()
            history = await memory.load(self.uuid, session_id)
            chat_history = (
                self._convert_to_langchain_messages(history.messages) if history else []
            )

            # Build messages for LangGraph
            messages = chat_history + [HumanMessage(content=input_text)]

            # Build callback handler bound to this session
            from app.api.websocket import manager as ws_manager  # local import avoids circular dep

            callback_handler = UniversalToolCallbackHandler(
                agent_uuid=self.uuid,
                session_id=session_id,
                ws_manager=ws_manager,
            )

            # Invoke LangGraph ReAct agent with observability callbacks
            result = await self._graph.ainvoke(
                {"messages": messages},
                config={"callbacks": [callback_handler]},
            )

            # Extract output from result
            output = ""
            token_usage = TokenUsage()

            if "messages" in result:
                # Get the last AI message
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        output = msg.content
                        # Extract token usage from response metadata
                        if hasattr(msg, "response_metadata") and msg.response_metadata:
                            usage = msg.response_metadata.get("token_usage", {})
                            token_usage = TokenUsage(
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                                total_tokens=usage.get("total_tokens", 0),
                            )
                        break

            # Track tokens
            if token_usage.total_tokens > 0:
                token_tracker.record(
                    self.uuid,
                    session_id,
                    token_usage.prompt_tokens,
                    token_usage.completion_tokens,
                )

            # Save to memory
            await memory.append(
                self.uuid, session_id, Message(role="user", content=input_text)
            )
            await memory.append(
                self.uuid, session_id, Message(role="assistant", content=output)
            )

            with self._lock:
                self._invocation_count += 1
                self._last_used_at = datetime.utcnow()
                self._total_tokens.prompt_tokens += token_usage.prompt_tokens
                self._total_tokens.completion_tokens += token_usage.completion_tokens
                self._total_tokens.total_tokens += token_usage.total_tokens

            agent_logger.debug(f"Invocation complete, tokens: {token_usage.total_tokens}")

            return AgentResponse(
                content=output,
                session_id=session_id,
                token_usage=token_usage,
            )

        except Exception as e:
            agent_logger.error(f"Invocation failed: {e}")
            raise

        finally:
            with self._lock:
                self._active_sessions.discard(session_id)
                if not self._active_sessions:
                    self._status = AgentStatus.READY

    async def stream(
        self,
        input_text: str,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream agent response using LangGraph ReAct pattern.

        Args:
            input_text: User input
            session_id: Session ID for conversation continuity

        Yields:
            Response chunks
        """
        session_id = session_id or str(uuid.uuid4())
        agent_logger = get_agent_logger(self.uuid, session_id)

        with self._lock:
            if self._status != AgentStatus.READY:
                raise RuntimeError(f"Agent not ready: {self._status}")
            self._status = AgentStatus.BUSY
            self._active_sessions.add(session_id)

        full_response = ""

        try:
            agent_logger.debug(f"Streaming with input: {input_text[:100]}...")

            # Load conversation history
            memory = get_memory_storage()
            history = await memory.load(self.uuid, session_id)
            chat_history = (
                self._convert_to_langchain_messages(history.messages) if history else []
            )

            # Build messages for LangGraph
            messages = chat_history + [HumanMessage(content=input_text)]

            # Build callback handler bound to this session
            from app.api.websocket import manager as ws_manager  # local import avoids circular dep

            callback_handler = UniversalToolCallbackHandler(
                agent_uuid=self.uuid,
                session_id=session_id,
                ws_manager=ws_manager,
            )

            # Stream response from LangGraph ReAct agent with observability callbacks
            async for event in self._graph.astream_events(
                {"messages": messages},
                config={"callbacks": [callback_handler]},
                version="v2",
            ):
                kind = event.get("event", "")

                # Stream AI message chunks
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and isinstance(chunk, AIMessageChunk):
                        content = chunk.content
                        if content:
                            yield content
                            full_response += content

            # Save to memory
            await memory.append(
                self.uuid, session_id, Message(role="user", content=input_text)
            )
            await memory.append(
                self.uuid, session_id, Message(role="assistant", content=full_response)
            )

            with self._lock:
                self._invocation_count += 1
                self._last_used_at = datetime.utcnow()

            agent_logger.debug("Streaming complete")

        except Exception as e:
            agent_logger.error(f"Streaming failed: {e}")
            raise

        finally:
            with self._lock:
                self._active_sessions.discard(session_id)
                if not self._active_sessions:
                    self._status = AgentStatus.READY

    async def shutdown(self) -> None:
        """Shutdown agent gracefully."""
        with self._lock:
            self._status = AgentStatus.SHUTDOWN

        self._logger.info(f"Agent shutdown: {self.name}")
