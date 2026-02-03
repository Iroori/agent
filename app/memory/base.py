"""Memory persistence abstraction for agent conversations."""

import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="Message role: system, user, assistant, tool")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConversationHistory(BaseModel):
    """Conversation history for an agent session."""

    agent_uuid: str
    session_id: str
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseMemory(ABC):
    """Abstract base class for conversation memory storage.

    Implementations can store conversation history in:
    - In-memory (default, for development)
    - Redis
    - PostgreSQL/MySQL
    - MongoDB
    - File system
    """

    @abstractmethod
    async def save(
        self,
        agent_uuid: str,
        session_id: str,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save conversation messages.

        Args:
            agent_uuid: Agent identifier
            session_id: Session identifier
            messages: List of messages to save
            metadata: Optional metadata to store
        """
        pass

    @abstractmethod
    async def load(
        self, agent_uuid: str, session_id: str
    ) -> ConversationHistory | None:
        """Load conversation history.

        Args:
            agent_uuid: Agent identifier
            session_id: Session identifier

        Returns:
            ConversationHistory if found, None otherwise
        """
        pass

    @abstractmethod
    async def append(
        self, agent_uuid: str, session_id: str, message: Message
    ) -> None:
        """Append a single message to conversation.

        Args:
            agent_uuid: Agent identifier
            session_id: Session identifier
            message: Message to append
        """
        pass

    @abstractmethod
    async def clear(self, agent_uuid: str, session_id: str) -> bool:
        """Clear conversation history.

        Args:
            agent_uuid: Agent identifier
            session_id: Session identifier

        Returns:
            True if cleared, False if not found
        """
        pass

    @abstractmethod
    async def list_sessions(self, agent_uuid: str) -> list[str]:
        """List all sessions for an agent.

        Args:
            agent_uuid: Agent identifier

        Returns:
            List of session IDs
        """
        pass

    async def get_recent_messages(
        self, agent_uuid: str, session_id: str, limit: int = 10
    ) -> list[Message]:
        """Get recent messages from conversation.

        Args:
            agent_uuid: Agent identifier
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        history = await self.load(agent_uuid, session_id)
        if not history:
            return []
        return history.messages[-limit:]


class InMemoryStorage(BaseMemory):
    """Thread-safe in-memory storage implementation.

    Suitable for development and testing.
    For production, use Redis or database-backed storage.
    """

    def __init__(self) -> None:
        self._storage: dict[str, ConversationHistory] = {}
        self._lock = threading.Lock()

    def _make_key(self, agent_uuid: str, session_id: str) -> str:
        """Create storage key from agent UUID and session ID."""
        return f"{agent_uuid}:{session_id}"

    async def save(
        self,
        agent_uuid: str,
        session_id: str,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save conversation messages."""
        key = self._make_key(agent_uuid, session_id)
        now = datetime.utcnow()

        with self._lock:
            if key in self._storage:
                history = self._storage[key]
                history.messages = messages
                history.updated_at = now
                if metadata:
                    history.metadata.update(metadata)
            else:
                self._storage[key] = ConversationHistory(
                    agent_uuid=agent_uuid,
                    session_id=session_id,
                    messages=messages,
                    created_at=now,
                    updated_at=now,
                    metadata=metadata or {},
                )

    async def load(
        self, agent_uuid: str, session_id: str
    ) -> ConversationHistory | None:
        """Load conversation history."""
        key = self._make_key(agent_uuid, session_id)
        with self._lock:
            history = self._storage.get(key)
            if history:
                # Return a copy to prevent external modifications
                return history.model_copy(deep=True)
            return None

    async def append(
        self, agent_uuid: str, session_id: str, message: Message
    ) -> None:
        """Append a single message to conversation."""
        key = self._make_key(agent_uuid, session_id)
        now = datetime.utcnow()

        with self._lock:
            if key not in self._storage:
                self._storage[key] = ConversationHistory(
                    agent_uuid=agent_uuid,
                    session_id=session_id,
                    messages=[],
                    created_at=now,
                    updated_at=now,
                )
            self._storage[key].messages.append(message)
            self._storage[key].updated_at = now

    async def clear(self, agent_uuid: str, session_id: str) -> bool:
        """Clear conversation history."""
        key = self._make_key(agent_uuid, session_id)
        with self._lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False

    async def list_sessions(self, agent_uuid: str) -> list[str]:
        """List all sessions for an agent."""
        prefix = f"{agent_uuid}:"
        with self._lock:
            sessions = [
                key.split(":", 1)[1]
                for key in self._storage.keys()
                if key.startswith(prefix)
            ]
        return sorted(sessions)

    def get_stats(self) -> dict[str, int]:
        """Get storage statistics."""
        with self._lock:
            total_messages = sum(
                len(h.messages) for h in self._storage.values()
            )
            return {
                "total_conversations": len(self._storage),
                "total_messages": total_messages,
            }


# Global memory storage instance (can be replaced with other implementations)
_memory_storage: BaseMemory | None = None
_storage_lock = threading.Lock()


def get_memory_storage() -> BaseMemory:
    """Get global memory storage instance."""
    global _memory_storage
    with _storage_lock:
        if _memory_storage is None:
            _memory_storage = InMemoryStorage()
        return _memory_storage


def set_memory_storage(storage: BaseMemory) -> None:
    """Set global memory storage instance.

    Use this to replace the default in-memory storage with
    Redis, database, or other implementations.
    """
    global _memory_storage
    with _storage_lock:
        _memory_storage = storage
