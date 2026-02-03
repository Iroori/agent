"""Structured logging configuration using loguru."""

import sys
import threading
from contextvars import ContextVar
from typing import Any

from loguru import logger

from app.core.settings import settings

# Context variables for structured logging
agent_uuid_ctx: ContextVar[str | None] = ContextVar("agent_uuid", default=None)
session_id_ctx: ContextVar[str | None] = ContextVar("session_id", default=None)


def get_thread_id() -> int:
    """Get current thread ID."""
    return threading.current_thread().ident or 0


def structured_format(record: dict[str, Any]) -> str:
    """Format log record with structured context."""
    extra = record.get("extra", {})
    agent_uuid = extra.get("agent_uuid") or agent_uuid_ctx.get()
    session_id = extra.get("session_id") or session_id_ctx.get()
    thread_id = get_thread_id()

    context_parts = []
    if agent_uuid:
        context_parts.append(f"agent={agent_uuid}")
    if session_id:
        context_parts.append(f"session={session_id}")
    context_parts.append(f"thread={thread_id}")

    context_str = " | ".join(context_parts)

    if settings.log_format == "json":
        import json

        log_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "thread_id": thread_id,
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
        }
        if agent_uuid:
            log_data["agent_uuid"] = agent_uuid
        if session_id:
            log_data["session_id"] = session_id

        # Include extra fields
        for key, value in extra.items():
            if key not in ("agent_uuid", "session_id"):
                log_data[key] = value

        return json.dumps(log_data) + "\n"

    return (
        f"<green>{record['time']:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<level>{record['level'].name:8}</level> | "
        f"<cyan>{context_str}</cyan> | "
        f"<level>{record['message']}</level>\n"
    )


def configure_logger() -> None:
    """Configure loguru with structured logging."""
    logger.remove()

    logger.add(
        sys.stderr,
        format=structured_format,
        level=settings.log_level,
        colorize=settings.log_format == "text",
        backtrace=settings.debug,
        diagnose=settings.debug,
    )


def get_agent_logger(agent_uuid: str, session_id: str | None = None) -> "logger":
    """Get a logger bound with agent context."""
    bound_logger = logger.bind(agent_uuid=agent_uuid)
    if session_id:
        bound_logger = bound_logger.bind(session_id=session_id)
    return bound_logger


class TokenUsageTracker:
    """Track token usage per agent/session."""

    def __init__(self) -> None:
        self._usage: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        agent_uuid: str,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record token usage."""
        key = f"{agent_uuid}:{session_id}"
        with self._lock:
            if key not in self._usage:
                self._usage[key] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            self._usage[key]["prompt_tokens"] += prompt_tokens
            self._usage[key]["completion_tokens"] += completion_tokens
            self._usage[key]["total_tokens"] += prompt_tokens + completion_tokens

        logger.info(
            "Token usage recorded",
            agent_uuid=agent_uuid,
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def get_usage(self, agent_uuid: str, session_id: str) -> dict[str, int]:
        """Get token usage for agent/session."""
        key = f"{agent_uuid}:{session_id}"
        with self._lock:
            return self._usage.get(
                key, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ).copy()

    def get_all_usage(self) -> dict[str, dict[str, int]]:
        """Get all token usage data."""
        with self._lock:
            return {k: v.copy() for k, v in self._usage.items()}


# Global token tracker
token_tracker = TokenUsageTracker()

# Configure logger on import
configure_logger()
