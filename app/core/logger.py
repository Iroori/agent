"""Centralized logging configuration using loguru.

All modules must use get_logger() or the shared logger from this module.
Direct use of print() is strictly prohibited inside agent components.
"""

import re
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.settings import settings

# Context variables for structured logging
agent_uuid_ctx: ContextVar[str | None] = ContextVar("agent_uuid", default=None)
session_id_ctx: ContextVar[str | None] = ContextVar("session_id", default=None)

# ──────────────────────────────────────────────────────────────
# Sensitive Information Masking
# ──────────────────────────────────────────────────────────────

_SENSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # OpenAI / generic sk- keys
    (re.compile(r"(sk-[A-Za-z0-9]{10,})", re.IGNORECASE), "sk-***MASKED***"),
    # Bearer tokens
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-_\.]+", re.IGNORECASE), r"\1***MASKED***"),
    # Generic api_key / apikey / api-key values (JSON or query param)
    (
        re.compile(
            r'("?api[_\-]?key"?\s*[=:]\s*["\']?)([A-Za-z0-9\-_\.]{8,})(["\']?)',
            re.IGNORECASE,
        ),
        r"\1***MASKED***\3",
    ),
    # Authorization header value
    (
        re.compile(r"(authorization[:\s]+)[^\s,\}\"]+", re.IGNORECASE),
        r"\1***MASKED***",
    ),
    # Token patterns (token=...)
    (
        re.compile(r"(token\s*[=:]\s*)[A-Za-z0-9\-_\.]{8,}", re.IGNORECASE),
        r"\1***MASKED***",
    ),
]


def mask_sensitive(text: str) -> str:
    """Replace sensitive values in *text* with masked placeholders.

    Only applied when ``settings.log_sensitive_masking`` is True.
    """
    if not settings.log_sensitive_masking:
        return text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ──────────────────────────────────────────────────────────────
# Log Format
# ──────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────
# Logger Configuration
# ──────────────────────────────────────────────────────────────


def configure_logger() -> None:
    """Configure loguru console sink."""
    logger.remove()

    logger.add(
        sys.stderr,
        format=structured_format,
        level=settings.log_level,
        colorize=settings.log_format == "text",
        backtrace=settings.debug,
        diagnose=settings.debug,
    )


def configure_file_logger() -> None:
    """Add a timed-rotating file sink to loguru.

    - Rotation  : every day at midnight (``rotation="00:00"``)
    - Retention : ``settings.log_rotation_backup_count`` days
    - Format    : same structured format as console
    - Encoding  : UTF-8

    Call once during application startup (e.g. inside ``lifespan()``).
    """
    log_path = Path(settings.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format=structured_format,
        level=settings.log_level,
        rotation="00:00",                              # rotate at midnight
        retention=f"{settings.log_rotation_backup_count} days",
        encoding="utf-8",
        backtrace=settings.debug,
        diagnose=settings.debug,
        enqueue=True,       # thread-safe async write
        delay=False,
    )
    logger.info(
        f"File logger initialised: path={log_path}, "
        f"retention={settings.log_rotation_backup_count}d"
    )


# ──────────────────────────────────────────────────────────────
# Public Helpers
# ──────────────────────────────────────────────────────────────


def get_logger(name: str | None = None) -> "logger":
    """Return a loguru logger bound with an optional *name* context.

    Usage::

        from app.core.logger import get_logger
        log = get_logger(__name__)
        log.info("hello")
    """
    if name:
        return logger.bind(module=name)
    return logger


def get_agent_logger(agent_uuid: str, session_id: str | None = None) -> "logger":
    """Get a logger bound with agent context."""
    bound_logger = logger.bind(agent_uuid=agent_uuid)
    if session_id:
        bound_logger = bound_logger.bind(session_id=session_id)
    return bound_logger


# ──────────────────────────────────────────────────────────────
# Token Usage Tracker
# ──────────────────────────────────────────────────────────────


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

# Configure console logger on import
configure_logger()
