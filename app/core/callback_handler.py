"""Universal Tool Callback Handler for observability.

Tracks every tool invocation through the LangChain callback system and:
  - Logs start / end / error events to the rotating file via loguru
  - Emits WebSocket stream events so the frontend receives real-time status

Usage
-----
Instantiate once per agent session and pass it via the LangGraph/LangChain
``config`` dict::

    handler = UniversalToolCallbackHandler(
        agent_uuid="...", session_id="...", ws_manager=manager
    )
    await graph.ainvoke(input, config={"callbacks": [handler]})
"""

from __future__ import annotations

import time
import traceback
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from loguru import logger as _root_logger

from app.core.logger import get_logger, mask_sensitive
from app.core.settings import settings

_log = get_logger(__name__)

# WebSocket event type constants
_WS_TOOL_START = "TOOL_CALL_START_STREAM"
_WS_TOOL_END = "TOOL_CALL_END_STREAM"
_WS_TOOL_ERROR = "TOOL_CALL_ERROR_STREAM"

# Maximum characters logged for tool output
_MAX_OUTPUT_LOG_CHARS = 1000


class UniversalToolCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that traces every tool call.

    Parameters
    ----------
    agent_uuid:
        UUID of the owning agent — embedded in every log record.
    session_id:
        Current session ID — embedded in every log record.
    ws_manager:
        ``ConnectionManager`` instance from ``app.api.websocket``.
        Pass ``None`` to disable WebSocket notifications (useful in tests).
    """

    def __init__(
        self,
        agent_uuid: str,
        session_id: str,
        ws_manager: Any | None = None,
    ) -> None:
        super().__init__()
        self._agent_uuid = agent_uuid
        self._session_id = session_id
        self._ws_manager = ws_manager
        # Maps run_id → (tool_name, start_time)
        self._pending: dict[str, tuple[str, float]] = {}
        # Bind context to loguru for every record emitted by this handler
        self._log = _root_logger.bind(
            agent_uuid=agent_uuid,
            session_id=session_id,
        )

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    async def _send_ws(self, event_type: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget WebSocket message; silently skips if no manager."""
        if self._ws_manager is None:
            return
        try:
            await self._ws_manager.send_json(
                self._agent_uuid,
                self._session_id,
                {"type": event_type, **payload},
            )
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f"WebSocket send failed [{event_type}]: {exc}")

    @staticmethod
    def _run_id_str(run_id: UUID | str | None) -> str:
        return str(run_id) if run_id else "unknown"

    # ──────────────────────────────────────────────────────────────
    # Callback implementations
    # ──────────────────────────────────────────────────────────────

    async def on_tool_start(  # type: ignore[override]
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool begins execution.

        Logs the tool name + masked input and sends a WS start event.
        """
        tool_name: str = serialized.get("name", "unknown_tool")
        run_id_str = self._run_id_str(run_id)
        masked_input = mask_sensitive(str(input_str))

        # Record start time for latency calculation
        self._pending[run_id_str] = (tool_name, time.monotonic())

        self._log.info(
            f"[TOOL_START] tool={tool_name} run_id={run_id_str} "
            f"input={masked_input!r}"
        )

        await self._send_ws(
            _WS_TOOL_START,
            {
                "tool": tool_name,
                "run_id": run_id_str,
                "input": masked_input,
            },
        )

    async def on_tool_end(  # type: ignore[override]
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes successfully.

        Logs output (capped at 1 000 chars) + latency.
        Emits a WARNING when latency exceeds the configured threshold.
        Sends a WS end event.
        """
        run_id_str = self._run_id_str(run_id)
        tool_name, start_time = self._pending.pop(run_id_str, ("unknown_tool", time.monotonic()))
        latency_secs = time.monotonic() - start_time
        latency_ms = round(latency_secs * 1000, 2)

        output_str = str(output)
        output_preview = mask_sensitive(
            output_str[:_MAX_OUTPUT_LOG_CHARS]
            + ("..." if len(output_str) > _MAX_OUTPUT_LOG_CHARS else "")
        )

        self._log.info(
            f"[TOOL_END] tool={tool_name} run_id={run_id_str} "
            f"latency_ms={latency_ms} output_preview={output_preview!r}"
        )

        if latency_secs >= settings.tool_latency_warning_threshold_secs:
            self._log.warning(
                f"[SLOW_TOOL] tool={tool_name} run_id={run_id_str} "
                f"latency_ms={latency_ms} "
                f"threshold_secs={settings.tool_latency_warning_threshold_secs}"
            )

        await self._send_ws(
            _WS_TOOL_END,
            {
                "tool": tool_name,
                "run_id": run_id_str,
                "output_preview": output_preview,
                "latency_ms": latency_ms,
            },
        )

    async def on_tool_error(  # type: ignore[override]
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool raises an exception.

        Logs the full traceback at ERROR level and sends a WS error event.
        """
        run_id_str = self._run_id_str(run_id)
        tool_name, _ = self._pending.pop(run_id_str, ("unknown_tool", time.monotonic()))

        tb = traceback.format_exc()
        self._log.error(
            f"[TOOL_ERROR] tool={tool_name} run_id={run_id_str} "
            f"error={error!r}\n{tb}"
        )

        await self._send_ws(
            _WS_TOOL_ERROR,
            {
                "tool": tool_name,
                "run_id": run_id_str,
                "error_message": str(error),
            },
        )
