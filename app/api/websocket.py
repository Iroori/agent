"""WebSocket endpoint for streaming agent responses."""

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.agents.pool import get_agent_pool
from app.core.logger import get_agent_logger

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, dict[str, WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self, websocket: WebSocket, agent_uuid: str, session_id: str
    ) -> None:
        """Accept and track a WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if agent_uuid not in self._connections:
                self._connections[agent_uuid] = {}
            self._connections[agent_uuid][session_id] = websocket
        logger.info(f"WebSocket connected: agent={agent_uuid}, session={session_id}")

    async def disconnect(self, agent_uuid: str, session_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if agent_uuid in self._connections:
                self._connections[agent_uuid].pop(session_id, None)
                if not self._connections[agent_uuid]:
                    del self._connections[agent_uuid]
        logger.info(f"WebSocket disconnected: agent={agent_uuid}, session={session_id}")

    async def send_json(
        self, agent_uuid: str, session_id: str, data: dict[str, Any]
    ) -> bool:
        """Send JSON data to a specific connection."""
        async with self._lock:
            ws = self._connections.get(agent_uuid, {}).get(session_id)
        if ws:
            try:
                await ws.send_json(data)
                return True
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
        return False

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        count = 0
        for sessions in self._connections.values():
            count += len(sessions)
        return count


# Global connection manager
manager = ConnectionManager()


@router.websocket("/agents/{agent_uuid}/stream")
async def stream_chat(websocket: WebSocket, agent_uuid: str) -> None:
    """WebSocket endpoint for streaming chat with an agent.

    Protocol:
    - Client sends: {"message": "user input", "session_id": "optional"}
    - Server sends: {"type": "chunk", "content": "..."} for each chunk
    - Server sends: {"type": "done", "session_id": "..."} when complete
    - Server sends: {"type": "error", "message": "..."} on error
    """
    session_id = str(uuid.uuid4())
    agent_logger = get_agent_logger(agent_uuid, session_id)

    pool = get_agent_pool()

    # Try to get or create agent
    try:
        agent = await pool.get_or_create(agent_uuid)
    except ValueError as e:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return
    except RuntimeError as e:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    await manager.connect(websocket, agent_uuid, session_id)

    # Send session info
    await websocket.send_json({
        "type": "connected",
        "agent_uuid": agent_uuid,
        "session_id": session_id,
    })

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                request = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            message = request.get("message")
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "message": "Missing 'message' field",
                })
                continue

            # Use provided session_id or default
            req_session_id = request.get("session_id", session_id)

            agent_logger.info(f"Streaming message: {message[:100]}...")

            try:
                # Stream response
                async for chunk in agent.stream(message, req_session_id):
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk,
                    })

                # Send completion
                await websocket.send_json({
                    "type": "done",
                    "session_id": req_session_id,
                })

            except Exception as e:
                agent_logger.error(f"Streaming error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        agent_logger.info("Client disconnected")
    except Exception as e:
        agent_logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(agent_uuid, session_id)


@router.get("/ws/stats")
async def websocket_stats() -> dict[str, Any]:
    """Get WebSocket connection statistics."""
    return {
        "active_connections": manager.get_connection_count(),
    }
