"""FastAPI REST endpoints for agent interaction."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from app.agents.pool import get_agent_pool
from app.core.logger import token_tracker
from app.loaders.base import AgentInfo
from app.memory.base import get_memory_storage

router = APIRouter(tags=["agents"])


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Session ID for continuity")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    content: str
    session_id: str
    token_usage: dict[str, int]
    agent_uuid: str


class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""

    uuid: str = Field(description="Unique agent identifier")
    name: str = Field(description="Agent name")
    model: str = Field(default="gpt-4o", description="LLM model")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt",
    )
    tools: list[str] = Field(default_factory=list, description="Tool names to bind")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


class AgentStatusResponse(BaseModel):
    """Agent status response."""

    uuid: str
    name: str
    status: str
    created_at: str | None
    last_used_at: str | None
    invocation_count: int
    active_sessions: int
    total_tokens: dict[str, int]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    pool_size: int
    agents: list[str]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    pool = get_agent_pool()
    return HealthResponse(
        status="healthy",
        pool_size=pool.size,
        agents=pool.list_agents(),
    )


@router.get("/agents", response_model=list[str])
async def list_agents() -> list[str]:
    """List all agents in the pool."""
    pool = get_agent_pool()
    return pool.list_agents()


@router.post("/agents", response_model=AgentStatusResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(request: AgentCreateRequest) -> AgentStatusResponse:
    """Create a new agent from configuration."""
    pool = get_agent_pool()

    # Check if agent already exists
    existing = pool.get(request.uuid)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent already exists: {request.uuid}",
        )

    try:
        agent_info = AgentInfo(
            uuid=request.uuid,
            name=request.name,
            model=request.model,
            system_prompt=request.system_prompt,
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        agent = await pool.create_from_info(agent_info)
        stats = agent.stats
        return AgentStatusResponse(
            uuid=stats["uuid"],
            name=stats["name"],
            status=stats["status"],
            created_at=stats["created_at"],
            last_used_at=stats["last_used_at"],
            invocation_count=stats["invocation_count"],
            active_sessions=stats["active_sessions"],
            total_tokens=stats["total_tokens"],
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {e}",
        )


@router.get("/agents/{uuid}", response_model=AgentStatusResponse)
async def get_agent_status(uuid: str) -> AgentStatusResponse:
    """Get agent status by UUID."""
    pool = get_agent_pool()

    try:
        agent = await pool.get_or_create(uuid)
        stats = agent.stats
        return AgentStatusResponse(
            uuid=stats["uuid"],
            name=stats["name"],
            status=stats["status"],
            created_at=stats["created_at"],
            last_used_at=stats["last_used_at"],
            invocation_count=stats["invocation_count"],
            active_sessions=stats["active_sessions"],
            total_tokens=stats["total_tokens"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.delete("/agents/{uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(uuid: str) -> None:
    """Remove an agent from the pool."""
    pool = get_agent_pool()
    removed = await pool.remove(uuid)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {uuid}",
        )


@router.post("/agents/{uuid}/chat", response_model=ChatResponse)
async def chat_with_agent(uuid: str, request: ChatRequest) -> ChatResponse:
    """Send a message to an agent and get a response."""
    pool = get_agent_pool()

    try:
        agent = await pool.get_or_create(uuid)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    try:
        response = await agent.invoke(request.message, request.session_id)
        return ChatResponse(
            content=response.content,
            session_id=response.session_id,
            token_usage=response.token_usage.model_dump(),
            agent_uuid=uuid,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {e}",
        )


@router.post("/agents/{uuid}/reload", response_model=AgentStatusResponse)
async def reload_agent(uuid: str) -> AgentStatusResponse:
    """Reload an agent from its configuration."""
    pool = get_agent_pool()

    try:
        agent = await pool.reload(uuid)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {uuid}",
            )
        stats = agent.stats
        return AgentStatusResponse(
            uuid=stats["uuid"],
            name=stats["name"],
            status=stats["status"],
            created_at=stats["created_at"],
            last_used_at=stats["last_used_at"],
            invocation_count=stats["invocation_count"],
            active_sessions=stats["active_sessions"],
            total_tokens=stats["total_tokens"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/agents/{uuid}/sessions", response_model=list[str])
async def list_sessions(uuid: str) -> list[str]:
    """List all sessions for an agent."""
    memory = get_memory_storage()
    return await memory.list_sessions(uuid)


@router.delete("/agents/{uuid}/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(uuid: str, session_id: str) -> None:
    """Clear a session's conversation history."""
    memory = get_memory_storage()
    cleared = await memory.clear(uuid, session_id)
    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )


@router.get("/stats/pool")
async def get_pool_stats() -> dict[str, Any]:
    """Get agent pool statistics."""
    pool = get_agent_pool()
    return {
        "pool_size": pool.size,
        "max_size": pool._max_size,
        "is_full": pool.is_full,
        "agents": pool.get_all_stats(),
    }


@router.get("/stats/tokens")
async def get_token_stats() -> dict[str, dict[str, int]]:
    """Get token usage statistics."""
    return token_tracker.get_all_usage()
