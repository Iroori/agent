"""Sub-agent tool for agent orchestration."""

from typing import Any, TYPE_CHECKING

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.base_agent import BaseAgent


class SubAgentInput(BaseModel):
    """Input schema for sub-agent invocation."""

    message: str = Field(description="The message/task to send to the sub-agent")


class SubAgentTool(BaseTool):
    """Tool that wraps a sub-agent for use by a main agent.

    Allows main agents to delegate tasks to specialized sub-agents.
    """

    name: str = Field(description="Tool name (sub-agent name)")
    description: str = Field(description="What this sub-agent does")
    args_schema: type[BaseModel] = SubAgentInput
    agent_uuid: str = Field(description="Sub-agent UUID")

    # Store agent reference (not serialized)
    _agent: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        agent: "BaseAgent",
        name: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize sub-agent tool.

        Args:
            agent: The sub-agent to wrap
            name: Tool name (defaults to agent name)
            description: Tool description (defaults to agent system prompt prefix)
            **kwargs: Additional arguments
        """
        tool_name = name or f"sub_agent_{agent.name.lower().replace(' ', '_')}"
        tool_description = description or self._generate_description(agent)

        super().__init__(
            name=tool_name,
            description=tool_description,
            agent_uuid=agent.uuid,
            **kwargs,
        )
        self._agent = agent

    @staticmethod
    def _generate_description(agent: "BaseAgent") -> str:
        """Generate tool description from agent info.

        Args:
            agent: The sub-agent

        Returns:
            Generated description
        """
        prompt = agent.info.system_prompt
        # Use first 200 chars of system prompt as description
        if len(prompt) > 200:
            description = prompt[:197] + "..."
        else:
            description = prompt

        return f"Delegate task to {agent.name} agent: {description}"

    def _run(self, message: str) -> str:
        """Synchronous execution (raises error - use async).

        Args:
            message: Message to send

        Returns:
            Error message
        """
        raise NotImplementedError(
            "SubAgentTool requires async execution. Use _arun instead."
        )

    async def _arun(self, message: str) -> str:
        """Invoke the sub-agent asynchronously.

        Args:
            message: Message/task to send to the sub-agent

        Returns:
            Sub-agent's response
        """
        if self._agent is None:
            return f"Error: Sub-agent {self.agent_uuid} not available"

        try:
            logger.debug(f"Invoking sub-agent {self.agent_uuid}: {message[:50]}...")

            # Use a dedicated session for sub-agent calls
            session_id = f"sub_agent_call_{self.agent_uuid}"
            response = await self._agent.invoke(message, session_id)

            logger.debug(f"Sub-agent {self.agent_uuid} completed")
            return response.content

        except Exception as e:
            logger.error(f"Sub-agent {self.agent_uuid} failed: {e}")
            return f"Error from sub-agent: {e}"


def create_sub_agent_tool(
    agent: "BaseAgent",
    name: str | None = None,
    description: str | None = None,
) -> SubAgentTool:
    """Create a sub-agent tool from an agent.

    Args:
        agent: The sub-agent to wrap
        name: Optional custom tool name
        description: Optional custom description

    Returns:
        SubAgentTool instance
    """
    return SubAgentTool(agent=agent, name=name, description=description)
