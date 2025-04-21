from agents.agent import Agent
from typing import Optional, Literal, Any
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from schema import ChatRequest
from utils import logger
from pydantic import BaseModel, Field


class PlannerAgent(Agent):
    """An agent that analyzes the input and provides a response."""

    def __init__(
        self,
        name: str,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 200,
        require_thought: Optional[bool] = False,
        request: ChatRequest = None,
    ):
        """An agent to analyze the incoming request"""
        prompt = get_prompt(
            config.PROMPT_PATH, config.PROMPT_AGENT_TYPE, config.PROMPT_PLANNER_AGENT
        )

        formatted_prompt = get_formatted_prompt(
            prompt=prompt["prompt"],
            variables={"data_source": ", ".join(request.selected_sources)},
        )

        class PlannerTask(BaseModel):
            id: str
            description: str
            type: str = Literal["tool", "direct_response"]
            execution_type: str = Literal["sequential", "parallel"]
            tool: str
            args: list[Any] = Field(
                description="List of all arguments to pass to the tool"
            )
            capability: list[str]
            dependencies: list[str]

        class PlannerResponse(BaseModel):
            explanation: str = Field(
                description="A brief explanation about the plan and choice of resulting plan."
            )
            plan: list[PlannerTask] = Field(
                description="Plan with a list of tool exections"
            )

        super().__init__(
            name=name,
            description="Agent to plan the request.",
            role="planner",
            system_prompt=formatted_prompt,
            model=model,
            temperature=temperature,
            response_format=PlannerResponse,
            require_thought=require_thought,
            tool_calls=[config.TOOL_QDRANT] if config.TOOL_QDRANT else [],
            max_tokens=max_tokens,
        )
