from agents.agent import Agent
from typing import Optional
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from schema import ChatRequest
from utils import logger


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

        super().__init__(
            name=name,
            description="Agent to plan the request.",
            role="planner",
            system_prompt=prompt,
            model=model,
            temperature=temperature,
            response_format=PlannerResponse,
            require_thought=require_thought,
            tool_calls=[config.TOOL_QDRANT] if config.TOOL_QDRANT else [],
            max_tokens=max_tokens,
        )
