from .agent import Agent
from typing import Optional
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from schema import ChatRequest
from schema.idiscovery_orchestrator import PlannerResponse


class PlannerAgent(Agent):
    """An agent that generates a plan of tasks with available tools."""

    def __init__(
        self,
        name,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1000,
        require_thought: Optional[bool] = True,
        request: ChatRequest = None,
    ):
        prompt = get_prompt(
            config.PROMPT_PATH, config.PROMPT_AGENT_TYPE, config.PROMPT_PLANNER_AGENT
        )
        formatted_prompt = get_formatted_prompt(
            prompt=prompt["prompt"],
            variables={"data_source": ", ".join(request.selected_sources)},
        )

        super().__init__(
            name=name,
            description="Agent to plan the request based on the analysis",
            role="planner",
            system_prompt=formatted_prompt,
            model=model,
            temperature=temperature,
            response_format=PlannerResponse,
            require_thought=require_thought,
            max_tokens=max_tokens,
        )
