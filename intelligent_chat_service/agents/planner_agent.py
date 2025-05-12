from .agent import Agent
from typing import Optional, Literal, Any
import config


class PlannerAgent(Agent):
    """Analyzer Agent that analyzes user queries"""

    def __init__(
        self,
        name: str = "planner",
        description: str = "An agent that generates the plan based on the user query.",
        role: str = "planner",
        system_prompt: str = None,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        tools: list[dict[str, Any]] = None,
        response_format: Optional[Any] = None,
        require_thought: Optional[bool] = False,
        human_in_the_loop: bool = False,
        stream: Optional[bool] = False,
    ):

        super().__init__(
            name=name,
            description=description,
            role=role,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
            require_thought=require_thought,
            human_in_the_loop=human_in_the_loop,
            stream=stream,
        )
