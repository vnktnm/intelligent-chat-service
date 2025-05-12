from .agent import Agent
from typing import Optional, Literal, Any
import config


class AnalyzerAgent(Agent):
    """Analyzer Agent that analyzes user queries"""

    def __init__(
        self,
        name: str = "analyzer",
        description: str = "An agent that analyzes the user query and classifies based on the complexity",
        role: str = "analyzer",
        system_prompt: str = None,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 200,
        tools: list[dict[str, Any]] = None,
        response_format: Optional[Any] = None,
        require_thought: Optional[bool] = False,
        human_in_the_loop: bool = False,
        stream: Optional[bool] = False,
    ):
        """Agent to analyze the incoming request"""

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
