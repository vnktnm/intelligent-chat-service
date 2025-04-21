from agents.agent import Agent
from typing import Optional, Literal
from pydantic import BaseModel
import config
from utils.prompt_utils import get_prompt
from utils import logger


class AnalyzerAgent(Agent):
    """An agent that analyzes the input and provides a response."""

    def __init__(
        self,
        name: str,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 200,
        require_thought: Optional[bool] = False,
        human_in_the_loop: bool = config.HUMAN_IN_THE_LOOP_ENABLED,
    ):
        """An agent to analyze the incoming request"""
        prompt = get_prompt(
            config.PROMPT_PATH, config.PROMPT_AGENT_TYPE, config.PROMPT_ANALYZER_AGENT
        )

        class AnalyzerResponse(BaseModel):
            explanation: str
            analysis: Literal["simple", "complex", "ambiguous"]

        super().__init__(
            name=name,
            description="Agent to analyze the incoming request.",
            role="analyzer",
            system_prompt=prompt,
            model=config.OPENAI_DEFAULT_MODEL,
            temperature=temperature,
            response_format=AnalyzerResponse,
            require_thought=require_thought,
            max_tokens=max_tokens,
            human_in_the_loop=human_in_the_loop,
        )
