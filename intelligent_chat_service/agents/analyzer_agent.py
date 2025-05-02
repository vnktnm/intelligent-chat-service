from .agent import Agent
from typing import Optional, Literal
import config
from utils.prompt_utils import get_prompt
from utils import logger
from pydantic import BaseModel


class AnalyzerAgent(Agent):
    """Analyzer Agent that analyzes user queries"""

    def __init__(
        self,
        name,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 200,
        require_thought: Optional[bool] = False,
        human_in_the_loop: bool = False,
    ):
        """Agent to analyze the incoming request"""
        prompt = get_prompt(
            config.PROMPT_PATH, config.PROMPT_AGENTS_TYPE, config.PROMPT_ANALYZER_AGENT
        )

        class AnalyzerResponse(BaseModel):
            explanation: str
            analysis: Literal["simple", "complex", "ambiguous"]

        super().__init__(
            name=name,
            description="Agent to analyze the incoming request.",
            role="analyzer",
            system_prompt=prompt["prompt"],
            model=model,
            temperature=temperature,
            response_format=AnalyzerResponse,
            require_thought=require_thought,
            max_tokens=max_tokens,
            human_in_the_loop=human_in_the_loop,
        )
