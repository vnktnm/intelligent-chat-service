from agents.agent import Agent
from typing import Optional, Dict, Any
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from utils import logger


class AnalyzerAgent(Agent):
    """An agent that analyzes user input."""

    def __init__(
        self,
        name: str,
        model: str = config.OPENAI_DEFAULT_MODEL,
        require_thought: bool = False,
        human_in_the_loop: bool = False,
    ):
        # Fix the constructor by providing all required parameters
        super().__init__(
            name=name,
            model=model,
            require_thought=require_thought,
            description="Analyzer agent that evaluates user inputs and context",
            role="analyzer",
            system_prompt="You are an analyzer agent that evaluates inputs and provides insights.",
        )
        self.human_in_the_loop = human_in_the_loop
        logger.info(
            f"Analyzer agent initialized with human_in_the_loop={human_in_the_loop}"
        )
