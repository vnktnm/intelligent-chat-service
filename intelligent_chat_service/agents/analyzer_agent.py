from agents.agent import Agent
from typing import Optional
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
        prompt_data = get_prompt(
            config.PROMPT_PATH, config.PROMPT_AGENT_TYPE, config.PROMPT_ANALYZER_AGENT
        )

        if not prompt_data:
            logger.error(
                f"Failed to load prompt for {name} agent. Using fallback prompt."
            )
            system_prompt = "You are an expert query analyzer. Analyze the user query and provide insights."
        else:
            system_prompt = prompt_data["prompt"]

        # Enhance the system prompt to encourage human help requests for complex ethical questions
        if human_in_the_loop:
            system_prompt += "\n\nIMPORTANT INSTRUCTION: For complex ethical questions, policy issues, or questions requiring human judgment, you MUST request human input using the format 'HUMAN_HELP_NEEDED: [your specific question]'. This is especially important for questions about balancing competing priorities, ethics, or value judgments where human perspective is valuable."

        super().__init__(
            name=name,
            description="Agent to analyze the incoming request.",
            role="analyzer",
            system_prompt=system_prompt,
            model=config.OPENAI_DEFAULT_MODEL,
            temperature=temperature,
            require_thought=require_thought,
            max_tokens=max_tokens,
            human_in_the_loop=human_in_the_loop,
        )
