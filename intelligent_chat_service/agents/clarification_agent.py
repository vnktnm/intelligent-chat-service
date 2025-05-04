from .agent import Agent
from typing import Optional, Literal
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from schema import ChatRequest
from utils import logger
from pydantic import BaseModel, Field


class ClarificationAgent(Agent):
    """Agent that helps clarify the requests"""

    def __init__(
        self,
        name,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 200,
        require_thought: Optional[bool] = False,
        human_in_the_loop: bool = True,
        request: ChatRequest = None,
    ):
        """Agent to clarify if the user query is ambiguous"""
        prompt = get_prompt(
            config.PROMPT_PATH,
            config.PROMPT_AGENT_TYPE,
            config.PROMPT_CLARIFICATION_AGENT,
        )

        formatted_prompt = get_formatted_prompt(
            prompt=prompt["prompt"],
            variables={"data_sources": ", ".join(request.selected_sources)},
        )

        class ClarificationResponse(BaseModel):
            explanation: str
            type: Literal["clarification", "suggestion"]
            question: str = Field(
                description="Question to ask user if the user query is ambiguous based on the context"
            )
            valid: bool = Field(
                description="Identifies if the user query is valid. By default it is False"
            )

        super().__init__(
            name=name,
            description="Agent to clarify if the user query is ambiguous",
            role="clarification",
            system_prompt=formatted_prompt,
            model=model,
            temperature=temperature,
            response_format=ClarificationResponse,
            require_thought=require_thought,
            max_tokens=max_tokens,
            human_in_the_loop=human_in_the_loop,
        )
