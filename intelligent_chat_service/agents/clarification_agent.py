from .agent import Agent
from typing import Optional, Literal
from pydantic import Field, BaseModel
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from schema import ChatRequest
from utils import logger


class ClarificationAgent(Agent):
    def __init__(
        self,
        name,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 300,
        require_thought: Optional[bool] = False,
        human_in_the_loop: bool = True,
        request: ChatRequest = None,
    ):
        prompt = get_prompt(
            config.PROMPT_PATH,
            config.PROMPT_AGENT_TYPE,
            config.PROMPT_CLARIFICATION_AGENT,
        )

        formatted_prompt = get_formatted_prompt(
            prompt=prompt["prompt"],
            variables={"data_source": ", ".join(request.selected_sources)},
        )

        super().__init__(
            name=name,
            description="Agent to clarify if the user query is ambigous",
            role="clarification",
            system_prompt=formatted_prompt,
            model=model,
            temperature=temperature,
            response_format=ClarificationResponse,
            require_thought=require_thought,
            max_tokens=max_tokens,
            human_in_the_loop=human_in_the_loop,
        )
