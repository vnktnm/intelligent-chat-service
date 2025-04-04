from typing import Dict, Any, Optional, Callable
from core import OpenAIService
from dataclasses import dataclass


class Step:
    """Base class for orchestrator steps that interact with LLM"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.result = None

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """Execute this workflow step

        Args:
            context (Dict[str, Any]): _description_
            openai_service (OpenAIService): _description_
            callback (Optional[Callable[[str, Dict[str, Any]], None]], optional): _description_. Defaults to None.
        """
        raise NotImplementedError("Subclasses must implement this method")


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
