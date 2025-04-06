from agents.agent import Agent
from typing import Optional, Dict, Any
import config
from schema import ChatRequest
from utils import logger


class PlannerAgent(Agent):
    """An agent that plans and generates a response."""

    def __init__(
        self,
        name: str,
        model: str = config.OPENAI_DEFAULT_MODEL,
        require_thought: bool = False,
        request: Optional[ChatRequest] = None,
    ):
        # Fix the constructor by providing all required parameters
        super().__init__(
            name=name,
            model=model,
            require_thought=require_thought,
            description="Planner agent that generates the final response",
            role="planner",
            system_prompt="You are a planning agent that designs and generates responses based on analysis.",
        )
        # Store the original request safely
        self.request = request
        # Default values to use if request is None
        self.thread_id = None
        self.session_id = None
        self.user_id = None

        # Extract and store necessary values from request if available
        if request and request.config:
            self.thread_id = request.config.thread_id
            self.session_id = (
                request.config.session_id
                if hasattr(request.config, "session_id")
                else None
            )
            self.user_id = (
                request.config.user_id if hasattr(request.config, "user_id") else None
            )

        logger.info(f"Planner agent initialized with thread_id: {self.thread_id}")
