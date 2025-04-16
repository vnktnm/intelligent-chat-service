from agents.agent import Agent
from typing import Optional, Dict, Any, List
import config
from schema import ChatRequest
from utils import logger
import uuid


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
            system_prompt="""You are a planning agent that designs and generates responses based on analysis.
            
When you create a plan, organize it as a structured list of tasks. Each task should include:
1. A unique task_id
2. The specific tool required for execution
3. Parameters needed for the tool
4. A clear description of what the task accomplishes
5. Any dependencies on other tasks

Your output must be properly formatted for automated processing.""",
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

    def _format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format the response to ensure it has the required structure."""
        formatted = response.copy()

        # Ensure plan_id exists
        if "plan_id" not in formatted:
            formatted["plan_id"] = f"plan_{uuid.uuid4().hex[:8]}"

        # Ensure goal exists
        if "goal" not in formatted:
            formatted["goal"] = "Complete the user request"

        # Ensure tasks exist and are properly formatted
        if "tasks" not in formatted or not isinstance(formatted["tasks"], list):
            formatted["tasks"] = []

        # Format each task
        for i, task in enumerate(formatted["tasks"]):
            # Ensure task_id exists
            if "task_id" not in task:
                task["task_id"] = f"task_{i+1}_{uuid.uuid4().hex[:6]}"

            # Ensure tool exists
            if "tool" not in task:
                task["tool"] = "unknown_tool"

            # Ensure parameters exist
            if "parameters" not in task or not isinstance(task["parameters"], dict):
                task["parameters"] = {}

            # Ensure description exists
            if "description" not in task:
                task["description"] = f"Task {i+1}"

            # Ensure dependencies exist
            if "dependencies" not in task or not isinstance(task["dependencies"], list):
                task["dependencies"] = []

        return formatted

    async def process_response(
        self, context: Dict[str, Any], response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the planner agent response before returning it."""
        # Format the response to ensure it has the required structure
        formatted_response = self._format_response(response)

        # Store the formatted response in the context
        context[f"{self.name}_result"] = formatted_response

        logger.info(
            f"Planner {self.name} generated {len(formatted_response.get('tasks', []))} tasks"
        )
        return context
