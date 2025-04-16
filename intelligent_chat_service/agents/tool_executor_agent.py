from agents.agent import Agent
from typing import Dict, Any, Optional
from core import OpenAIService
import config
from utils import logger
import json


class ToolExecutorAgent(Agent):
    """An agent that executes a specific tool with given parameters."""

    def __init__(
        self,
        name: str,
        tool_name: str,
        parameters: Dict[str, Any] = None,
        model: str = config.OPENAI_DEFAULT_MODEL,
    ):
        super().__init__(
            name=name,
            model=model,
            require_thought=False,
            description=f"Tool executor for {tool_name}",
            role="tool_executor",
            system_prompt=f"You are a tool executor agent that runs the {tool_name} tool with specific parameters.",
        )
        self.tool_name = tool_name
        self.parameters = parameters or {}

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Execute the tool with the provided parameters."""
        try:
            if callback:
                await callback(
                    "tool_execution_start",
                    {
                        "tool_name": self.tool_name,
                        "parameters": self.parameters,
                        "agent_name": self.name,
                    },
                )

            # This is where you would implement the actual tool execution
            # For now, we'll just log the attempt and return a placeholder result
            logger.info(
                f"Executing tool {self.tool_name} with parameters: {json.dumps(self.parameters)}"
            )

            # In a real implementation, you would:
            # 1. Look up the tool in a tool registry
            # 2. Call the tool's execute method with the parameters
            # 3. Process and return the result

            # Placeholder implementation
            result = {
                "status": "success",
                "message": f"Tool {self.tool_name} execution simulated",
                "data": {"tool_name": self.tool_name, "parameters": self.parameters},
            }

            # Store result in context
            context[f"{self.name}_result"] = result

            if callback:
                await callback(
                    "tool_execution_complete",
                    {
                        "tool_name": self.tool_name,
                        "parameters": self.parameters,
                        "result": result,
                        "agent_name": self.name,
                    },
                )

            return context

        except Exception as e:
            error_message = f"Error executing tool {self.tool_name}: {str(e)}"
            logger.error(error_message)

            # Store error in context
            context[f"{self.name}_error"] = error_message

            if callback:
                await callback(
                    "tool_execution_error",
                    {
                        "tool_name": self.tool_name,
                        "parameters": self.parameters,
                        "error": str(e),
                        "agent_name": self.name,
                    },
                )

            # Re-raise to let the orchestrator handle it
            raise
