from typing import List, Dict, Any, Optional, Callable, Set
from core import OpenAIService
from utils import logger
from schema import Step
from agents import Agent
import networkx as nx
import json


class Orchestrator:
    """Base class for orchestrating steps or workflows."""

    def __init__(self, name: str, description: str, steps: List[Step] = None):
        self.name = name
        self.description = description
        self.steps = steps if steps else []
        self.tool_manager = None
        self.execution_history = []
        self.execution_id = None

    def add_step(self, step: Step) -> None:
        """Add a step to this workflow."""
        self.steps.append(step)

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute all steps in the orchestrator."""
        context = dict(context)

        if callback:
            await callback(
                "orchestrator_start",
                {
                    "name": self.name,
                    "description": self.description,
                    "total_steps": len(self.steps),
                },
            )

        logger.info(f"Starting orchestration {self.name} with {len(self.steps)} steps.")

        for i, step in enumerate(self.steps):
            logger.info(f"Starting step {i+1}/{len(self.steps)}: {step.name}")

            if callback:
                await callback(
                    "orchestrate_progress",
                    {
                        "current_step": i + 1,
                        "total_steps": len(self.steps),
                        "step_name": step.name,
                    },
                )

            try:
                context = await step.execute(context, openai_service, callback)
            except Exception as e:
                logger.error(f"Error in workflow step {step.name}: {str(e)}")
                if callback:
                    await callback(
                        "step_error", {"step_name": step.name, "error": str(e)}
                    )
                raise e

        if callback:
            await callback(
                "orchestrator_complete",
                {"name": self.name, "description": self.description},
            )

        logger.info(f"Completed Orchestrator: {self.name}")
        return context

    def cleanup(self):
        if self.tool_manager:
            self.tool_manager.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert orchestrator to dictionary representation for API endpoints."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [step.name for step in self.steps],
        }
