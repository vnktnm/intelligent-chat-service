from typing import List, Dict, Any, Optional, Callable
from core import OpenAIService
from utils import logger
from schema import Step
from agents import Agent


class Orchestrator:
    """Orchestrates a sequence of steps using agents."""

    def __init__(self, name: str, description: str, steps: List[Step] = None):
        self.name = name
        self.description = description
        self.steps = steps if steps else []
        self.tool_manager = None

    def add_step(self, step: Step) -> None:
        """Add a step to this workflow."""
        self.steps.append(step)

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute all steps in the orchestrator.

        Args:
            context (Dict[str, Any]): _description_
            openai_service (OpenAIService): _description_
            callback (Optional[Callable[[str, Dict[str, Any]], None]], optional): _description_. Defaults to None.

        Returns:
            Dict[str, Any]: _description_
        """
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
