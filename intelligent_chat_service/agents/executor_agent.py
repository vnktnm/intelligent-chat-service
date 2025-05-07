from .agent import Agent
from typing import Optional, Any, Callable, Dict, List
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from utils import logger
from schema.idiscovery_orchestrator import ExecutionResponse
from schema.graph_orchestrator import GraphNodeDefinition
from core.llm.openai import OpenAIService
import json
import uuid
from orchestrator.graph_orchestrator import GraphOrchestrator
from datetime import datetime


class TaskExecutor:
    """A class for executing a specific task."""

    def __init__(
        self,
        task_id: str,
        task_type: str,
        tool: str,
        args: List[Any],
        executor: "ExecutorAgent",
    ):
        self.name = task_id  # Name attribute for compatibility
        self.description = f"Task: {task_id}"
        self.task_id = task_id
        self.task_type = task_type
        self.tool = tool
        self.args = args
        self.executor = executor
        self.status = "pending"
        self.result = None
        self.start_time = None
        self.end_time = None

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the task using the specified tool"""
        self.start_time = datetime.now().isoformat()
        self.status = "running"

        logger.info(f"executor_agent args: {self.args}")

        tool_call = {
            "id": f"task_{uuid.uuid4().hex[:8]}",
            "function": {"name": self.tool, "arguments": json.dumps(self.args)},
        }

        if callback:
            await callback(
                "task_start",
                {
                    "task_id": self.task_id,
                    "task_type": self.task_type,
                    "tool": self.tool,
                    "args": self.args,
                    "status": self.status,
                },
            )

        try:
            result = await self.executor.execute_tool(tool_call, context)

            context[f"task_result_{self.task_id}"] = result
            self.executor.task_results[self.task_id] = result
            self.result = result
            self.status = "completed"
            self.end_time = datetime.now().isoformat()

            if callback:
                await callback(
                    "task_complete",
                    {
                        "task_id": self.task_id,
                        "tool": self.tool,
                        "result": result,
                        "status": self.status,
                        "execution_time": self._get_execution_time(),
                    },
                )
        except Exception as e:
            error_msg = f"Error executing task {self.task_id}: {str(e)}"
            logger.error(error_msg)
            self.status = "failed"
            self.result = error_msg
            self.end_time = datetime.now().isoformat()

            if callback:
                await callback(
                    "task_error",
                    {
                        "task_id": self.task_id,
                        "tool": self.tool,
                        "error": error_msg,
                        "status": self.status,
                        "execution_time": self._get_execution_time(),
                    },
                )

            context[f"task_result_{self.task_id}"] = error_msg
            self.executor.task_results[self.task_id] = {
                "error": error_msg,
                "status": "failed",
            }
        return context

    def _get_execution_time(self) -> float:
        if not self.start_time or not self.end_time:
            return 0

        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)

        return (end - start).total_seconds()


class ExecutorAgent(Agent):
    """An agent that analyzes and executes based on the generated plan."""

    def __init__(
        self,
        name,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 200,
        require_thought: Optional[bool] = False,
        human_in_the_loop: bool = False,
    ):
        """Agent to execute the plan."""

        prompt = get_prompt(
            config.PROMPT_PATH,
            config.PROMPT_AGENT_TYPE,
            config.PROMPT_EXECUTOR_AGENT,
        )

        formatted_prompt = get_formatted_prompt(
            prompt=prompt.get(
                "prompt",
                "You are an executor agent responsible for executing planned tasks efficiently",
            ),
            variables={},
        )

        super().__init__(
            name=name,
            description="Agent to execute the plan.",
            role="executor",
            system_prompt=formatted_prompt,
            model=model,
            temperature=temperature,
            response_format=ExecutionResponse,
            require_thought=require_thought,
            max_tokens=max_tokens,
            human_in_the_loop=human_in_the_loop,
            tool_calls=(
                config.AVAILABLE_TOOLS if hasattr(config, "AVAILABLE_TOOLS") else []
            ),
        )

        self.current_subgraph = None
        self.task_results = {}

    def create_task_executor(self, task: Dict[str, Any]) -> TaskExecutor:
        """Factory method to create a TaskExecutor from a task definition."""
        return TaskExecutor(
            task_id=task.get("id"),
            task_type=task.get("type", "tool"),
            tool=task.get("tool", ""),
            args=task.get("args", []),
            executor=self,
        )

    async def execute_tasks(
        self,
        tasks: List[Dict[str, Any]],
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a list of tasks using a subgraph"""
        logger.info(f"Building execution graph for {len(tasks)} tasks")

        self.task_results = {}

        # Get the current graph from context or create a new one
        parent_graph = context.get(
            "current_graph",
            GraphOrchestrator(name="default", description="Default graph"),
        )

        # Use the enhanced build_execution_subgraph method
        execution_graph = parent_graph.build_execution_subgraph(
            tasks=tasks, task_executor_factory=self.create_task_executor
        )

        self.current_subgraph = execution_graph

        if callback:
            await callback(
                "execution_start",
                {
                    "agent": self.name,
                    "task_count": len(tasks),
                    "execution_id": str(uuid.uuid4()),
                },
            )

        try:
            # Set current graph in context for proper nesting
            previous_graph = context.get("current_graph", None)
            context["current_graph"] = execution_graph

            # Execute the subgraph
            context = await execution_graph.execute(context, openai_service, callback)

            # Restore previous graph
            if previous_graph:
                context["current_graph"] = previous_graph
            else:
                context.pop("current_graph", None)

            # Collect results
            results_summary = {
                "total_tasks": len(tasks),
                "completed_tasks": len(self.task_results),
                "results": self.task_results,
            }

            context["execution_results"] = results_summary

            if callback:
                await callback(
                    "execution_complete",
                    {
                        "agent": self.name,
                        "task_count": len(tasks),
                        "completed_count": len(self.task_results),
                        "summary": results_summary,
                    },
                )

            return context
        except Exception as e:
            logger.error(f"Error executing task graph: {str(e)}")
            if callback:
                await callback("execution_error", {"agent": self.name, "error": str(e)})

            raise e

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Executes the executor agent - processes planner output and executes tasks"""

        await self.load_tools()

        planner_output = context.get("planner_agent_output")
        if not planner_output:
            logger.warning("No planner output found in context")
            if callback:
                await callback(
                    "node_skipped", {"node_id": self.name, "step": self.name}
                )
            return context

        try:
            # Parse planner output
            planner_data = self._parse_planner_output(planner_output)

            # Extract tasks from planner data
            tasks = planner_data.get("plan", [])
            logger.info(f"Tasks: {tasks}")

            if not tasks:
                logger.info(f"No tasks found in planner output")
                return context

            # Execute the tasks
            logger.info(f"Executing {len(tasks)} tasks from planner")
            context = await self.execute_tasks(tasks, context, openai_service, callback)

            # Prepare response
            response_data = self._create_response_data(tasks)

            # Update context with results
            context[self.name] = json.dumps(response_data)
            context[f"{self.name}_output"] = json.dumps(response_data)
            self.result = json.dumps(response_data)

            return context
        except Exception as e:
            logger.error(f"Error in executor agent: {str(e)}")
            if callback:
                await callback("step_error", {"step": self.name, "error": str(e)})
            return context

    def _parse_planner_output(self, planner_output: Any) -> Dict[str, Any]:
        """Parse the planner output to extract task data."""
        if isinstance(planner_output, str):
            try:
                return json.loads(planner_output)
            except json.JSONDecodeError:
                logger.error("Failed to parse planner output as JSON")
                return {"plan": []}
        else:
            return planner_output

    def _create_response_data(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create the response data structure based on task execution results."""
        return {
            "summary": f"Executed {len(self.task_results)}/{len(tasks)} tasks",
            "execution_status": (
                "success" if len(self.task_results) == len(tasks) else "partial_success"
            ),
            "results": self.task_results,
        }
