from agents.agent import Agent
from typing import Optional, Dict, Any, Callable, List
from pydantic import BaseModel, Field
import config
from utils.prompt_utils import get_prompt, get_formatted_prompt
from schema import ChatRequest
from utils import logger
import json
from orchestrator.graph_orchestrator import GraphOrchestrator
import networkx as nx
import uuid
from core import OpenAIService


class ExecutorSubGraph(GraphOrchestrator):
    """A sub-graph orchestrator specifically for task execution."""

    def __init__(self, name: str, parent_agent: "ExecutorAgent"):
        super().__init__(
            name=f"exec-subgraph-{name}", description=f"Execution sub-graph for {name}"
        )
        self.parent_agent = parent_agent
        self.task_results = {}

    def add_task_node(
        self,
        task_id: str,
        task_type: str,
        tool: str,
        args: List[Any],
        dependencies: List[str],
    ) -> str:
        """Add a task node to the execution graph."""
        # Create a lightweight Step object that wraps the task execution
        from schema import Step

        class TaskStep(Step):
            def __init__(
                self,
                task_id: str,
                task_type: str,
                tool: str,
                args: List[Any],
                executor: "ExecutorAgent",
            ):
                super().__init__(name=task_id, description=f"Task: {task_id}")
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
                logger.info(f"Executing task {self.task_id} with tool {self.tool}")
                self.start_time = datetime.now().isoformat()
                self.status = "running"

                # Create a tool call structure compatible with the executor's tool execution interface
                tool_call = {
                    "id": f"task_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": self.tool,
                        # Handle both dictionary and list args
                        "arguments": json.dumps(
                            self.args[0]
                            if isinstance(self.args, list) and self.args
                            else (self.args if isinstance(self.args, dict) else {})
                        ),
                    },
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
                    # Execute the tool using the executor agent's tool execution capability
                    result = await self.executor.execute_tool(tool_call, context)

                    # Store the result in context and task
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

                    # Still store the error as result for tracking purposes
                    context[f"task_result_{self.task_id}"] = error_msg
                    self.executor.task_results[self.task_id] = {
                        "error": error_msg,
                        "status": "failed",
                    }

                return context

            def _get_execution_time(self) -> float:
                """Calculate execution time in seconds if available."""
                if not self.start_time or not self.end_time:
                    return 0

                from datetime import datetime

                start = datetime.fromisoformat(self.start_time)
                end = datetime.fromisoformat(self.end_time)
                return (end - start).total_seconds()

        # Create and add the task step
        task_step = TaskStep(task_id, task_type, tool, args, self.parent_agent)
        return self.add_node(task_step, task_id)


class ExecutorAgent(Agent):
    """An agent that executes tasks by creating and running sub-graphs."""

    def __init__(
        self,
        name: str,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 400,
        require_thought: Optional[bool] = False,
        request: ChatRequest = None,
    ):
        """An agent to execute tasks from the planner"""
        prompt = (
            get_prompt(
                config.PROMPT_PATH,
                config.PROMPT_AGENT_TYPE,
                config.PROMPT_EXECUTOR_AGENT,
            )
            if hasattr(config, "PROMPT_EXECUTOR_AGENT")
            else {
                "prompt": "You are an executor agent responsible for executing planned tasks efficiently."
            }
        )

        formatted_prompt = get_formatted_prompt(
            prompt=prompt.get(
                "prompt",
                "You are an executor agent responsible for executing planned tasks efficiently.",
            ),
            variables={},
        )

        class ExecutorResponse(BaseModel):
            summary: str = Field(description="A summary of task execution results")
            execution_status: str = Field(
                description="Status of execution: success, partial_success, or failure"
            )
            results: Dict[str, Any] = Field(
                description="Dictionary of task IDs and their results"
            )

        super().__init__(
            name=name,
            description="Agent to execute planned tasks using a graph-based workflow",
            role="executor",
            system_prompt=formatted_prompt,
            model=model,
            temperature=temperature,
            response_format=ExecutorResponse,
            require_thought=require_thought,
            max_tokens=max_tokens,
            # Enable any tool calls that this agent needs to execute tasks
            tool_calls=(
                config.AVAILABLE_TOOLS if hasattr(config, "AVAILABLE_TOOLS") else []
            ),
        )

        self.current_subgraph = None
        self.task_results = {}

    async def _build_execution_graph(self, tasks: List[Dict]) -> ExecutorSubGraph:
        """Build an execution graph from the planner's tasks."""
        # Create a new sub-graph for this execution
        graph_id = f"exec_{uuid.uuid4().hex[:6]}"
        subgraph = ExecutorSubGraph(graph_id, self)

        # Create nodes for all tasks
        for task in tasks:
            task_id = task.get("id")
            task_type = task.get("type", "tool")
            tool = task.get("tool", "")
            args = task.get("args", [])
            dependencies = task.get("dependencies", [])

            subgraph.add_task_node(task_id, task_type, tool, args, dependencies)

        # Add edges based on dependencies
        for task in tasks:
            task_id = task.get("id")
            dependencies = task.get("dependencies", [])

            for dep in dependencies:
                if dep in subgraph.steps_map:
                    subgraph.add_edge(dep, task_id, label="depends_on")
                else:
                    logger.warning(
                        f"Dependency {dep} for task {task_id} not found in graph"
                    )

        return subgraph

    async def execute_tasks(
        self,
        tasks: List[Dict],
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a list of tasks using a dynamically created execution graph."""
        logger.info(f"Building execution graph for {len(tasks)} tasks")

        # Reset task results for this execution
        self.task_results = {}

        # Build the execution graph
        execution_graph = await self._build_execution_graph(tasks)
        self.current_subgraph = execution_graph

        # Execute the graph
        if callback:
            await callback(
                "execution_start",
                {
                    "agent": self.name,
                    "task_count": len(tasks),
                    "execution_id": (
                        execution_graph.execution_id
                        if hasattr(execution_graph, "execution_id")
                        else str(uuid.uuid4())
                    ),
                },
            )

        try:
            # Execute the graph with the provided context
            context = await execution_graph.execute(context, openai_service, callback)

            # Summarize results
            results_summary = {
                "total_tasks": len(tasks),
                "completed_tasks": len(self.task_results),
                "results": self.task_results,
            }

            # Add result summary to context
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
        """Execute the executor agent.

        This method extracts tasks from the planner output, builds a graph, and executes it.
        """
        # First, call the base Agent.execute() to handle any initial processing
        context = await super().execute(context, openai_service, callback)

        # Check if planner output exists in the context
        planner_output = context.get("planner_agent_output")
        if not planner_output:
            logger.warning("No planner output found in context")
            if callback:
                await callback(
                    "step_complete",
                    {
                        "step": self.name,
                        "status": "skipped",
                        "message": "No planner output found",
                    },
                )
            return context

        try:
            # Parse the planner output to extract tasks
            if isinstance(planner_output, str):
                try:
                    planner_data = json.loads(planner_output)
                except json.JSONDecodeError:
                    # Handle case where output might not be JSON
                    logger.error("Failed to parse planner output as JSON")
                    return context
            else:
                planner_data = planner_output

            # Extract tasks from planner output
            tasks = planner_data.get("plan", [])
            if not tasks:
                logger.info("No tasks found in planner output")
                return context

            # Execute the tasks
            logger.info(f"Executing {len(tasks)} tasks from planner")
            context = await self.execute_tasks(tasks, context, openai_service, callback)

            # Generate a summary response using the task results
            response_data = {
                "summary": f"Executed {len(self.task_results)}/{len(tasks)} tasks",
                "execution_status": (
                    "success"
                    if len(self.task_results) == len(tasks)
                    else "partial_success"
                ),
                "results": self.task_results,
            }

            # Add response to context
            context[self.name] = json.dumps(response_data)
            context[f"{self.name}_output"] = json.dumps(response_data)
            self.result = json.dumps(response_data)

            return context

        except Exception as e:
            logger.error(f"Error in executor agent: {str(e)}")
            if callback:
                await callback("step_error", {"step": self.name, "error": str(e)})
            # Continue with execution rather than raising an exception
            return context
