from .agent import Agent
from typing import Optional, Any, Callable
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
from schema import Step


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
        args: list[Any],
        dependencies: list[str],
        capabilities: list[str],
    ) -> str:
        """Add a task node to the execution graph"""

        class TaskStep(Step):
            def __init__(
                self,
                task_id: str,
                task_type: str,
                tool: str,
                args: list[Any],
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
                context: dict[str, Any],
                openai_service: OpenAIService,
                callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
            ) -> dict[str, Any]:
                """Agent executor"""
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

        task_step = TaskStep(task_id, task_type, tool, args, self.parent_agent)

        task_step_node = GraphNodeDefinition(
            id=task_id,
            step=task_step,
            dependencies=[],
            priority=5,
            metadata={"capabilities": capabilities},
        )

        return self.add_node(task_step_node)


class ExecutorAgent(Agent):
    """An agent that analyzes the executes based on the generated plan."""

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

    async def _build_execution_graph(self, tasks: list[dict]) -> ExecutorSubGraph:
        graph_id = f"exec_{uuid.uuid4().hex[:6]}"
        subgraph = ExecutorSubGraph(graph_id, self)

        for task in tasks:
            task_id = task.get("id")
            task_type = task.get("type", "tool")
            tool = task.get("tool", "")
            args = task.get("args", [])
            dependencies = task.get("dependencies", [])
            capabilities = task.get("capabilities", [])

            subgraph.add_task_node(
                task_id, task_type, tool, args, dependencies, capabilities
            )

        for task in tasks:
            task_id = task.get("id")
            dependencies = task.get("dependencies", [])

            for dep in dependencies:
                if dep in subgraph.steps:
                    subgraph.graph.add_edge(dep, task_id)
                    subgraph.dynamic_graph.add_edge(dep, task_id)
                else:
                    logger.warning(
                        f"Dependencies {dep} for task {task_id} not found in graph."
                    )

        return subgraph

    async def execute_tasks(
        self,
        tasks: list[dict],
        context: dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """Execute a list of tasks"""
        logger.info(f"Building execution graph for {len(tasks)} tasks")

        self.task_results = {}

        execution_graph = await self._build_execution_graph(tasks)
        self.current_subgraph = execution_graph

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
            context = await execution_graph.execute(context, openai_service, callback)

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
        context: dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ):
        """Executes the exector agent"""

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
            if isinstance(planner_output, str):
                try:
                    planner_data = json.loads(planner_output)
                except json.JSONDecodeError:
                    logger.error("Failed to parse planner output as JSON")
                    return context
            else:
                planner_data = planner_output

            tasks = planner_data.get("plan", [])
            logger.info(f"Tasks: {tasks}")
            if not tasks:
                logger.info(f"No tasks found in planner output")
                return context

            logger.info(f"Executing {len(tasks)} tasks from planner")
            context = await self.execute_tasks(tasks, context, openai_service, callback)

            response_data = {
                "summary": f"Executed {len(self.task_results)}/{len(tasks)} tasks",
                "execution_status": (
                    "success"
                    if len(self.task_results) == len(tasks)
                    else "partial_success"
                ),
                "results": self.task_results,
            }

            context[self.name] = json.dumps(response_data)
            context[f"{self.name}_output"] = json.dumps(response_data)
            self.result = json.dumps(response_data)

            return context
        except Exception as e:
            logger.error(f"Error in executor agent: {str(e)}")
            if callback:
                await callback("step_error", {"step": self.name, "error": str(e)})
            return context
