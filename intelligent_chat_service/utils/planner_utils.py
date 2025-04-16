from typing import Dict, Any, List, Optional
from schema.graph_orchestrator import GraphNodeDefinition, NodeType, PlannerTask
from schema import Step, Tool
import uuid
from utils import logger


def create_step_from_tool(tool_name: str, parameters: Dict[str, Any]) -> Optional[Step]:
    """
    Create a step instance from a tool name and parameters.
    This function should be implemented to load and instantiate the appropriate tool.
    """
    try:
        # This is a placeholder. You'll need to implement the actual tool loading mechanism
        # based on your tool registry implementation
        from agents.tool_executor_agent import ToolExecutorAgent

        return ToolExecutorAgent(
            name=f"tool_executor_{uuid.uuid4().hex[:6]}",
            tool_name=tool_name,
            parameters=parameters,
        )
    except Exception as e:
        logger.error(f"Error creating step from tool {tool_name}: {str(e)}")
        return None


def tasks_to_graph_nodes(
    tasks: List[Dict[str, Any]], parent_node_id: Optional[str] = None
) -> List[GraphNodeDefinition]:
    """
    Convert planner tasks to graph node definitions.

    Args:
        tasks: List of tasks from planner output
        parent_node_id: Optional ID of the parent node (usually the planner)

    Returns:
        List of GraphNodeDefinition objects
    """
    graph_nodes = []

    for task_data in tasks:
        try:
            # Convert task data to a PlannerTask object
            task = PlannerTask(
                task_id=task_data.get("task_id", f"task_{uuid.uuid4().hex[:8]}"),
                tool=task_data.get("tool", ""),
                parameters=task_data.get("parameters", {}),
                description=task_data.get("description", ""),
                dependencies=task_data.get("dependencies", []),
            )

            # Create a step from the tool
            step = create_step_from_tool(task.tool, task.parameters)
            if not step:
                logger.error(
                    f"Failed to create step for task {task.task_id} using tool {task.tool}"
                )
                continue

            # Create dependencies list - always include parent if specified
            dependencies = list(task.dependencies)
            if parent_node_id and parent_node_id not in dependencies:
                dependencies.append(parent_node_id)

            # Create graph node definition
            node_def = GraphNodeDefinition(
                id=task.task_id,
                step=step,
                dependencies=dependencies,
                metadata={
                    "description": task.description,
                    "tool": task.tool,
                    "parameters": task.parameters,
                },
                node_type=NodeType.TOOL,
            )

            graph_nodes.append(node_def)
        except Exception as e:
            logger.error(f"Error converting task to graph node: {str(e)}")

    return graph_nodes
