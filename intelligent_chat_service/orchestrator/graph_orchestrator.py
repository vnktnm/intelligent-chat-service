from typing import List, Dict, Any, Optional, Callable, Set
import asyncio
import time
from datetime import datetime
from core import OpenAIService
from utils import logger
from schema import Step
from schema.graph_orchestrator import (
    GraphNode,
    NodeStatus,
    GraphNodeDefinition,
    GraphExecutionState,
    ExecutionStats,
    GraphSummary,
)
from orchestrator import Orchestrator


class GraphOrchestrator(Orchestrator):
    """Orchestrates execution as a directed graph of nodes."""

    def __init__(
        self, name: str, description: str, nodes: List[GraphNodeDefinition] = None
    ):
        super().__init__(name, description, [])
        self.nodes: Dict[str, GraphNode] = {}
        self.execution_state = GraphExecutionState()

        # Register nodes
        if nodes:
            for node_def in nodes:
                self.add_node(node_def)

    def add_node(self, node_def: GraphNodeDefinition) -> None:
        """Add a node to the graph orchestrator."""
        graph_node = GraphNode(
            id=node_def.id,
            step=node_def.step,
            dependencies=set(node_def.dependencies),
            dependents=set(),
            condition=node_def.condition,
            metadata=node_def.metadata,
        )
        self.nodes[node_def.id] = graph_node
        self.execution_state.nodes[node_def.id] = graph_node
        self.execution_state.pending.add(node_def.id)

    def validate_graph(self) -> bool:
        """Validate that the graph has no cycles and all dependencies exist."""
        # Check all dependencies exist
        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    raise ValueError(
                        f"Node {node_id} depends on non-existent node {dep_id}"
                    )

        # Set up dependents based on dependencies
        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                self.nodes[dep_id].dependents.add(node_id)

        # Check for cycles using DFS
        visited = set()
        path = set()

        def has_cycle(node_id):
            if node_id in path:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            path.add(node_id)
            for dep_id in self.nodes[node_id].dependents:
                if has_cycle(dep_id):
                    return True
            path.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    raise ValueError(
                        f"Cycle detected in graph involving node {node_id}"
                    )

        return True

    def get_ready_nodes(self) -> Set[str]:
        """Get nodes that are ready to execute (all dependencies satisfied)."""
        ready_nodes = set()
        for node_id in self.execution_state.pending:
            node = self.nodes[node_id]
            if all(
                dep_id in self.execution_state.completed
                or dep_id in self.execution_state.skipped
                for dep_id in node.dependencies
            ):
                ready_nodes.add(node_id)
        return ready_nodes

    async def execute_node(
        self,
        node_id: str,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a single node in the graph."""
        node = self.nodes[node_id]

        # Check if node should be executed based on condition
        should_execute = True
        if node.condition:
            try:
                should_execute = node.condition(context)
            except Exception as e:
                logger.error(f"Error evaluating condition for node {node_id}: {str(e)}")
                should_execute = False

        if not should_execute:
            # Skip this node
            node.status = NodeStatus.SKIPPED
            self.execution_state.pending.remove(node_id)
            self.execution_state.skipped.add(node_id)
            self.execution_state.execution_order.append(node_id)

            if callback:
                await callback(
                    "node_skipped", {"node_id": node_id, "step_name": node.step.name}
                )
            return context

        # Start executing the node
        node.status = NodeStatus.RUNNING
        node.stats.start_time = datetime.now()
        self.execution_state.pending.remove(node_id)
        self.execution_state.running.add(node_id)

        if callback:
            await callback(
                "node_started", {"node_id": node_id, "step_name": node.step.name}
            )

        try:
            # Execute the step associated with this node
            context = await node.step.execute(context, openai_service, callback)
            node.result = context.get(f"{node.step.name}_result")

            # Mark as completed
            node.status = NodeStatus.COMPLETED
            node.stats.end_time = datetime.now()
            node.stats.duration_ms = (
                node.stats.end_time - node.stats.start_time
            ).total_seconds() * 1000
            self.execution_state.running.remove(node_id)
            self.execution_state.completed.add(node_id)
            self.execution_state.execution_order.append(node_id)

            if callback:
                await callback(
                    "node_completed",
                    {
                        "node_id": node_id,
                        "step_name": node.step.name,
                        "duration_ms": node.stats.duration_ms,
                    },
                )

        except Exception as e:
            # Handle error
            node.status = NodeStatus.ERROR
            node.stats.end_time = datetime.now()
            node.stats.duration_ms = (
                node.stats.end_time - node.stats.start_time
            ).total_seconds() * 1000
            node.stats.error = str(e)
            self.execution_state.running.remove(node_id)
            self.execution_state.error.add(node_id)
            self.execution_state.execution_order.append(node_id)

            if callback:
                await callback(
                    "node_error",
                    {"node_id": node_id, "step_name": node.step.name, "error": str(e)},
                )

            logger.error(f"Error executing node {node_id}: {str(e)}")

        return context

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the graph orchestrator."""
        context = dict(context)
        start_time = time.time()

        # Validate the graph
        self.validate_graph()

        if callback:
            await callback(
                "orchestrator_start",
                {
                    "name": self.name,
                    "description": self.description,
                    "total_nodes": len(self.nodes),
                    "type": "graph",
                },
            )

        logger.info(
            f"Starting graph orchestration {self.name} with {len(self.nodes)} nodes."
        )

        # Reset execution state
        self.execution_state = GraphExecutionState()
        for node_id, node in self.nodes.items():
            node.status = NodeStatus.PENDING
            node.stats = ExecutionStats()
            self.execution_state.nodes[node_id] = node
            self.execution_state.pending.add(node_id)

        # Execute nodes in dependency order
        while self.execution_state.pending or self.execution_state.running:
            # Find nodes that are ready to execute
            ready_nodes = self.get_ready_nodes()

            if not ready_nodes and not self.execution_state.running:
                # No ready nodes and nothing running means we can't make progress
                remaining = ", ".join(self.execution_state.pending)
                logger.error(f"Graph execution deadlock. Remaining nodes: {remaining}")
                if callback:
                    await callback(
                        "orchestrator_error",
                        {
                            "error": f"Execution deadlock. Nodes {remaining} cannot be executed."
                        },
                    )
                break

            # Execute ready nodes in parallel
            tasks = []
            for node_id in ready_nodes:
                tasks.append(
                    self.execute_node(node_id, context, openai_service, callback)
                )

            if tasks:
                await asyncio.gather(*tasks)
            elif self.execution_state.running:
                # Wait for running nodes to complete
                await asyncio.sleep(0.1)

        # Calculate statistics
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        success = len(self.execution_state.error) == 0

        if callback:
            await callback(
                "orchestrator_complete",
                {
                    "name": self.name,
                    "description": self.description,
                    "duration_ms": duration_ms,
                    "success": success,
                    "nodes_total": len(self.nodes),
                    "nodes_completed": len(self.execution_state.completed),
                    "nodes_error": len(self.execution_state.error),
                    "nodes_skipped": len(self.execution_state.skipped),
                    "execution_order": self.execution_state.execution_order,
                },
            )

        logger.info(f"Completed Graph Orchestrator: {self.name}. Success: {success}")
        return context

    def get_graph_summary(self) -> GraphSummary:
        """Get a summary of the graph for visualization."""
        nodes = []
        edges = []

        # Add nodes
        for node_id, node in self.nodes.items():
            nodes.append(
                {
                    "id": node_id,
                    "label": node.step.name,
                    "status": node.status.value,
                    "metadata": node.metadata,
                }
            )

            # Add edges
            for dep_id in node.dependencies:
                edges.append({"from": dep_id, "to": node_id})

        # Calculate stats
        stats = {
            "total": len(self.nodes),
            "pending": len(self.execution_state.pending),
            "running": len(self.execution_state.running),
            "completed": len(self.execution_state.completed),
            "error": len(self.execution_state.error),
            "skipped": len(self.execution_state.skipped),
        }

        return GraphSummary(nodes=nodes, edges=edges, stats=stats)

    def cleanup(self):
        """Clean up any resources."""
        super().cleanup()
        # Clean up any additional resources specific to graph orchestrator
