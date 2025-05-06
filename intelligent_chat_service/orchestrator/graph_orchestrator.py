from typing import List, Dict, Any, Optional, Callable, Set
import asyncio
import time
from datetime import datetime
import networkx as nx
from core import OpenAIService
from utils import logger
from schema.graph_orchestrator import (
    GraphNode,
    NodeStatus,
    GraphNodeDefinition,
    GraphExecutionState,
    ExecutionStats,
    GraphSummary,
    ConditionalEdge,
)
import uuid


class GraphOrchestrator:
    """Orchestrates execution as a directed graph of nodes."""

    def __init__(
        self, name: str, description: str, nodes: List[GraphNodeDefinition] = None
    ):
        self.name = name
        self.description = description
        self.nodes: Dict[str, GraphNode] = {}
        self.execution_state = GraphExecutionState()
        self.graph = nx.DiGraph()
        self.dynamic_graph = nx.DiGraph()  # Graph that changes during execution

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
            conditional_edges=node_def.conditional_edges,
            priority=node_def.priority,
        )
        self.nodes[node_def.id] = graph_node
        self.execution_state.nodes[node_def.id] = graph_node
        self.execution_state.pending.add(node_def.id)

        # Add to networkx graph
        self.graph.add_node(node_def.id, node=graph_node)
        self.dynamic_graph.add_node(node_def.id, node=graph_node)

        # Add standard edges
        for dep_id in node_def.dependencies:
            # Add edge from dependency to the current node
            self.graph.add_edge(dep_id, node_def.id)
            self.dynamic_graph.add_edge(dep_id, node_def.id)

    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add a standard edge between two nodes."""
        # Verify that both nodes exist
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} does not exist")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} does not exist")

        # Add dependency relationship
        self.nodes[to_node_id].dependencies.add(from_node_id)
        self.nodes[from_node_id].dependents.add(to_node_id)

        # Add edge to graphs
        self.graph.add_edge(from_node_id, to_node_id)
        self.dynamic_graph.add_edge(from_node_id, to_node_id)

        logger.info(f"Added edge from {from_node_id} to {to_node_id}")

    def add_conditional_edge(
        self,
        from_node_id: str,
        to_node_id: str,
        condition: Callable[[Dict[str, Any]], bool],
        priority: int = 0,
    ) -> None:
        """Add a conditional edge between two nodes."""
        # Verify that both nodes exist
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} does not exist")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} does not exist")

        # Create the conditional edge
        edge = ConditionalEdge(
            target_node=to_node_id, condition=condition, priority=priority
        )

        # Add to source node's conditional edges
        self.nodes[from_node_id].conditional_edges.append(edge)

        logger.info(
            f"Added conditional edge from {from_node_id} to {to_node_id} with priority {priority}"
        )

    def validate_graph(self) -> bool:
        """Validate that the graph has no cycles and all dependencies exist."""
        # Check all dependencies exist
        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    raise ValueError(
                        f"Node {node_id} depends on non-existent node {dep_id}"
                    )

            # Check conditional edges
            for edge in node.conditional_edges:
                if edge.target_node not in self.nodes:
                    raise ValueError(
                        f"Node {node_id} has conditional edge to non-existent node {edge.target_node}"
                    )

        # Set up dependents based on dependencies
        for node_id, node in self.nodes.items():
            node.dependents.clear()  # Clear existing dependents

        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                self.nodes[dep_id].dependents.add(node_id)

        # Check for cycles using networkx in the static graph
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                cycle_str = " -> ".join(cycles[0] + [cycles[0][0]])
                raise ValueError(f"Cycle detected in graph: {cycle_str}")
        except nx.NetworkXNoCycle:
            # No cycles found - this is good
            pass

        return True

    def get_ready_nodes(self) -> Set[str]:
        """Get nodes that are ready to execute (all dependencies satisfied)."""
        ready_nodes = set()
        for node_id in self.execution_state.pending:
            node = self.nodes[node_id]
            deps_satisfied = all(
                dep_id in self.execution_state.completed
                or dep_id in self.execution_state.skipped
                for dep_id in node.dependencies
            )

            # Also check that any activated conditional dependencies are satisfied
            conditional_deps = self.execution_state.conditional_activations.get(
                node_id, []
            )
            conditional_deps_satisfied = all(
                dep_id in self.execution_state.completed
                or dep_id in self.execution_state.skipped
                for dep_id in conditional_deps
            )

            if deps_satisfied and conditional_deps_satisfied:
                ready_nodes.add(node_id)

        return ready_nodes

    def _evaluate_conditional_edges(
        self, node_id: str, context: Dict[str, Any]
    ) -> List[str]:
        """Evaluate conditional edges for a node and return activated target nodes."""
        node = self.nodes[node_id]
        activated_edges = []

        for edge in node.conditional_edges:
            try:
                if edge.condition(context):
                    activated_edges.append(edge.target_node)
                    logger.info(
                        f"Activated conditional edge from {node_id} to {edge.target_node}"
                    )

                    # Add the edge to the dynamic graph
                    self.dynamic_graph.add_edge(
                        node_id, edge.target_node, conditional=True
                    )

                    # Add dependency relationship for execution ordering
                    if (
                        edge.target_node
                        not in self.execution_state.conditional_activations
                    ):
                        self.execution_state.conditional_activations[
                            edge.target_node
                        ] = []

                    self.execution_state.conditional_activations[
                        edge.target_node
                    ].append(node_id)
            except Exception as e:
                logger.error(
                    f"Error evaluating condition for edge from {node_id} to {edge.target_node}: {str(e)}"
                )

        return activated_edges

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
                    "node_skipped",
                    {
                        "node_id": node_id,
                        "step_name": (
                            getattr(node.step, "name", node_id)
                            if hasattr(node, "step")
                            else node_id
                        ),
                    },
                )
            return context

        # Start executing the node
        node.status = NodeStatus.RUNNING
        node.stats.start_time = datetime.now()
        self.execution_state.pending.remove(node_id)
        self.execution_state.running.add(node_id)

        if callback:
            await callback(
                "node_started",
                {
                    "node_id": node_id,
                    "step_name": (
                        getattr(node.step, "name", node_id)
                        if hasattr(node, "step")
                        else node_id
                    ),
                },
            )

        try:
            # Execute the step associated with this node if it exists
            if hasattr(node, "step") and node.step:
                # Use the execute method of any object assigned to step
                if hasattr(node.step, "execute"):
                    context = await node.step.execute(context, openai_service, callback)
                    # Store result using name if available, otherwise use node_id
                    result_key = (
                        f"{node.step.name}_result"
                        if hasattr(node.step, "name")
                        else f"{node_id}_result"
                    )
                    node.result = context.get(result_key)
                else:
                    logger.warning(
                        f"Node {node_id} has a step but it doesn't have an execute method"
                    )
                    node.result = None
            else:
                # If no step is defined, just continue with current context
                node.result = None

            sorted_edges = sorted(
                node.conditional_edges, key=lambda edge: edge.priority, reverse=True
            )

            for edge in sorted_edges:
                try:
                    if edge.condition(context):
                        target_node = edge.target_node
                        target_node = self.nodes[target_node]

                        if (
                            target_node.status == NodeStatus.PENDING
                            and target_node not in self.execution_state.pending
                        ):
                            self.execution_state.pending.add(target_node)

                            if callback:
                                await callback(
                                    "conditional_edge_activated",
                                    {
                                        "from_node": node_id,
                                        "to_node": target_node,
                                        "priority": edge.priority,
                                    },
                                )
                except Exception as e:
                    logger.error(
                        f"Error evaluating conditional edges from {node_id} to {edge.target_node}: {str(e)}"
                    )

            # Mark as completed
            node.status = NodeStatus.COMPLETED
            node.stats.end_time = datetime.now()
            node.stats.duration_ms = (
                node.stats.end_time - node.stats.start_time
            ).total_seconds() * 1000
            self.execution_state.running.remove(node_id)
            self.execution_state.completed.add(node_id)
            self.execution_state.execution_order.append(node_id)

            # Evaluate conditional edges
            activated_edges = self._evaluate_conditional_edges(node_id, context)
            if activated_edges and callback:
                await callback(
                    "conditional_edges_activated",
                    {
                        "source_node": node_id,
                        "activated_edges": activated_edges,
                    },
                )

            if callback:
                await callback(
                    "node_completed",
                    {
                        "node_id": node_id,
                        "step_name": (
                            getattr(node.step, "name", node_id)
                            if hasattr(node, "step")
                            else node_id
                        ),
                        "duration_ms": node.stats.duration_ms,
                        "activated_edges": activated_edges,
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
                    {
                        "node_id": node_id,
                        "step_name": (
                            getattr(node.step, "name", node_id)
                            if hasattr(node, "step")
                            else node_id
                        ),
                        "error": str(e),
                    },
                )

            logger.error(f"Error executing node {node_id}: {str(e)}")

        return context

    def create_subgraph(
        self, name: str, description: str = None
    ) -> "GraphOrchestrator":
        """Create a subgraph with shared context that can be executed as part of this graph."""
        if description is None:
            description = f"Subgraph of {self.name}: {name}"

        subgraph = GraphOrchestrator(name=name, description=description)
        subgraph.parent_graph = self
        return subgraph

    def add_subgraph(
        self,
        subgraph: "GraphOrchestrator",
        entry_node_id: str = None,
        exit_node_id: str = None,
    ) -> None:
        """Add a subgraph to this graph, optionally connecting it to specific entry/exit nodes."""
        if not hasattr(subgraph, "parent_graph"):
            subgraph.parent_graph = self

        # Track the subgraphs for proper cleanup
        if not hasattr(self, "subgraphs"):
            self.subgraphs = []
        self.subgraphs.append(subgraph)

        # If entry and exit nodes are specified, we could add extra connections
        # This is left as a future enhancement

    def build_execution_subgraph(
        self, tasks: List[Dict[str, Any]], task_executor_factory: Callable
    ) -> "GraphOrchestrator":
        """Build a subgraph for task execution from a list of task definitions.

        Args:
            tasks: List of task definitions with id, type, dependencies, etc.
            task_executor_factory: A function that creates a task executor for each task

        Returns:
            A configured execution subgraph
        """
        graph_id = f"exec_{uuid.uuid4().hex[:6]}"
        subgraph = self.create_subgraph(
            name=graph_id, description=f"Execution subgraph for {len(tasks)} tasks"
        )

        # Add task nodes to the subgraph
        for task in tasks:
            task_id = task.get("id")
            dependencies = task.get("dependencies", [])

            # Create task executor using the provided factory
            task_executor = task_executor_factory(task)

            # Create and add the node
            task_node = GraphNodeDefinition(
                id=task_id,
                step=task_executor,
                dependencies=dependencies,
                priority=task.get("priority", 5),
                metadata=task.get("metadata", {}),
            )

            subgraph.add_node(task_node)

        return subgraph

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the graph orchestrator."""
        # Use parent's context if this is a subgraph
        if hasattr(self, "parent_graph") and self.parent_graph is not None:
            # We're operating as a subgraph, so we should share the context
            logger.info(f"Executing subgraph {self.name} with shared context")
        else:
            # We're the main graph, create a copy of the context
            context = dict(context)

        start_time = time.time()

        # Validate the graph
        self.validate_graph()

        # Reset the dynamic graph for this execution
        self.dynamic_graph = self.graph.copy()

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
        # Clean up any subgraphs
        if hasattr(self, "subgraphs") and self.subgraphs:
            for subgraph in self.subgraphs:
                subgraph.cleanup()

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
                    "label": (
                        getattr(node.step, "name", node_id)
                        if hasattr(node, "step")
                        else node_id
                    ),
                    "status": node.status.value,
                    "metadata": node.metadata,
                }
            )

            # Add standard edges
            for dep_id in node.dependencies:
                edges.append({"from": dep_id, "to": node_id, "type": "static"})

            # Add any active conditional edges from the dynamic graph
            for u, v, data in self.dynamic_graph.edges(data=True):
                if u == node_id and data.get("conditional", False):
                    edges.append(
                        {"from": u, "to": v, "type": "conditional", "active": True}
                    )

        # Calculate stats
        stats = {
            "total": len(self.nodes),
            "pending": len(self.execution_state.pending),
            "running": len(self.execution_state.running),
            "completed": len(self.execution_state.completed),
            "error": len(self.execution_state.error),
            "skipped": len(self.execution_state.skipped),
            "conditional_activations": len(
                self.execution_state.conditional_activations
            ),
        }

        return GraphSummary(nodes=nodes, edges=edges, stats=stats)

    def cleanup(self):
        """Clean up any resources."""
        # Clean up subgraphs
        if hasattr(self, "subgraphs") and self.subgraphs:
            for subgraph in self.subgraphs:
                subgraph.cleanup()

        # Clear references
        if hasattr(self, "parent_graph"):
            self.parent_graph = None
