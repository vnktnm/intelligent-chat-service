import networkx as nx
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from schema import Step
from core import OpenAIService
from utils import logger
from orchestrator.orchestrator import Orchestrator
import uuid
import json
from datetime import datetime


class GraphOrchestrator(Orchestrator):
    """Graph-based orchestrator that uses NetworkX for workflow execution."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.graph = nx.DiGraph()
        self.steps_map = {}  # Maps step_id to Step object
        self.execution_history = []
        self.current_execution = None

    def add_node(self, step: Step, node_id: Optional[str] = None) -> str:
        """Add a node to the graph with the given step.

        Args:
            step (Step): The step to add to the graph
            node_id (Optional[str]): Optional ID for the node. If not provided, step.name is used.

        Returns:
            str: The ID of the added node
        """
        node_id = node_id or step.name
        if node_id in self.steps_map:
            raise ValueError(f"Node with ID {node_id} already exists in the graph")

        self.steps_map[node_id] = step
        self.graph.add_node(node_id, step=step)
        return node_id

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        label: Optional[str] = None,
    ) -> None:
        """Add an edge between nodes, optionally with a condition.

        Args:
            from_node (str): Source node ID
            to_node (str): Target node ID
            condition (Optional[Callable]): Optional condition function that takes
                                        context and returns True/False
            label (Optional[str]): Optional label for the edge
        """
        if from_node not in self.steps_map:
            raise ValueError(f"Source node {from_node} does not exist")
        if to_node not in self.steps_map:
            raise ValueError(f"Target node {to_node} does not exist")

        self.graph.add_edge(from_node, to_node, condition=condition, label=label or "")

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        start_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow graph.

        Args:
            context (Dict[str, Any]): Initial context for execution
            openai_service (OpenAIService): Service for OpenAI interactions
            callback (Optional[Callable]): Callback for execution events
            start_nodes (Optional[List[str]]): Optional list of starting nodes.
                                              If None, uses nodes with no predecessors.

        Returns:
            Dict[str, Any]: The final context after execution
        """
        if not self.graph.nodes:
            logger.warning(f"No nodes in orchestration graph for {self.name}")
            return context

        context = dict(context)
        execution_id = str(uuid.uuid4())
        self.execution_id = execution_id
        self.current_execution = {
            "id": execution_id,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "nodes_visited": [],
            "edges_traversed": [],
        }

        if callback:
            await callback(
                "orchestrator_start",
                {
                    "name": self.name,
                    "description": self.description,
                    "execution_id": execution_id,
                    "node_count": self.graph.number_of_nodes(),
                    "edge_count": self.graph.number_of_edges(),
                },
            )

        logger.info(
            f"Starting graph orchestration {self.name} with execution_id: {execution_id}"
        )

        # Determine start nodes if not specified
        if not start_nodes:
            # Find nodes with no predecessors
            start_nodes = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
            if not start_nodes:
                # If there are no nodes without predecessors, we might be dealing with a cycle
                # In this case, pick the first node
                start_nodes = [list(self.graph.nodes)[0]]

        visited = set()
        queue = start_nodes.copy()

        while queue:
            current_node_id = queue.pop(0)

            if current_node_id in visited:
                continue

            visited.add(current_node_id)
            self.current_execution["nodes_visited"].append(current_node_id)

            step = self.steps_map[current_node_id]

            logger.info(f"Executing node: {current_node_id}")

            if callback:
                await callback(
                    "node_start",
                    {
                        "node_id": current_node_id,
                        "step_name": step.name,
                        "execution_id": execution_id,
                    },
                )

            try:
                context = await step.execute(context, openai_service, callback)

                if callback:
                    await callback(
                        "node_complete",
                        {
                            "node_id": current_node_id,
                            "step_name": step.name,
                            "execution_id": execution_id,
                        },
                    )

                # Process outgoing edges
                for _, next_node_id, edge_data in self.graph.out_edges(
                    current_node_id, data=True
                ):
                    condition = edge_data.get("condition")

                    # Check edge condition if it exists
                    if condition and not condition(context):
                        logger.info(
                            f"Skipping edge {current_node_id} -> {next_node_id} due to condition"
                        )
                        continue

                    self.current_execution["edges_traversed"].append(
                        (current_node_id, next_node_id)
                    )

                    if callback:
                        await callback(
                            "edge_traversed",
                            {
                                "from_node": current_node_id,
                                "to_node": next_node_id,
                                "label": edge_data.get("label", ""),
                                "execution_id": execution_id,
                            },
                        )

                    # Check if all incoming edges to the next node have been traversed
                    if next_node_id not in queue and next_node_id not in visited:
                        # For simplicity, we'll add the node to the queue
                        # In a more complex implementation, you might want to check if all required
                        # predecessor nodes have been visited
                        queue.append(next_node_id)

            except Exception as e:
                logger.error(f"Error in node {current_node_id}: {str(e)}")
                if callback:
                    await callback(
                        "node_error",
                        {
                            "node_id": current_node_id,
                            "step_name": step.name,
                            "error": str(e),
                            "execution_id": execution_id,
                        },
                    )

                # We could decide to halt execution here by clearing the queue
                # queue = []
                # Or we could continue with other nodes:
                continue

        self.current_execution["status"] = "completed"
        self.current_execution["completed_at"] = datetime.now().isoformat()
        self.execution_history.append(self.current_execution)

        if callback:
            await callback(
                "orchestrator_complete",
                {
                    "name": self.name,
                    "description": self.description,
                    "execution_id": execution_id,
                    "visited_nodes": list(visited),
                },
            )

        logger.info(
            f"Completed Graph Orchestration: {self.name}, visited {len(visited)} nodes"
        )
        return context

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history of this orchestrator."""
        return self.execution_history

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific execution by ID."""
        for execution in self.execution_history:
            if execution["id"] == execution_id:
                return execution
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a dictionary representation for API endpoints."""
        nodes = []
        for node_id in self.graph.nodes:
            step = self.steps_map[node_id]
            nodes.append(
                {
                    "id": node_id,
                    "name": step.name,
                    "type": step.__class__.__name__,
                }
            )

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append(
                {
                    "from": u,
                    "to": v,
                    "label": data.get("label", ""),
                    "has_condition": data.get("condition") is not None,
                }
            )

        return {
            "name": self.name,
            "description": self.description,
            "nodes": nodes,
            "edges": edges,
        }

    def export_graphviz(self) -> str:
        """Export the graph in DOT format for visualization."""
        try:
            import pydot
            from networkx.drawing.nx_pydot import write_dot

            # Create a temporary graph for visualization
            viz_graph = nx.DiGraph()

            # Add all nodes with additional attributes for better visualization
            for node_id in self.graph.nodes:
                step = self.steps_map[node_id]
                node_type = step.__class__.__name__
                viz_graph.add_node(
                    node_id,
                    label=f"{node_id}\n({node_type})",
                    shape="box",
                    style="filled",
                    fillcolor=self._get_node_color(node_type),
                )

            # Add all edges with labels
            for u, v, data in self.graph.edges(data=True):
                label = data.get("label", "")
                edge_attrs = {"label": label}

                if data.get("condition"):
                    label = f"[Conditional] {label}"
                    edge_attrs.update(
                        {"style": "dashed", "color": "blue", "label": label}
                    )

                viz_graph.add_edge(u, v, **edge_attrs)

            # Return DOT format as string
            return nx.nx_pydot.to_pydot(viz_graph).to_string()

        except ImportError:
            logger.warning("pydot not installed, cannot export graph visualization")
            return "Error: pydot not installed"

    def _get_node_color(self, node_type: str) -> str:
        """Return a color for visualization based on node type."""
        color_map = {
            "AnalyzerAgent": "lightblue",
            "PlannerAgent": "lightgreen",
            "ExecutorAgent": "lightyellow",
            "TaskStep": "lightcoral",
        }
        return color_map.get(node_type, "white")

    def export_html(self) -> str:
        """Export the graph as HTML for web visualization using vis.js."""
        nodes_data = []
        for node_id in self.graph.nodes:
            step = self.steps_map[node_id]
            node_type = step.__class__.__name__
            nodes_data.append(
                {
                    "id": node_id,
                    "label": f"{node_id}\n({node_type})",
                    "title": step.description,
                    "color": self._get_node_color(node_type),
                }
            )

        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            label = data.get("label", "")
            has_condition = data.get("condition") is not None

            edge = {"from": u, "to": v, "label": label, "arrows": "to"}

            if has_condition:
                edge["dashes"] = True
                edge["color"] = {"color": "blue"}

            edges_data.append(edge)

        # Create a simple HTML visualization using vis.js
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Visualization</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style type="text/css">
                #mynetwork {
                    width: 800px;
                    height: 600px;
                    border: 1px solid lightgray;
                }
            </style>
        </head>
        <body>
            <div id="mynetwork"></div>
            <script type="text/javascript">
                const nodes = new vis.DataSet(%s);
                const edges = new vis.DataSet(%s);
                
                const container = document.getElementById('mynetwork');
                const data = {
                    nodes: nodes,
                    edges: edges
                };
                const options = {
                    nodes: {
                        shape: 'box',
                        font: {
                            size: 14
                        }
                    },
                    edges: {
                        font: {
                            size: 12
                        }
                    },
                    physics: {
                        enabled: true,
                        hierarchicalRepulsion: {
                            centralGravity: 0.0,
                            springLength: 100,
                            springConstant: 0.01,
                            nodeDistance: 120
                        },
                        solver: 'hierarchicalRepulsion'
                    },
                    layout: {
                        hierarchical: {
                            direction: 'LR',
                            sortMethod: 'directed'
                        }
                    }
                };
                const network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """ % (
            json.dumps(nodes_data),
            json.dumps(edges_data),
        )

        return html
