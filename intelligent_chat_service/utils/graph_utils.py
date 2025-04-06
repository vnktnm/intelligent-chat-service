from typing import List, Dict, Any
from schema.graph_orchestrator import GraphNodeDefinition
import json
import networkx as nx
from utils import logger


def validate_graph_definition(nodes: List[GraphNodeDefinition]) -> bool:
    """Validate graph definition has no cycles and all dependencies exist."""
    # Check all dependencies exist
    node_ids = {node.id for node in nodes}
    for node in nodes:
        for dep_id in node.dependencies:
            if dep_id not in node_ids:
                raise ValueError(
                    f"Node {node.id} depends on non-existent node {dep_id}"
                )

    # Build a graph using networkx for cycle detection
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.id)
        for dep_id in node.dependencies:
            G.add_edge(dep_id, node.id)  # Edge direction: dependency -> dependent

    # Check for cycles
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            cycle_str = " -> ".join(cycles[0] + [cycles[0][0]])
            raise ValueError(f"Cycle detected in graph: {cycle_str}")
    except nx.NetworkXNoCycle:
        pass

    return True


def get_execution_order(nodes: List[GraphNodeDefinition]) -> List[str]:
    """Get a valid execution order for nodes based on dependencies."""
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.id)
        for dep_id in node.dependencies:
            G.add_edge(dep_id, node.id)  # Edge direction: dependency -> dependent

    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        raise ValueError("Graph contains cycles and cannot be topologically sorted")


def export_graph_to_dot(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> str:
    """Export graph to DOT format for visualization."""
    dot_str = "digraph G {\n"

    # Add nodes
    for node in nodes:
        status = node.get("status", "pending")
        color = {
            "pending": "gray",
            "running": "blue",
            "completed": "green",
            "error": "red",
            "skipped": "orange",
        }.get(status, "black")

        dot_str += f'  "{node["id"]}" [label="{node.get("label", node["id"])}", style=filled, fillcolor={color}];\n'

    # Add edges
    for edge in edges:
        dot_str += f'  "{edge["from"]}" -> "{edge["to"]}";\n'

    dot_str += "}"
    return dot_str


def export_graph_to_json(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> str:
    """Export graph to JSON format for visualization."""
    data = {"nodes": nodes, "edges": edges}
    return json.dumps(data)
