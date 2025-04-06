from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from core import OpenAIService, get_openai_service
from utils import logger, export_graph_to_dot, export_graph_to_json
import uuid
from datetime import datetime
import json

graph_router = APIRouter(prefix="/graph", tags=["Graph"])


# Registry to keep track of graph orchestrators
graph_registry = {}


class GraphDefinition(BaseModel):
    """API schema for graph definition."""

    name: str
    description: str
    nodes: List[Dict[str, Any]]


@graph_router.post("/register")
async def register_graph(graph_def: GraphDefinition):
    """Register a graph orchestrator for later use."""
    graph_id = f"graph_{uuid.uuid4().hex[:8]}"
    graph_registry[graph_id] = {
        "definition": graph_def.dict(),
        "created_at": datetime.now().isoformat(),
        "status": "registered",
    }

    return {
        "graph_id": graph_id,
        "message": "Graph registered successfully",
    }


@graph_router.get("/list")
async def list_graphs():
    """List all registered graph orchestrators."""
    return {
        "graphs": [
            {
                "graph_id": graph_id,
                "name": graph_data["definition"]["name"],
                "created_at": graph_data["created_at"],
                "status": graph_data["status"],
            }
            for graph_id, graph_data in graph_registry.items()
        ]
    }


@graph_router.get("/{graph_id}")
async def get_graph(graph_id: str):
    """Get details about a specific graph."""
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph not found")

    return graph_registry[graph_id]


@graph_router.get("/{graph_id}/visualize")
async def visualize_graph(
    graph_id: str, format: str = Query("json", enum=["json", "dot"])
):
    """Visualize a graph in the specified format."""
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph not found")

    graph_data = graph_registry[graph_id]

    # Extract nodes and edges from definition
    nodes = []
    edges = []

    for node in graph_data["definition"]["nodes"]:
        nodes.append(
            {
                "id": node["id"],
                "label": node.get("name", node["id"]),
                "status": "pending",
                "metadata": node.get("metadata", {}),
            }
        )

        for dep_id in node.get("dependencies", []):
            edges.append({"from": dep_id, "to": node["id"]})

    # Return in requested format
    if format == "dot":
        dot_content = export_graph_to_dot(nodes, edges)
        return {"dot": dot_content}
    else:
        return {"graph": {"nodes": nodes, "edges": edges}}


@graph_router.delete("/{graph_id}")
async def delete_graph(graph_id: str):
    """Delete a registered graph."""
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph not found")

    del graph_registry[graph_id]
    return {"message": f"Graph {graph_id} deleted successfully"}
