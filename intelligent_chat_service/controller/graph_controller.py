from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List
from utils import logger, export_graph_to_dot, export_graph_to_json
import uuid
from datetime import datetime

graph_router = APIRouter(prefix="/graph", tags=["graph"])

graph_registry = {}


class GraphDefinition(BaseModel):
    name: str
    description: str
    nodes: list[dict[str, Any]]


@graph_router.post("/register")
async def register_graph(graph_def: GraphDefinition):
    """Register a graph orchestrator for later use."""
    # todo: register using yaml file
    graph_id = f"graph_{uuid.uuid4().hex[:8]}"
    graph_registry[graph_id] = {
        "definition": graph_def.model_dump(),
        "created_at": datetime.now().isoformat(),
        "status": "registered",
    }

    return {"graph_id": graph_id, "message": "Graph registered successfully"}


@graph_router.get("/list")
async def list_graphs():
    """List all registered orchestrators"""
    return {
        "graph": [
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
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph not found.")

    return graph_registry[graph_id]
